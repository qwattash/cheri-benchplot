import os
import re
import shlex
import signal
import subprocess
import time
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from queue import Queue
from threading import Condition, Lock, Semaphore, Thread
from uuid import UUID, uuid4

from paramiko.client import SSHClient, SSHException, WarningPolicy

from .config import InstanceConfig
from .scheduler import ResourceManager
from .util import new_logger


class TimeoutError(RuntimeError):
    """
    Special exception type to signal timeouts.
    """

    pass


class InstanceState(IntEnum):
    """
    State of an instance
    DORMANT: initialized but not started booting
    BOOTING: boot in process
    READY: ready to accept work
    SHUTDOWN: shutdown in process
    DEAD: shutdown completed or killed because of error
    """

    DORMANT = 0
    BOOTING = 1
    SETUP = 2
    READY = 3
    SHUTDOWN = 4
    DEAD = 5


class MsgType(Enum):
    #: collect command stdout
    OUT = "out"
    #: collect command stderr
    ERR = "err"
    #: switch instance state
    NEW_STATE = "new-state"
    #: command exited
    EXITED = "cmd-exited"
    #: command failed
    FAILED = "cmd-failed"


@dataclass
class Msg:
    """
    Helper message enqueued by command runners into an instance watch thread
    """

    msgtype: MsgType
    origin: str
    data: str

    def __eq__(self, other):
        return (
            self.msgtype == other.msgtype
            and self.origin == other.origin
            and self.data == other.data
        )

    def __post_init__(self):
        # Ensure that the lines are decoded in stream messages
        if self.msgtype == MsgType.OUT or self.msgtype == MsgType.ERR:
            try:
                self.data = self.data.decode()
            except (UnicodeError, AttributeError):
                pass
            self.data = self.data.strip()


class CommandRunner(ABC):
    """
    Base class for command runners.
    These helpers handle the I/O from subprocesses or ssh commands, the output
    is enqueued to the instance message queue.
    Using multiple threads is annoying but we can't always poll() so we may just as well never do that.
    """

    def __init__(self, logger, msg_queue):
        self.logger = logger
        #: Target output message queue
        self._queue = msg_queue
        #: I/O worker threads.
        self._stdout_thread = None
        self._stderr_thread = None
        #: Command executed
        self._cmd_name = None

    def _watch_thread(self, stream, msgtype):
        try:
            self._watch_loop(stream, msgtype)
        except Exception as ex:
            self.logger.error(
                "Command %s watch thread %s died: %s", self._cmd_name, msgtype, ex
            )
            self._queue.put(Msg(MsgType.FAILED, self._cmd_name, ex))
            raise

    def _watch_loop(self, stream, msgtype):
        assert self._cmd_name is not None, (
            "The run method should have set self._cmd_name"
        )
        for line in stream:
            self._queue.put(Msg(msgtype, self._cmd_name, line))
        # Wait to ensure an exit code
        self._ensure_exited()
        self.logger.debug(
            "Command %s exited %s watch thread with %s",
            self._cmd_name,
            msgtype,
            self._check_exited(),
        )
        # Only emit the EXITED event from one of the watch loops,
        # assuming the other will also exit soon
        if msgtype == MsgType.OUT:
            self._queue.put(Msg(MsgType.EXITED, self._cmd_name, self))

    @abstractmethod
    def _check_exited(self) -> int | None:
        """
        Check whether the underlying command has exited and return a return code.
        This is used by the watch loop to parameterize the loop exit condition
        """
        ...

    def run(
        self,
        cmd: str | Path,
        args: list[str] = None,
        env: dict[str, str] = None,
        cmd_id: str | None = None,
    ):
        """
        Run the command and schedule the watch thread.
        This is responsible for starting the watch thread and passing the required arguments.

        :param cmd: The command to run
        :param args: List of arguments
        :param env: Environment variables
        """
        if cmd_id is None:
            cmd_id = cmd
        self._cmd_name = cmd_id

    @abstractmethod
    def stop(self):
        """
        Stop the command and the watch thread.
        This should wait synchronously for the command to stop.
        """
        ...

    @abstractmethod
    def _ensure_exited(self) -> int:
        """
        Wait for the command to exit and produce an exit code.
        """
        ...

    def wait(self, timeout=None):
        """
        Wait for the command to complete and watch threads to exit
        """
        self._stdout_thread.join(timeout=timeout)
        self._stderr_thread.join(timeout=timeout)
        if self._stdout_thread.is_alive() or self._stderr_thread.is_alive():
            raise TimeoutError(
                f"Timed out waiting for command {self._cmd_name} to complete"
            )


class HostCommandRunner(CommandRunner):
    """
    Helper to execute a command locally and react to I/O.
    """

    def __init__(self, logger, msg_queue):
        super().__init__(logger, msg_queue)
        self.logger = new_logger("hostcmd", parent=logger)
        #: Command subprocess
        self._subp = None

    def _check_exited(self):
        assert self._subp is not None
        return self._subp.returncode

    def _ensure_exited(self):
        assert self._subp is not None
        self._subp.wait()
        return self._subp.returncode

    def wait(self, timeout=None):
        if self._subp is None:
            self.logger.error("wait() called on command that is not running")
            return
        super().wait(timeout=timeout)

    def run(self, cmd, args=None, env=None, cmd_id=None):
        super().run(cmd, args, env, cmd_id)
        args = args or []
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        self.logger.debug("Exec local: %s %s", cmd, args)
        self._subp = subprocess.Popen(
            [str(cmd)] + args,
            env=full_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.logger.debug(
            "Spawned %s pid=%d pgid=%d", cmd, self._subp.pid, os.getpgid(self._subp.pid)
        )
        self._stdout_thread = Thread(
            target=self._watch_thread, args=(self._subp.stdout, MsgType.OUT)
        )
        self._stderr_thread = Thread(
            target=self._watch_thread, args=(self._subp.stderr, MsgType.ERR)
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self, how=signal.SIGTERM):
        """
        Cleanly stop the process and mop up threads
        """
        if self._subp is None:
            self.logger.error("stop() called on command that is not running")
            return
        if self._subp.returncode is not None:
            self.logger.debug(
                "Command %s already exited with %s",
                self._cmd_name,
                self._subp.returncode,
            )
            return
        self.logger.debug("Stopping %s with %s", self._cmd_name, how)
        self._subp.send_signal(how)
        self.wait()
        self.logger.debug("Command %s cleanup done", self._cmd_name)


class SSHCommandRunner(CommandRunner):
    """
    Execute a remote command via SSH and watch the output.
    """

    def __init__(self, logger, conn, msg_queue):
        super().__init__(logger, msg_queue)
        self.logger = new_logger("sshcmd", parent=logger)
        #: SSH connection to the target instance
        self._conn = conn
        #: remote stderr
        self._stderr = None
        #: remote stdout
        self._stdout = None
        #: remote stdin
        self._stdin = None

    def _check_exited(self):
        assert self._stdout, "ssh exec_command() did not run?"
        if self._stdout.channel.exit_status_ready():
            return self._stdout.channel.recv_exit_status()
        return None

    def _ensure_exited(self):
        return self._stdout.channel.recv_exit_status()

    def run(self, cmd, args=None, env=None, cmd_id=None):
        super().run(cmd, args, env, cmd_id)
        cmdline = shlex.join([str(cmd)] + [str(a) for a in args])
        stdin, stdout, stderr = self._conn.exec_command(cmdline, environment=env)
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self._stdout_thread = Thread(
            target=self._watch_thread, args=(stdout, MsgType.OUT)
        )
        self._stderr_thread = Thread(
            target=self._watch_thread, args=(stderr, MsgType.ERR)
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self):
        """
        Cleanly stop the process and mop up threads
        """
        if not self._thread.is_alive():
            self.logger.error("stop() called on command that is not running")
            return
        self.logger.debug("Stopping %s with %s", self._cmd_name)
        self._stdin.send(b"\x03")  # ctrl-c in lack of a better way
        self.wait()
        self.logger.debug("Command %s cleanup done", self._cmd_name)


class Instance(ABC):
    """
    Base instance implementation.
    Manages a virtual machine/FPGA or other guest OS connectable via SSH.
    """

    @dataclass
    class Observer:
        """
        Message loop observers
        """

        msgtype: MsgType
        origin: str | None
        predicate: typing.Callable[bool, any]
        action: typing.Callable
        oneshot: bool = True

    def __init__(self, manager: "InstanceManager", config: InstanceConfig):
        #: Parent manager
        self.manager = manager
        #: Unique instance identifier, this is not the g_uuid which identifies the instance configuration
        self.uuid = uuid4()
        #: Instance configuration
        self.config = config
        # Per-instance logger
        self.logger = new_logger(f"instance[{self.config.name}]")
        #: SSH port allocated for the benchmarks to connect to the instance
        self.ssh_port = manager.next_ssh_port()
        #: SSH connection to the instance
        self._ssh_conn = None
        #: SFTP connection to the instace
        self._sftp_conn = None
        #: Instance state lock
        self._state_lock = Lock()
        #: Instance state changed event
        self._state_changed = Condition(self._state_lock)
        #: Instance state
        self._state = InstanceState.DORMANT
        #: The instance event loop thread that manages the instance state transitions and collects I/O
        self._thread = Thread(target=self._instance_loop)
        #: Comman execution I/O watcher threads
        self._active_commands = []
        #: Message queue from exec threads to the watchdog thread
        self._msg_queue = Queue()
        #: msg observers and associated lock
        self._observers_lock = Lock()
        self._msg_observers = []

        self._thread.start()

    def __str__(self):
        return (
            f"Instance {self.config.name}({self.uuid}) {self.config.platform}-{self.config.cheri_target}-"
            + f"{self.config.kernel}"
        )

    def _instance_loop(self):
        """
        Per-instance watchdog thread.
        """
        try:
            with self._state_lock:
                assert self._state == InstanceState.DORMANT
            while True:
                msg = self._msg_queue.get()
                if msg.msgtype == MsgType.NEW_STATE:
                    self._switch_state(msg.data)
                    self._handle_state(msg.data)
                elif msg.msgtype == MsgType.OUT:
                    self.logger.debug("%s: %s", msg.origin, msg.data)
                elif msg.msgtype == MsgType.ERR:
                    self.logger.warning("%s: %s", msg.origin, msg.data)
                elif msg.msgtype == MsgType.EXITED or msg.msgtype == MsgType.FAILED:
                    self.logger.debug("Instance command %s completed", msg.origin)
                    with self._state_lock:
                        if msg.data in self._active_commands:
                            self._active_commands.remove(msg.data)
                        else:
                            self.logger.warning(
                                "Multiple %s messages for %s", msg.msgtype, msg.origin
                            )
                else:
                    self.logger.error("Unknown message type %s", msg.msgtype)
                self._msg_queue.task_done()
                # Run through the msg observers
                self._handle_observers(msg)
                with self._state_lock:
                    # If we reached the dead state always exit the thread
                    if self._state == InstanceState.DEAD:
                        break
        except Exception as ex:
            self.logger.exception("Fatal error: %s - shutdown", ex)
            # XXX what to do if the instance is DEAD here?
            self._switch_state(InstanceState.SHUTDOWN)
            self._shutdown()
        finally:
            self._switch_state(InstanceState.DEAD)
            self.logger.debug("Exiting instance loop")

    @abstractmethod
    def _boot(self):
        """
        Boot the instance. This must be overridden by concrete classes.
        """
        ...

    @abstractmethod
    def _shutdown(self):
        """
        Shutdown the instance. This can be called when an error occurs or when
        the instance is not needed anymore.
        """
        ...

    def _handle_observers(self, msg: Msg):
        """
        Run through the observers list for the given message
        """
        with self._observers_lock:
            remove = []
            for obs in self._msg_observers:
                if obs.msgtype != msg.msgtype or obs.origin != msg.origin:
                    continue
                if obs.predicate and not obs.predicate(msg.data):
                    continue
                obs.action()
                if obs.oneshot:
                    remove.append(obs)
            for obs in remove:
                self._msg_observers.remove(obs)

    def _attach_msg_observer(
        self,
        msgtype: MsgType,
        origin: str,
        predicate: typing.Callable[bool, any],
        callback: typing.Callable,
        oneshot=True,
    ):
        """
        Attach an observer to the message loop.
        This can be used to trigger state transitions based on messages.
        """
        observer = self.Observer(msgtype, origin, predicate, callback, oneshot)
        with self._observers_lock:
            self._msg_observers.append(observer)

    def _handle_state(self, state: InstanceState):
        """
        Handle the give instance loop state
        """
        self.logger.debug("Reached %s", state)
        if state == InstanceState.BOOTING:
            self._boot()
        elif state == InstanceState.SETUP:
            self._ssh_connect()
            self._signal_state(InstanceState.READY)
        elif state == InstanceState.SHUTDOWN:
            self._shutdown()
            self._signal_state(InstanceState.DEAD)

    def _signal_state(self, next_state: InstanceState):
        """
        Signal that we would like to transition the state of the instance.
        This enqueues a state transition message to the message loop
        """
        self._msg_queue.put(Msg(MsgType.NEW_STATE, None, next_state))

    def _switch_state(self, next_state: InstanceState):
        """
        Switch the instance state and notify waiters.
        """
        with self._state_changed:
            if next_state < self._state:
                self.logger.error(
                    "Forbidden state transition %s -> %s", self._state, next_state
                )
                raise RuntimeError("Forbidden instance state transition")
            if next_state == self._state:
                return
            self.logger.debug("State %s -> %s", self._state, next_state)
            self._state = next_state
            self._state_changed.notify_all()

    def _wait_state(self, target_state):
        """
        Wait for the instance to reach a specific state.
        If the instance can not reach the target state we raise an exception.
        """
        self.logger.debug("Wait %s", target_state)
        with self._state_changed:
            if self._state > target_state:
                self.logger.warning(
                    "Waiting for %s, but the instance is %s", target_state, self._state
                )
                raise RuntimeError("Can not reach waited state")

            self._state_changed.wait_for(
                lambda: self._state == target_state or self._state == InstanceState.DEAD
            )
            if self._state != target_state and self._state == InstanceState.DEAD:
                self.logger.warning("Instance died while waiting for %s", target_state)
                raise RuntimeError("instance died")

    def _get_ssh_info(self) -> tuple[str, dict]:
        """
        Produce the ssh host:port pair for this instance and other parameters such as the username

        :return: A tuple, the first element is the hostname, the second a dict containing key-value parameters
        """
        return ("localhost", {"port": self.ssh_port, "username": "root"})

    def _ssh_connect(self, retry=2):
        """
        Setup the initial ssh connection to the instance. This must be done when the instance has booted but
        before signaling that we are ready to run commands.
        """
        self._ssh_conn = SSHClient()
        self._ssh_conn.load_system_host_keys()
        self._ssh_conn.set_missing_host_key_policy(WarningPolicy)
        host, kwargs = self._get_ssh_info()
        # Give qemu a bit of time to settle so that we don't get useless retry errors
        time.sleep(5)
        try:
            while retry > 0:
                try:
                    retry -= 1
                    keyfile_path = self.session.config.ssh_key.expanduser()
                    self._ssh_conn.connect(
                        host, key_filename=str(keyfile_path), passphrase="", **kwargs
                    )
                    break
                except SSHException as ex:
                    if retry <= 0:
                        raise ex
                    time.sleep(5)
            self._sftp_conn = self._ssh_conn.open_sftp()
        except Exception:
            self.logger.exception("Failed to connect to the instance, trigger cleanup")
            self._ssh_conn = None
            raise

    def _exec_host(
        self, path: str | Path, args: list, cmd_id=None
    ) -> HostCommandRunner:
        """
        Run a command on the host machine and initiate an I/O thread that inspects the
        subprocess I/O.
        """
        cmd = HostCommandRunner(self.logger, self._msg_queue)
        with self._state_lock:
            if self._state > InstanceState.READY:
                raise RuntimeError("Can not execute command after shutdown")
            self._active_commands.append(cmd)
        cmd.run(path, args, cmd_id=cmd_id)
        return cmd

    def _exec_guest(
        self, path: str | Path, args: list[str] = None, env: dict = None, sync=True
    ):
        """
        Run a command on the guest machine and initiate an I/O thread that inspects
        the subprocess I/O if the sync flag is not set.

        :param cmd: The program path or name, if the program is in the remote $PATH
        :param args: List of arguments
        :param env: Environment variables to set
        """
        cmd = SSHCommandRunner(self.logger, self._ssh_conn, self._msg_queue)
        with self._state_lock:
            if self._state > InstanceState.READY:
                raise RuntimeError("Can not execute command after shutdown")
            self._active_commands.append(cmd)
        cmd.run(path, args, env)
        if sync:
            cmd.wait()
        return cmd

    @property
    def state(self) -> InstanceState:
        with self._state_lock:
            return self._state

    @property
    def user_config(self):
        return self.manager.session.user_config

    @property
    def session(self):
        return self.manager.session

    def signal_boot(self):
        """
        Signal a request to transition from DORMANT to BOOTING.
        If the state is not DORMANT, raise an exception.
        """
        with self._state_lock:
            if self._state != InstanceState.DORMANT:
                self.logger.error(
                    "Attempted to signal boot but state is %s", self._state
                )
                raise RuntimeError("Can not boot")
        self._signal_state(InstanceState.BOOTING)

    def signal_shutdown(self):
        """
        Signal a request to transition from DORMANT, BOOTING or READY to SHUTDOWN.
        If the state is not one of the three states, raise and exception.
        """
        with self._state_lock:
            if self._state > InstanceState.READY:
                self.logger.error(
                    "Attempted to signal shutdown but state is %s", self._state
                )
                raise RuntimeError("Can not shutdown")
        self._signal_state(InstanceState.SHUTDOWN)

    def wait_ready(self):
        """
        Wait for the instance to reach the READY state.
        If the instance dies this throws an exception instead.
        """
        self._wait_state(InstanceState.READY)

    def wait_dead(self):
        """
        Wait for the instance to reach the DEAD state.
        This should not throw an exception, eventually everything dies.
        """
        self._wait_state(InstanceState.DEAD)

    def run_cmd(self, cmd: str | Path, args: list[str] = None, env: dict = None):
        """
        Run a command on the remote instance synchronously.

        :param cmd: The program path or name, if the program is in the remote $PATH
        :param args: List of arguments
        :param env: Environment variables to set
        """
        self._exec_guest(cmd, args, env, sync=True)

    def extract_file(self, guest_src: Path, host_dst: Path):
        """
        Extract file from instance

        :param guest_src: Source path on the remote guest
        :param host_dst: Destination path on the current machine
        """
        # Paramiko is annoying and does not like Path objects
        self._sftp_conn.get(str(guest_src), str(host_dst))

    def import_file(self, host_src: Path, guest_dst: Path):
        """
        Import file into instance

        :param host_src: Source path on the current machine
        :param guest_dst: Destination path on the remote guest
        """
        # Paramiko is annoying and does not like Path objects
        self._sftp_conn.put(str(host_src), str(guest_dst))


class QEMUInstance(Instance):
    """
    A virtual machine instance managed by cheribuild.
    """

    def __init__(self, manager, config):
        super().__init__(manager, config)
        self.cheribuild = self.user_config.cheribuild_path / "cheribuild.py"
        #: The cheribuild command wrapper
        self._cheribuild_runner = None

    def _cheribuild_target(self, prefix):
        return f"{prefix}-{self.config.cheri_target}"

    def _run_option(self, opt):
        prefix = self._cheribuild_target("run")
        return f"--{prefix}/{opt}"

    def _cheribsd_option(self, opt):
        prefix = self._cheribuild_target("cheribsd")
        return f"--{prefix}/{opt}"

    def _get_qemu_trace_sink(self):
        # This will be a binary file containing serialized perfetto protobufs.
        return self.config.platform_options.qemu_trace_file

    def _get_qemu_interceptor_sink(self):
        return self.config.platform_options.qemu_interceptor_trace_file

    def _boot(self):
        run_cmd = [self._cheribuild_target("run"), "--skip-update"]
        run_cmd += [self._run_option("alternative-kernel"), self.config.kernel]
        run_cmd += [self._run_option("ephemeral")]
        run_cmd += [self._run_option("ssh-forwarding-port"), str(self.ssh_port)]
        # Let cheribuild know that it may look up all kernels
        run_cmd += [
            self._cheribsd_option("build-alternate-abi-kernels"),
            self._cheribsd_option("build-fett-kernels"),
            self._cheribsd_option("build-bench-kernels"),
            self._cheribsd_option("build-fpga-kernels"),
        ]
        if not self.config.cheribuild_kernel:
            # If the kernel specified is not a cheribuild-managed kernel we have to tell
            # cheribuild that it exists
            run_cmd += [
                self._cheribsd_option("extra-kernel-configs"),
                self.config.kernel,
            ]
        # Extra qemu options and tracing tags?
        if self.config.platform_options.qemu_trace == "perfetto":
            qemu_options = [
                "--cheri-trace-backend",
                "perfetto",
                "--cheri-trace-perfetto-logfile",
                str(self._get_qemu_trace_sink()),
                "--cheri-trace-perfetto-categories",
                ",".join(self.config.platform_options.qemu_trace_categories),
                "-icount",
                "shift=5,align=off",
            ]
            run_cmd += [self._run_option("extra-options"), " ".join(qemu_options)]
        elif self.config.platform_options.qemu_trace == "perfetto-dynamorio":
            qemu_options = [
                "--cheri-trace-backend",
                "perfetto",
                "--cheri-trace-perfetto-logfile",
                str(self._get_qemu_trace_sink().with_suffix(".pb")),
                "--cheri-trace-perfetto-enable-interceptor",
                "--cheri-trace-perfetto-interceptor-logfile",
                str(self._get_qemu_interceptor_sink()),
                "--cheri-trace-perfetto-categories",
                ",".join(self.config.platform_options.qemu_trace_categories),
                "-icount",
                "shift=5,align=off",
            ]
            run_cmd += [self._run_option("extra-options"), " ".join(qemu_options)]
        else:
            assert self.config.platform_options.qemu_trace == "no"

        # Attach the observers before running the command to avoid potential races
        sshd_pattern = re.compile(r"Starting sshd\.")
        self._attach_msg_observer(
            MsgType.OUT,
            "cheribuild",
            lambda v: sshd_pattern.match(v),
            lambda: self._signal_state(InstanceState.SETUP),
        )
        self._attach_msg_observer(
            MsgType.EXITED,
            "cheribuild",
            None,
            lambda: self._signal_state(InstanceState.SHUTDOWN),
        )
        self._attach_msg_observer(
            MsgType.FAILED,
            "cheribuild",
            None,
            lambda: self._signal_state(InstanceState.SHUTDOWN),
        )
        self._cheribuild_runner = self._exec_host(
            self.cheribuild, run_cmd, "cheribuild"
        )

    def _try_clean_shutdown(self):
        """
        Attempt to cleanly shutdown the instance.
        If this fails we fall back to killing things.
        """
        # Do not use an SSHCommandRunner here as we are cleaning up
        stdin, stdout, stderr = self._ssh_conn.exec_command("poweroff")
        # Wait for command to finish
        stdout.channel.recv_exit_status()
        # Wait for the cheribuild process to exit
        self._cheribuild_runner.wait(timeout=30)

    def _hard_shutdown(self):
        """
        Just kill the cheribuild process.
        If somebody already pulled it from the active_commands list it means that we
        are quitting as a result of cheribuild dying on us. This occurs because shutdown
        is triggered via the instance_loop or msg_observers.
        """
        with self._state_lock:
            need_stop = self._cheribuild_runner in self._active_commands
        if need_stop:
            self._active_commands.remove(self._cheribuild_runner)

    def _shutdown(self):
        """
        Ensure shutdown of the qemu instance.

        First try to send a poweroff command and wait for the cheribuild runner to exit.
        If we can not connect and the cheribuild runner is doing something, stop cheribuild with SIGINT.
        """
        try:
            if self._cheribuild_runner is None:
                # Nothing is running, done
                return
            # Stop pending ssh commands
            # During shutdown the cheribuild process will exit spontaneously so we don't have to
            # drop it from the queue here, we will see it send an EXITED message to the instance loop
            # when it is done.
            with self._state_lock:
                for cmd in self._active_commands:
                    if cmd != self._cheribuild_runner:
                        cmd.stop()
            # Not sure if we want to do this whole thing while holding the lock, to prevent
            # anybody else from sending work? Given the current structure this is technically unnecessary.
            if (
                self._ssh_conn is not None
                and self._ssh_conn.get_transport() is not None
                and self._ssh_conn.get_transport().is_active()
            ):
                self._try_clean_shutdown()
            else:
                self._hard_shutdown()
        except TimeoutError:
            self._hard_shutdown()
        except Exception as ex:
            # There should not be exceptions during shutdown because there is nothing left to
            # mop up after us.
            self.logger.critical("Critical error during instance shutdown %s", ex)
        finally:
            # Restore this to the initial state just in case
            self._cheribuild_runner = None
            self._ssh_conn = None


class VCU118Instance(Instance):
    """
    FPGA instance managed by cheribuild scripts.
    This is highly tied to the internal Cambridge computer laboratory setup.
    """

    def __init__(self, manager, config):
        super().__init__(manager, config)
        # Always use port 22
        self.ssh_port = 22
        #: The VCU118 cheribuild script
        self.runner_script = self.user_config.cheribuild_path / "vcu118-run.py"
        #: The command runner handling the vcu118-run.py script
        self._vcu118_runner = None

    def _get_fpga_bios(self):
        if self.config.platform_options.vcu118_bios:
            return self.config.platform_options.vcu118_bios
        else:
            return (
                self.user_config.sdk_path
                / "sdk"
                / "bbl-gfe"
                / str(self.config.cheri_target)
                / "bbl"
            )

    def _get_fpga_kernel(self):
        sdk = self.user_config.sdk_path
        kernel = sdk / f"kernel-{self.config.cheri_target}.{self.config.kernel}"
        assert kernel.exists(), f"Missing kernel {kernel}"
        return kernel

    def _get_fpga_gdb(self):
        gdb = self.user_config.sdk_path / "sdk" / "bin" / "gdb"
        gdb = Path("/homes/jrtc4/out/criscv-gdb")
        assert gdb.exists()
        return gdb

    def _get_fpga_cores(self):
        return str(self.config.platform_options.cores)

    def _get_ssh_info(self):
        ssh_host = self.config.platform_options.vcu118_ip
        return (ssh_host, {"port": self.ssh_port, "username": "root"})

    def _get_fpga_pubkey(self):
        ssh_privkey = self.session_config.ssh_key
        ssh_pubkey = ssh_privkey.with_suffix(".pub")
        with open(ssh_pubkey, "r") as fd:
            pubkey_str = fd.read()
        return pubkey_str.rstrip()

    def _boot(self):
        # Assume that the FPGA image has been initialized already.
        # XXX we should learn to detect first boot or do this once at startup.
        # vcu118-run --bitfile design.bit --ltxfile design.ltx --bios bbl-gfe-riscv64-purecap --kernel kernel --gdb gdb --openocd /usr/bin/openocd --num-cores N --benchmark-config
        run_cmd = [
            "--bios",
            self._get_fpga_bios(),
            "--kernel",
            self._get_fpga_kernel(),
            "--gdb",
            self._get_fpga_gdb(),
            "--openocd",
            self.user_config.openocd_path,
            "--num-cores",
            self._get_fpga_cores(),
            "--benchmark-config",
        ]
        ip_addr = self.config.platform_options.vcu118_ip
        run_cmd += [f"--test-command=ifconfig xae0 {ip_addr} netmask 255.255.255.0 up"]
        run_cmd += ["--test-command=mkdir -p /root/.ssh"]
        pubkey = self._get_fpga_pubkey()
        run_cmd += [
            f"--test-command=printf '%s' '{pubkey}' > /root/.ssh/authorized_keys"
        ]
        run_cmd += ["--test-command=echo Instance startup done"]
        run_cmd = [str(arg) for arg in run_cmd]

        # Attach observers to switch instance state
        wait_pattern = re.compile(r"Instance startup done")
        self._attach_msg_observer(
            MsgType.OUT,
            self.runner_script,
            lambda v: wait_pattern.match(v),
            lambda: self._signal_state(InstanceState.SETUP),
        )
        self._attach_msg_observer(
            MsgType.EXITED,
            self.runner_script,
            None,
            lambda: self._signal_state(InstanceState.DEAD),
        )
        self._attach_msg_observer(
            MsgType.FAILED,
            self.runner_script,
            None,
            lambda: self._signal_state(InstanceState.DEAD),
        )
        self._vcu118_runner = self._exec_host(self.runner_script, run_cmd)

    def _shutdown(self):
        """
        Ensure shutdown of the FPGA instance.

        In theory we can try to poweroff and do a clean shutdown, but we can also just terminate the script.
        """
        try:
            # XXX try to shutdown cleanly
            if self._vcu11_runner is not None:
                self._vcu118_runner.stop(how=signal.SIGTERM)
            # XXX this should killpg instead
            # self.logger.debug("Sending SIGTERM to vcu118 runner")
            # os.killpg(os.getpgid(self._run_task, signal.SIGTERM))
        except Exception as ex:
            # There should not be exceptions during shutdown because there is nothing left to
            # mop up after us.
            self.logger.critical("Critical error during instance shutdown %s", ex)


class InstanceManager(ResourceManager):
    """
    Manager for the instances to run benchmarks on.
    These can be VM or physical machines such as FPGAs.
    The resource limit here is the maximum instance concurrency supported by the system,
    e.g. we may only have one FPGA. The parent _limit_guard semaphore guards the total number
    of instances currently in use by benchmark tasks.
    We have an additional guard that checks for the total number of running instances, this
    is the sum of idle and active instances and may never exceed the concurrency limit.

    The ResourceRequest pool is interpreted as the instance configuration UUID (e.g. the g_uuid).
    """

    resource_name = "instance-resource"

    def __init__(self, session, limit=None):
        if limit is None:
            limit = session.config.concurrent_instances
        super().__init__(session, limit)
        #: last SSH port allocated to the instances, this is also protected by the manager lock
        self._last_ssh_port = 12000
        #: Idle instances, protected by _resource_lock, maps the g_uuid UUID -> List[Instance]
        self._idle_instances = defaultdict(list)
        #: Instances shutting down, protected by _resource_lock
        self._shutdown_instances = []
        #: Active instances currently in use, protected by _resource_lock, same type as _idle_instances
        self._active_instances = defaultdict(list)
        #: Instances waiting for boot
        self._booting_instances = {}
        #: Guard against instance concurrency maximum
        if limit:
            self._total_instances_guard = Semaphore(limit)
        #: Instance lists lock
        self._manager_lock = Lock()

        self.logger.info("Initialized %s manager", self.resource_name)

    def _validate_request(self, req):
        if req.pool not in self.session.parameterization_matrix["instance"]:
            self.logger.error(
                "Invalid resource request: pool %s does not correspond to any g_uuid",
                req.pool,
            )
            raise ValueError("Invalid resource request")
        if "instance_config" not in req.acquire_args:
            self.logger.error(
                "Invalid resource request: missing instance_config kwargs"
            )
            raise ValueError("Invalid resource request")

    def _find_instance_type(
        self, instance_config: InstanceConfig
    ) -> typing.Type[Instance]:
        """
        Resolve the instance type for a given instance configuration.
        This is mostly decoupled to help mocking in tests.
        """
        return QEMUInstance
        # if instance_config.platform == InstancePlatform.QEMU:
        #     return QEMUInstance
        # elif instance_config.platform == InstancePlatform.VCU118:
        #     return VCU118Instance
        # else:
        #     self.logger.error(
        #         "Unknown instance %s platform %s",
        #         instance_config.name,
        #         instance_config.platform,
        #     )
        #     raise TypeError("Unknown instance platform")
        # elif instance_config.platform == InstancePlatform.MORELLO:
        #     return MorelloInstance

    def _create_instance(
        self, g_uuid: UUID, instance_config: InstanceConfig
    ) -> Instance:
        """
        Allocate a new instance and boot it. Assume that the instance limit has been acquired
        and we are able to allocate the instance. Once booted we return the instance.
        """
        instance_klass = self._find_instance_type(instance_config)
        instance = instance_klass(self, instance_config)
        self.logger.debug("Created instance %s", instance.uuid)
        with self._manager_lock:
            assert instance.uuid not in self._booting_instances
            self._booting_instances[instance.uuid] = instance
        # Creates watchdog thread and wait for boot
        instance.signal_boot()
        instance.wait_ready()
        self.logger.debug("Booted instance %s", instance.uuid)
        with self._manager_lock:
            del self._booting_instances[instance.uuid]
            self._active_instances[g_uuid].append(instance)
        return instance

    def _try_create_instance(
        self, g_uuid: UUID, instance_config: InstanceConfig
    ) -> Instance | None:
        """
        Try to reserve the space for a new instance and boot it. If we fail, we immediately return.
        """
        if self.limit:
            acquired = self._total_instances_guard.acquire(blocking=False)
            if not acquired:
                return None
        # We now can create a new instance safely
        try:
            return self._create_instance(g_uuid, instance_config)
        except:
            if self.limit:
                self._total_instances_guard.release()
            raise

    def _get_resource(self, req) -> Instance:
        """
        Produce a running instance and its associated connection.
        The resulting instance will be booted and ready to run some work.
        """
        # Validate the resource request
        self._validate_request(req)
        instance_config = req.acquire_args["instance_config"]
        instance = None
        # First try to reuse an idle instance, this would not change the total instances count
        with self._manager_lock:
            idle_pool = self._idle_instances[req.pool]
            if idle_pool:
                instance = idle_pool.pop()
                self._active_instances[req.pool].append(instance)
        if instance:
            self.logger.debug(
                "Reuse idle instance %s(%s) id=%s",
                instance_config.name,
                req.pool,
                instance.uuid,
            )
            return instance

        # If no matching idle instance is found we need to create a new instance.
        instance = self._try_create_instance(req.pool, instance_config)
        if instance:
            self.logger.debug(
                "New instance %s(%s) id=%s",
                instance_config.name,
                req.pool,
                instance.uuid,
            )
            return instance
        assert self.limit, "Should not be reached without a limit set"
        # If we could not create one quickly we need to wait for one to finish,
        # check if there is an idle instance we can kill.
        with self._manager_lock:
            target_pool = None
            for idle_pool in self._idle_instances.values():
                if target_pool is None or len(idle_pool) > len(target_pool):
                    target_pool = idle_pool
            if target_pool:
                # We can kill an instance
                target = target_pool.pop()
                self._shutdown_instances.append(target)
                target.signal_shutdown()
        # Now we wait for an instance slot to release
        self._total_instances_guard.acquire()
        instance = self._create_instance(instance.uuid, instance_config)
        self.logger.debug(
            "New instance %s(%s) id=%s", instance_config.name, req.pool, instance.uuid
        )
        return instance

    def _put_resource(self, instance: Instance, req):
        """
        Return an instance to the idle list or shut it down if we do not allow reuse.
        """
        self.logger.debug("Return instance %s to pool %s", instance.uuid, req.pool)
        with self._manager_lock:
            active_pool = self._active_instances[req.pool]
            active_pool.remove(instance)
            if self.session.config.reuse_instances:
                self.logger.debug(
                    "Append instance %s to idle pool %s", instance.uuid, req.pool
                )
                self._idle_instances[req.pool].append(instance)
            else:
                self.logger.debug(
                    "Instance reuse disabled, trigger instance %s shutdown",
                    instance.uuid,
                )
                self._shutdown_instances.append(instance)
                instance.signal_shutdown()

    def next_ssh_port(self):
        with self._manager_lock:
            self._last_ssh_port += 1
            return self._last_ssh_port

    def sched_shutdown(self):
        """
        Shutdown everything cleanly.
        Note that this should be called after every task has been cancelled.
        """
        super().sched_shutdown()
        self.logger.info("Shutdown %s manager", self.resource_name)
        with self._manager_lock:
            for pool in self._active_instances.values():
                for instance in pool:
                    instance.signal_shutdown()
                    self._shutdown_instances.append(instance)
            self._active_instances.clear()
            for pool in self._idle_instances.values():
                for instance in pool:
                    instance.signal_shutdown()
                    self._shutdown_instances.append(instance)
            self._idle_instances.clear()
            for instance in self._shutdown_instances:
                instance.wait_dead()
            self._shutdown_instances.clear()
