import asyncio as aio
import copy
import os
import re
import signal
import traceback
import typing
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import asyncssh

from .config import Config, TemplateConfig, path_field
from .util import new_logger


class InstanceManagerError(Exception):
    pass


class InstancePlatform(Enum):
    QEMU = "qemu"
    FLUTE = "flute"
    TOOOBA = "toooba"
    MORELLO_FPGA = "morello"
    FPGA = "fpga"

    def __str__(self):
        return self.value


class InstanceCheriBSD(Enum):
    RISCV64_PURECAP = "riscv64-purecap"
    RISCV64_HYBRID = "riscv64-hybrid"
    MORELLO_PURECAP = "morello-purecap"
    MORELLO_HYBRID = "morello-hybrid"

    def is_riscv(self):
        return (self == InstanceCheriBSD.RISCV64_PURECAP or self == InstanceCheriBSD.RISCV64_HYBRID)

    def is_morello(self):
        return (self == InstanceCheriBSD.MORELLO_PURECAP or self == InstanceCheriBSD.MORELLO_HYBRID)

    def __str__(self):
        return self.value


class InstanceKernelABI(Enum):
    NOCHERI = "nocheri"
    HYBRID = "hybrid"
    PURECAP = "purecap"

    def __str__(self):
        return self.value


@dataclass
class PlatformOptions(Config):
    """
    Base class for platform-specific options.
    This is internally used during benchmark dataset collection to
    set options for the instance that is to be run.
    """
    # The trace file used by default unless one of the datasets overrides it
    qemu_trace_file: Path = path_field("/tmp/qemu.pb")
    # Run qemu with tracing enabled
    qemu_trace: bool = False
    # Trace categories to enable for qemu-perfetto
    qemu_trace_categories: set[str] = field(default_factory=set)


@dataclass
class InstanceConfig(TemplateConfig):
    """
    Configuration for a CheriBSD instance to run benchmarks on.
    XXX-AM May need a custom __eq__() if iterable members are added
    """
    kernel: str
    baseline: bool = False
    name: typing.Optional[str] = None
    platform: InstancePlatform = InstancePlatform.QEMU
    cheri_target: InstanceCheriBSD = InstanceCheriBSD.RISCV64_PURECAP
    kernelabi: InstanceKernelABI = InstanceKernelABI.HYBRID
    # Is the kernel config name managed by cheribuild or is it an extra one
    # specified via --cheribsd/extra-kernel-configs?
    cheribuild_kernel: bool = True
    # Internal fields, should not appear in the config file and are missing by default
    platform_options: typing.Optional[PlatformOptions] = field(default=None, init=False)

    @property
    def user_pointer_size(self):
        if (self.cheri_target == InstanceCheriBSD.RISCV64_PURECAP
                or self.cheri_target == InstanceCheriBSD.MORELLO_PURECAP):
            return 16
        elif (self.cheri_target == InstanceCheriBSD.RISCV64_HYBRID
              or self.cheri_target == InstanceCheriBSD.MORELLO_HYBRID):
            return 8
        assert False, "Not reached"

    @property
    def kernel_pointer_size(self):
        if (self.cheri_target == InstanceCheriBSD.RISCV64_PURECAP
                or self.cheri_target == InstanceCheriBSD.MORELLO_PURECAP):
            if self.kernelabi == InstanceKernelABI.PURECAP:
                return self.user_pointer_size
            else:
                return 8
        elif (self.cheri_target == InstanceCheriBSD.RISCV64_HYBRID
              or self.cheri_target == InstanceCheriBSD.MORELLO_HYBRID):
            if self.kernelabi == InstanceKernelABI.PURECAP:
                return 16
            else:
                return self.user_pointer_size
        assert False, "Not reached"

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = f"{self.platform}-{self.cheri_target}-{self.kernelabi}"


class InstanceStatus(Enum):
    """
    Status of an instance managed by the daemon
    INIT: instance initialized but not yet started
    BOOT: instance booting
    READY: instance finished booting, the control connection is available
    SHUTDOWN: instance is stopping
    DEAD: instance is dead because of shutdown or because of an error
    """
    INIT = "initialized"
    BOOT = "booting"
    READY = "ready"
    SHUTDOWN = "shutdown"
    DEAD = "dead"


class InstanceInfo:
    def __init__(self, logger, uuid, ssh_host, ssh_port, ssh_key):
        self.logger = logger
        # ID of the instance
        self.uuid = uuid
        # SSH host to reach the instance
        self.ssh_host = ssh_host
        # SSH port to reach the instance
        self.ssh_port = ssh_port
        # SSH key for the instance
        self.ssh_key = ssh_key
        self.conn = None

    async def connect(self):
        """
        Create new connection to the instance.
        Only one connection can exist at a time in the current implementation.
        """
        if self.conn is not None:
            await self.disconnect()
            self.conn = None
        self.conn = await asyncssh.connect(self.ssh_host,
                                           port=self.ssh_port,
                                           known_hosts=None,
                                           client_keys=[self.ssh_key],
                                           username="root",
                                           passphrase="")
        self.logger.debug("Connected to instance")

    async def disconnect(self):
        if self.conn is not None:
            self.conn.close()
            await self.conn.wait_closed()

    async def extract_file(self, guest_src: Path, host_dst: Path, **kwargs):
        """Extract file from instance"""
        src = (self.conn, guest_src)
        await asyncssh.scp(src, host_dst, **kwargs)

    async def import_file(self, host_src: Path, guest_dst: Path, **kwargs):
        """Import file into instance"""
        dst = (self.conn, guest_dst)
        await asyncssh.scp(host_src, dst, **kwargs)

    async def run_cmd(self, command: str, args: list = [], env: dict = {}):
        """Run command on the remote instance"""
        self.logger.debug("SH exec: %s %s", command, args)
        cmdline = f"{command} " + " ".join(map(str, args))
        result = await self.conn.run(cmdline)
        if result.returncode != 0:
            self.logger.error("Failed to run %s: %s", command, result.stderr)
        else:
            self.logger.debug("%s done: %s", command, result.stdout)
        return result


class Instance(ABC):
    last_ssh_port = 12000

    def __init__(self, inst_manager: "InstanceManager", config: InstanceConfig):
        self.manager = inst_manager
        # Main event loop
        self.event_loop = inst_manager.loop
        # Top level configuration
        self.manager_config: "BenchmarkManagerConfig" = inst_manager.manager_config
        # Instance configuration
        self.config = config
        # Unique ID of the instance
        self.uuid = uuid.uuid4()
        self.logger = new_logger(f"{self.uuid}({self.config.name})")
        # SSH port allocated for the benchmarks to connect to the instance
        self.ssh_port = self._get_ssh_port()
        # The task associated to this instance runner task
        self.task = None
        # Control connection to the instance
        self._ssh_ctrl_conn = None
        # Release event that initiates instance shutdown
        self.release_event = aio.Event()
        # Signal anybody waiting on a status change
        self.status_update_event = aio.Event()
        self.status = InstanceStatus.INIT

    def set_status(self, next_status):
        self.logger.debug("STATUS %s -> %s", self.status, next_status)
        self.status = next_status
        self.status_update_event.set()

    def get_info(self) -> InstanceInfo:
        """
        Get instance information needed by benchmarks to connect and run on the instance
        """
        info = InstanceInfo(self.logger, self.uuid, "localhost", self.ssh_port, self.manager_config.ssh_key)
        return info

    def start(self):
        """Create the runner task that will boot the instance"""
        self.task = self.event_loop.create_task(self._run_instance())

    def kill(self):
        """Forcibly kill the instance"""
        self.task.cancel()

    def release(self):
        """
        Release instance from the current benchmark. This indicates that the benchmark
        is done running and we can trigger the instance reset before advancing to the
        next benchmark in the run_queue.
        """
        self.logger.info("Release instance")
        self.release_event.set()

    def stop(self):
        self.logger.info("Stopping instance")
        if self.task:
            self.task.cancel()

    def __str__(self):
        return (f"Instance {self.uuid} {self.config.platform} cheribsd:{self.config.cheribsd} " +
                f"kernel:{self.config.kernel}")

    async def wait(self, status: InstanceStatus):
        while self.status != status:
            await self.status_update_event.wait()
            self.status_update_event.clear()
            if (self.status == InstanceStatus.DEAD and status != InstanceStatus.DEAD):
                # If the instance died but we are not waiting the DEAD status
                # we die so that the waiting task is unblocked
                self.logger.error("Instance %s died while waiting for %s", self.uuid, status)
                raise InstanceManagerError("Instance died unexpectedly")

    def _get_ssh_port(self):
        port = Instance.last_ssh_port
        Instance.last_ssh_port += 1
        return port

    async def _run_cmd(self, prog, *args):
        cmdline = f"{prog}" + " ".join(args)
        self.logger.debug("exec %s", cmdline)
        result = await self._ssh_ctrl_conn.run(cmdline)
        if result.returncode != 0:
            self.logger.error("Command failed with %d: %s", result.returncode, result.stderr)
            raise InstanceManagerError("Control command failed")
        self.logger.debug("%s", result.stdout)

    async def _make_control_connection(self):
        """
        Make sure that the cheribsd host accepts all environment variables we are going to send.
        To do this, update the configuration and restart sshd
        """
        ssh_keyfile = self.manager_config.ssh_key.expanduser()
        self.logger.debug("Connect root@localhost:%d key=%s", self.ssh_port, ssh_keyfile)
        retry = 3
        while True:
            try:
                self._ssh_ctrl_conn = await asyncssh.connect("localhost",
                                                             port=self.ssh_port,
                                                             known_hosts=None,
                                                             client_keys=[ssh_keyfile],
                                                             username="root",
                                                             passphrase="")
                break
            except Exception as ex:
                if retry == 0:
                    raise ex
                retry -= 1
                await aio.sleep(5)
        self.logger.info("Control connection established")

    @abstractmethod
    async def _boot(self):
        """
        Boot the instance. This must be overridden by concrete classes.
        """
        ...

    @abstractmethod
    async def _shutdown(self):
        """
        Shutdown the instance. This can be called when an error occurs or when
        the instance is not needed anymore.
        """
        ...

    async def _run_instance(self):
        """
        Main task that boots the instance and wait for the benchmark to release it.
        We boot the instance and notify when ready to run benchmarks.
        """
        try:
            assert self.status == InstanceStatus.INIT
            # boot
            self.set_status(InstanceStatus.BOOT)
            await self._boot()
            await self._make_control_connection()
            self.set_status(InstanceStatus.READY)
            # Now we are running, wait for the benchmark to release the instance
            await self.release_event.wait()
            self.release_event.clear()
            self.set_status(InstanceStatus.SHUTDOWN)
            await self._shutdown()
        except aio.CancelledError as ex:
            self.logger.debug("Instance run cancelled")
            await self._shutdown()
            raise ex
        except Exception as ex:
            self.logger.exception("Fatal error: %s - shutdown instance", ex)
            await self._shutdown()
        finally:
            self.set_status(InstanceStatus.DEAD)
            self.logger.debug("Exiting benchmark instance main loop")


class CheribuildInstance(Instance):
    def __init__(self, inst_manager: "InstanceManager", config: InstanceConfig):
        super().__init__(inst_manager, config)
        self._cheribuild = self.manager_config.cheribuild_path.expanduser()
        # The cheribuild process task
        self._cheribuild_task = None
        # cheribuild output reader tasks
        self._io_tasks = []
        # Events used to wait on cheribuild output
        self._io_loop_observers = []

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

    def get_info(self) -> InstanceInfo:
        info = super().get_info()
        info.ssh_host = "localhost"
        return info

    async def _stdout_loop(self):
        while not self._cheribuild_task.stdout.at_eof():
            raw_out = await self._cheribuild_task.stdout.readline()
            if not raw_out:
                continue
            out = raw_out.decode("ascii")
            if self.manager_config.verbose:
                self.logger.debug(out.rstrip())
            for evt, matcher in self._io_loop_observers:
                if matcher(out, False):
                    evt.set()
        raise InstanceManagerError("cheribuild died")

    async def _stderr_loop(self):
        while not self._cheribuild_task.stderr.at_eof():
            raw_out = await self._cheribuild_task.stderr.readline()
            if not raw_out:
                continue
            out = raw_out.decode("ascii")
            if self.manager_config.verbose:
                self.logger.warning(out.rstrip())
            for evt, matcher in self._io_loop_observers:
                if matcher(out, True):
                    evt.set()
        raise InstanceManagerError("cheribuild died")

    async def _wait_io(self, matcher: typing.Callable[[str, bool], bool]):
        """Wait for matching I/O from cheribuild on stdout or stderr"""
        event = aio.Event()
        self._io_loop_observers.append((event, matcher))
        waiter = self.event_loop.create_task(event.wait())
        try:
            done, pending = await aio.wait([waiter, *self._io_tasks], return_when=aio.FIRST_COMPLETED)
            self._io_loop_observers.remove((event, matcher))
            errors = [t.exception() for t in done if t.exception() is not None]
            if errors:
                self.logger.error("Errors occurred while waiting for instance I/O: %s", errors)
                raise errors[0]
        finally:
            # Mop up the waiter task
            if not waiter.done() and not waiter.cancelled():
                waiter.cancel()
                await waiter

    async def _boot(self):
        run_cmd = [self._cheribuild_target("run"), "--skip-update"]
        run_cmd += [self._run_option("alternative-kernel"), self.config.kernel]
        run_cmd += [self._run_option("ephemeral")]
        run_cmd += [self._run_option("ssh-forwarding-port"), str(self.ssh_port)]
        # Let cheribuild know that it may look up all kernels
        run_cmd += [
            self._cheribsd_option("build-alternate-abi-kernels"),
            self._cheribsd_option("build-fett-kernels"),
            self._cheribsd_option("build-bench-kernels"),
            self._cheribsd_option("build-fpga-kernels")
        ]
        if not self.config.cheribuild_kernel:
            # If the kernel specified is not a cheribuild-managed kernel we have to tell
            # cheribuild that it exists
            run_cmd += [self._cheribsd_option("extra-kernel-configs"), self.config.kernel]
        # Extra qemu options and tracing tags?
        if (self.config.platform_options.qemu_trace):
            # Other backends are not as powerful, so focus on this one
            qemu_options = [
                "--cheri-trace-backend", "perfetto", "--cheri-trace-perfetto-logfile",
                str(self._get_qemu_trace_sink()), "--cheri-trace-perfetto-categories",
                ",".join(self.config.platform_options.qemu_trace_categories)
            ]
            run_cmd += [self._run_option("extra-options"), " ".join(qemu_options)]

        self.logger.debug("%s %s", self._cheribuild, run_cmd)
        self._cheribuild_task = await aio.create_subprocess_exec(self._cheribuild,
                                                                 *run_cmd,
                                                                 stdin=aio.subprocess.PIPE,
                                                                 stdout=aio.subprocess.PIPE,
                                                                 stderr=aio.subprocess.PIPE,
                                                                 start_new_session=True)
        self.logger.debug("Spawned cheribuild pid=%d pgid=%d", self._cheribuild_task.pid,
                          os.getpgid(self._cheribuild_task.pid))
        self._io_tasks.append(self.event_loop.create_task(self._stdout_loop()))
        self._io_tasks.append(self.event_loop.create_task(self._stderr_loop()))
        # Now wait for the boot to complete and start sshd
        sshd_pattern = re.compile(r"Starting sshd\.")
        ssh_keyfile = self.manager_config.ssh_key.expanduser()
        await self._wait_io(lambda out, is_stderr: sshd_pattern.match(out) if not is_stderr else False)
        # give qemu some time
        await aio.sleep(5)

    async def _shutdown(self):
        """
        If we have an active connection, send the poweroff command and wait for
        cheribuild to finish. Otherwise kill cheribuild with SIGINT.
        """
        if self._ssh_ctrl_conn:
            await self._run_cmd("poweroff")
            self._ssh_ctrl_conn.close()
            # Add timeout and force kill?
            await self._cheribuild_task.wait()
        elif self._cheribuild_task and self._cheribuild_task.returncode is None:
            # Kill with SIGINT so that cheribuild will cleanly kill childrens
            self.logger.debug("Sending SIGINT to cheribuild")
            os.killpg(os.getpgid(self._cheribuild_task.pid), signal.SIGINT)
        if len(self._io_tasks):
            await aio.gather(*self._io_tasks, return_exceptions=True)
        # Cleanup to avoid accidental reuse
        self._cheribuild_task = None
        self._ssh_ctrl_conn = None


class InstanceManager:
    """
    Helper object used to manage the lifetime of instances we run things on
    """
    def __init__(self, loop: aio.AbstractEventLoop, manager_config: "BenchmarkManagerConfig"):
        self.logger = new_logger("instance-manager")
        self.loop = loop
        self.manager_config = manager_config
        # Map running instances to the benchmark owning them
        self._active_instances = {}
        # Spare instances if recycling instances is enabled
        self._shutdown_instances = []

    def _create_instance(self, config: InstanceConfig):
        if config.platform == InstancePlatform.QEMU:
            instance = CheribuildInstance(self, config)
        elif config.platform == InstancePlatform.FPGA:
            self.logger.error("Not yet implemented")
            raise InstanceManagerError("Not yet implemented")
        else:
            self.logger.error(f"Unknown instance platform {config.platform}")
            raise InstanceManagerError("Unknown instance platform")
        # creates main loop task and boot the instance
        instance.start()
        self.logger.debug("Created instance %s", instance.uuid)
        return instance

    def _alloc_instance(self, owner: uuid.UUID, instance: Instance):
        self._active_instances[owner] = instance
        self.logger.debug("Allocate instance %s to benchmark %s", instance.uuid, owner)

    async def request_instance(self, owner: uuid.UUID, config: InstanceConfig) -> InstanceInfo:
        instance = self._create_instance(config)
        self._alloc_instance(owner, instance)
        await instance.wait(InstanceStatus.READY)
        return instance.get_info()

    async def release_instance(self, owner: uuid.UUID):
        instance = self._active_instances[owner]
        del self._active_instances[owner]
        self._shutdown_instances.append(instance)
        try:
            instance.release()
            await instance.wait(InstanceStatus.SHUTDOWN)
        except aio.CancelledError as ex:
            raise ex
        except Exception as ex:
            self.logger.exception("Failed to release instance %s", instance.uuid)
            instance.stop()
        finally:
            self.logger.debug("Released instance: %s", instance.uuid)

    async def shutdown(self):
        # Force-release everything and wait for shutdown
        for instance in self._active_instances.values():
            instance.release()
        await aio.gather(*[i.wait(InstanceStatus.DEAD) for i in self._active_instances], return_exceptions=True)
        await aio.gather(*[i.wait(InstanceStatus.DEAD) for i in self._shutdown_instances], return_exceptions=True)
        # XXX possibly report any errors here
        self.logger.debug("Instances shutdown completed")

    async def kill(self):
        # Force kill all instances
        for instance in self._active_instances.values():
            instance.kill()
        await aio.gather(*[i.task for i in self._active_instances.values()], return_exceptions=True)
        self.logger.debug("Instances killed successfully")
