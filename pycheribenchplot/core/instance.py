import asyncio as aio
import copy
import os
import re
import signal
import typing
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import asyncssh
import numpy as np
from multi_await import multi_await

from .config import TemplateConfig, path_field
from .util import new_logger


class InstanceManagerError(Exception):
    pass


class InstancePlatform(Enum):
    QEMU = "qemu"
    VCU118 = "vcu118"

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

    def freebsd_kconf_dir(self):
        if self.is_riscv():
            arch = "riscv"
        elif self.is_morello():
            arch = "arm64"
        else:
            assert False, "Unknown arch"
        return Path("sys") / arch / "conf"

    def __str__(self):
        return self.value


class InstanceKernelABI(Enum):
    NOCHERI = "nocheri"
    HYBRID = "hybrid"
    PURECAP = "purecap"

    def __str__(self):
        return self.value


@dataclass
class PlatformOptions(TemplateConfig):
    """
    Base class for platform-specific options.
    This is internally used during benchmark dataset collection to
    set options for the instance that is to be run.
    """
    # Number of cores in the system
    cores: int = 1
    # The trace file used by default unless one of the datasets overrides it
    qemu_trace_file: typing.Optional[Path] = None
    # Run qemu with tracing enabled
    qemu_trace: bool = False
    # Trace categories to enable for qemu-perfetto
    qemu_trace_categories: typing.Set[str] = field(default_factory=set)
    # VCU118 bios
    vcu118_bios: Path = None
    # IP to use for the VCU118 board
    vcu118_ip: str = "10.88.88.2"


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
    platform_options: PlatformOptions = field(default_factory=PlatformOptions)

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
            self.name = f"{self.platform}-{self.cheri_target}-{self.kernelabi}-{self.kernel}"


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
        self.logger.debug("Attempt to connect to %s:%d", self.ssh_host, self.ssh_port)
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
        self.logger = new_logger(f"instance[{self.config.name}]")
        # SSH port allocated for the benchmarks to connect to the instance
        self.ssh_port = self._get_ssh_port()
        # The task associated to this instance runner task
        self.task = None
        # Control connection to the instance
        self._ssh_ctrl_conn = None
        # Release event that initiates instance shutdown
        self.release_event = aio.Event()
        # Boot event
        self.boot_event = aio.Event()
        # Signal anybody waiting on a status change
        self.status_update_event = aio.Event()
        self.status = InstanceStatus.INIT

        # output reader tasks
        self._io_tasks = []
        # Events used to wait on instance output
        self._io_loop_observers = []

    def set_status(self, next_status):
        self.logger.debug("STATUS %s -> %s", self.status, next_status)
        self.status = next_status
        self.status_update_event.set()

    def get_info(self) -> InstanceInfo:
        """
        Get instance information needed by benchmarks to connect and run on the instance
        """
        info = InstanceInfo(self.logger, self.uuid, "localhost", self.ssh_port,
                            self.manager_config.ssh_key.expanduser())
        return info

    def start(self):
        """Create the runner task that will boot the instance"""
        self.logger.info("Created instance with UUID=%s", self.uuid)
        self.task = self.event_loop.create_task(self._run_instance())

    def boot(self):
        """Signal instance that it can start booting"""
        self.boot_event.set()

    def release(self):
        """
        Release instance from the current benchmark. This indicates that the benchmark
        is done running and we can trigger the instance reset before advancing to the
        next benchmark in the run_queue.
        """
        self.logger.info("Release instance")
        self.release_event.set()

    def stop(self):
        """Forcibly kill the instance"""
        self.logger.info("Stopping instance")
        if self.task:
            self.task.cancel()

    def __str__(self):
        return (f"Instance {self.config.name}({self.uuid}) {self.config.platform}-{self.config.cheri_target}-" +
                f"{self.config.kernel}")

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
        assert self._ssh_ctrl_conn is not None, "No control connection"
        await self._ssh_ctrl_conn.run_cmd(prog, args)

    async def _make_control_connection(self):
        """
        Make sure that the cheribsd host accepts all environment variables we are going to send.
        To do this, update the configuration and restart sshd
        """
        self._ssh_ctrl_conn = self.get_info()
        retry = 3
        while True:
            try:
                await self._ssh_ctrl_conn.connect()
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
            # wait for boot event
            await self.boot_event.wait()
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

    def _attach_io_observers(self, proc_task):
        stdout_task = self.event_loop.create_task(self._stdout_loop(proc_task))
        stderr_task = self.event_loop.create_task(self._stderr_loop(proc_task))
        self._io_tasks.append(stdout_task)
        self._io_tasks.append(stderr_task)

    async def _stdout_loop(self, proc_task):
        while not proc_task.stdout.at_eof():
            raw_out = await proc_task.stdout.readline()
            if not raw_out:
                continue
            out = raw_out.decode("ascii")
            if self.manager_config.verbose:
                self.logger.debug(out.rstrip())
            for evt, matcher in self._io_loop_observers:
                if matcher(out, False):
                    evt.set()
        raise InstanceManagerError(f"cheribuild died {proc_task.returncode}")

    async def _stderr_loop(self, proc_task):
        while not proc_task.stderr.at_eof():
            raw_out = await proc_task.stderr.readline()
            if not raw_out:
                continue
            out = raw_out.decode("ascii")
            if self.manager_config.verbose:
                self.logger.warning(out.rstrip())
            for evt, matcher in self._io_loop_observers:
                if matcher(out, True):
                    evt.set()
        raise InstanceManagerError(f"cheribuild died {proc_task.returncode}")

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


class CheribuildInstance(Instance):
    def __init__(self, inst_manager: "InstanceManager", config: InstanceConfig):
        super().__init__(inst_manager, config)
        self.cheribuild = self.manager_config.cheribuild_path / "cheribuild.py"
        # The cheribuild process task
        self._cheribuild_task = None

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

        self.logger.debug("%s %s", self.cheribuild, run_cmd)

        self._cheribuild_task = await aio.create_subprocess_exec(str(self.cheribuild),
                                                                 *run_cmd,
                                                                 stdin=aio.subprocess.PIPE,
                                                                 stdout=aio.subprocess.PIPE,
                                                                 stderr=aio.subprocess.PIPE,
                                                                 start_new_session=True)

        self.logger.debug("Spawned cheribuild pid=%d pgid=%d", self._cheribuild_task.pid,
                          os.getpgid(self._cheribuild_task.pid))
        self._attach_io_observers(self._cheribuild_task)
        # Now wait for the boot to complete and start sshd
        sshd_pattern = re.compile(r"Starting sshd\.")
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
            await self._ssh_ctrl_conn.disconnect()
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


class VCU118Instance(Instance):
    """
    Runner for VCU 118 boards, using the vcu118-run script in cheribuild.
    """
    def __init__(self, inst_manager: "InstanceManager", config: InstanceConfig):
        super().__init__(inst_manager, config)
        self.runner_script = self.manager_config.cheribuild_path / "vcu118-run.py"
        if inst_manager.manager_config.concurrent_instances != 1:
            self.logger.error("VCU118 MUST use concurrent_instances = 1")
            raise InstanceManagerError("Too many concurrent instances")

        self.ssh_port = 22

    def _get_fpga_bios(self):
        if self.config.platform_options.vcu118_bios:
            return self.config.platform_options.vcu118_bios
        else:
            return self.manager_config.sdk_path / "sdk" / "bbl-gfe" / self.config.cheri_target / "bbl"

    def _get_fpga_kernel(self):
        sdk = self.manager_config.sdk_path
        kernel = sdk / f"kernel-{self.config.cheri_target}.{self.config.kernel}"
        assert kernel.exists(), f"Missing kernel {kernel}"
        return kernel

    def _get_fpga_gdb(self):
        gdb = self.manager_config.sdk_path / "sdk" / "bin" / "gdb"
        gdb = Path("/homes/jrtc4/out/criscv-gdb")
        assert gdb.exists()
        return gdb

    def _get_fpga_cores(self):
        return str(self.config.platform_options.cores)

    def _get_fpga_pubkey(self):
        ssh_privkey = self.manager_config.ssh_key
        ssh_pubkey = ssh_privkey.with_suffix(".pub")
        with open(ssh_pubkey, "r") as fd:
            pubkey_str = fd.read()
        return pubkey_str.rstrip()

    def get_info(self):
        info = super().get_info()
        info.ssh_host = self.config.platform_options.vcu118_ip
        info.ssh_port = self.ssh_port
        return info

    async def _boot(self):
        # Assume that the FPGA image has been initialized already.
        # vcu118-run --bitfile design.bit --ltxfile design.ltx --bios bbl-gfe-riscv64-purecap --kernel kernel --gdb gdb --openocd /usr/bin/openocd --num-cores N --benchmark-config
        run_cmd = [
            "--bios",
            self._get_fpga_bios(), "--kernel",
            self._get_fpga_kernel(), "--gdb",
            self._get_fpga_gdb(), "--openocd", self.manager_config.openocd_path, "--num-cores",
            self._get_fpga_cores(), "--benchmark-config"
        ]
        ip_addr = self.config.platform_options.vcu118_ip
        run_cmd += [f"--test-command=ifconfig xae0 {ip_addr} netmask 255.255.255.0 up"]
        run_cmd += ["--test-command=mkdir -p /root/.ssh"]
        pubkey = self._get_fpga_pubkey()
        run_cmd += [f"--test-command=printf \'%s\' \'{pubkey}\' > /root/.ssh/authorized_keys"]
        run_cmd += ["--test-command=echo Instance startup done"]
        run_cmd = [str(arg) for arg in run_cmd]
        self.logger.debug("%s %s", self.runner_script, run_cmd)
        self._run_task = await aio.create_subprocess_exec(str(self.runner_script),
                                                          *run_cmd,
                                                          stdin=aio.subprocess.PIPE,
                                                          stdout=aio.subprocess.PIPE,
                                                          stderr=aio.subprocess.PIPE,
                                                          start_new_session=True)
        self.logger.debug("Spawned vcu118 runner pid=%d pgid=%d", self._run_task.pid, os.getpgid(self._run_task.pid))
        self._attach_io_observers(self._run_task)
        # Now wait for boot to reach login
        wait_pattern = re.compile(r"Instance startup done")
        await self._wait_io(lambda out, is_stderr: wait_pattern.match(out) if not is_stderr else False)
        # give some time to settle
        await aio.sleep(2)

    async def _shutdown(self):
        if self._ssh_ctrl_conn:
            await self._run_cmd("poweroff")
            await self._ssh_ctrl_conn.disconnect()
            await self._run_task.wait()
        elif self._run_task and self._run_task.returncode is None:
            self.logger.debug("Sending SIGTERM to vcu118 runner")
            os.killpg(os.getpgid(self._run_task, signal.SIGTERM))
        if len(self._io_tasks):
            await aio.gather(*self._io_tasks, return_exceptions=True)
        self._run_task = None
        self._ssh_ctrl_conn = None


class InstanceRequest:
    def __init__(self, loop: aio.AbstractEventLoop, owner: uuid.UUID, config: InstanceConfig):
        self.instance = loop.create_future()
        self.owner = owner
        self.config = config

    @property
    def name(self):
        return self.config.name

    def __str__(self):
        return f"InstanceRequest({self.config.name})"


class InstanceManager:
    """
    Helper object used to manage the lifetime of instances we run things on
    """
    def __init__(self, loop: aio.AbstractEventLoop, manager_config: "BenchmarkManagerConfig"):
        self.logger = new_logger("instance-manager")
        self.loop = loop
        self.manager_config = manager_config
        # Max concurrent instances
        if self.manager_config.concurrent_instances == 0:
            self.concurrent_instances = np.inf
        else:
            self.concurrent_instances = self.manager_config.concurrent_instances
        self._sched_task = loop.create_task(self._schedule())
        self._shutdown_drain = loop.create_task(self._shutdown_drain())
        # Cleanup event triggering shutdown
        self._sched_shutdown = aio.Event()

        # internal instance request queues that are waiting for active instances to be done
        self._instance_request_queues = defaultdict(list)
        # done instance queue to the scheduler
        self._done_queue = aio.Queue()
        # requests to the scheduler
        self._request_queue = aio.Queue()
        # internale idle instance list
        self._idle_instances = []
        # instances pending shutdown
        self._shutdown_instances = []
        self._shutdown_drain_wakeup = aio.Event()
        # Current number of instances
        self._instance_count = 0
        # Map running instances to the benchmark owning them
        self._active_instances = {}

        self.logger.info("Instance manager: instance_reuse=%s max_concurrent=%s",
                         "Enabled" if self.manager_config.reuse_instances else "Disabled", self.concurrent_instances)

    def _create_instance(self, config: InstanceConfig):
        if config.platform == InstancePlatform.QEMU:
            instance = CheribuildInstance(self, config)
        elif config.platform == InstancePlatform.VCU118:
            instance = VCU118Instance(self, config)
        else:
            self.logger.error(f"Unknown instance platform {config.platform}")
            raise InstanceManagerError("Unknown instance platform")
        self._instance_count += 1
        # creates main loop task and wait for the boot trigger
        instance.start()
        self.logger.debug("Created instance %s", instance.uuid)
        return instance

    def _alloc_instance(self, owner: uuid.UUID, instance: Instance):
        self._active_instances[owner] = instance
        if instance.status == InstanceStatus.INIT:
            instance.boot()
        self.logger.debug("Allocate instance %s(%s) to benchmark %s", instance.uuid, instance.config.name, owner)

    def _sched_dispose(self, instance):
        """
        Dispose of an instance that has finished running a benchmark
        """
        if self.manager_config.reuse_instances:
            # check to see whether we have queued request for this instance
            self.logger.debug("Sched: new idle instance %s(%s)", instance.uuid, instance.config.name)
            self._idle_instances.append(instance)
            self._sched_try_next()
        else:
            # release and add to shutdown queue
            instance.release()
            self._shutdown_instances.append(instance)
            self._shutdown_drain_wakeup.set()

    def _sched_try_next(self):
        """
        Look for an instance request without any currently active instance running.
        Prioritize requests for which we have idle instances.
        If one is found, schedule it; otherwise grab any other queued request and schedule it.
        """
        self.logger.debug("Sched: try scheduling from queues")
        idle = [i.config.name for i in self._idle_instances]
        # keep idle instances busy if possible
        for queue_name in idle:
            queue = self._instance_request_queues[queue_name]
            if not queue:
                continue
            request = queue.pop(0)
            handled = self._schedule_request(request)
            if not handled:
                self.logger.warning("Could not assign request to idle instance immediately")

        # otherwise try to schedule something new
        for name, queue in self._instance_request_queues.items():
            if not queue:
                continue
            handled = True
            while queue and handled:
                request = queue.pop(0)
                handled = self._schedule_request(request)
                if not handled:
                    queue.insert(0, request)
                    break
            if not handled:
                break

    def _schedule_request(self, request) -> bool:
        """
        Handle an instance request.
        Frist try to locate an idle instance to satisfy the request. If none is present,
        try to allocate a new instance. If this would exceed the concurrent instances limit,
        first we grab an idle instance and shut it down.
        """
        # If manager_config.reuse_instances is False the idle list will always be empty.
        assert len(
            self._idle_instances) == 0 or self.manager_config.reuse_instances, "Idle instance but reuse is forbidden"
        found_instance = None
        for instance in self._idle_instances:
            if request.name == instance.config.name:
                found_instance = instance
                break
        if found_instance:
            self.logger.debug("Sched: reuse idle instance %s(%s)", instance.uuid, instance.config.name)
            self._idle_instances.remove(found_instance)
            self._alloc_instance(request.owner, found_instance)
            request.instance.set_result(found_instance)
            return True
        assert self._instance_count <= self.concurrent_instances, "Too many booted instances"
        if self._instance_count < self.concurrent_instances:
            self.logger.debug("Sched: new instance for %s", request.name)
            # We did not find an idle instance, boot one if possible
            instance = self._create_instance(request.config)
            self._alloc_instance(request.owner, instance)
            request.instance.set_result(instance)
            return True
        elif len(self._idle_instances):
            # Shutdown an idle instance
            instance = self._idle_instances.pop(0)
            self.logger.debug("Sched: shutdown idle instance %s(%s)", instance.uuid, instance.config.name)
            instance.release()
            self._shutdown_instances.append(instance)
            self._shutdown_drain_wakeup.set()
        return False

    async def _schedule_loop(self):
        """
        Instance scheduler loop
        """
        async with multi_await() as select:
            select.add(self._request_queue.get)
            select.add(self._done_queue.get)
            select.add(self._sched_shutdown.wait)

            while True:
                complete, failures = await select.get()
                for err in failures:
                    if err is not None:
                        self.logger.error("Error while waiting on queue %s", err)
                request, instance, sched_shutdown = complete
                # First try to serve the done queue
                if instance is not None:
                    self.logger.debug("Instance %s returned to scheduler", instance.uuid)
                    self._sched_dispose(instance)
                    self._done_queue.task_done()
                if request is not None:
                    handled = self._schedule_request(request)
                    if not handled:
                        self._instance_request_queues[request.name].append(request)
                    self._request_queue.task_done()
                if sched_shutdown:
                    break
            self._drain_done_queue()

    def _drain_done_queue(self):
        # move any instance from the done queue into the idle list to be destroyed
        while not self._done_queue.empty():
            instance = self._done_queue.get_nowait()
            self.logger.debug("Drain %s from sched done queue", instance)
            self._idle_instances.append(instance)

    async def _schedule(self):
        self.logger.debug("Start instance scheduler loop")
        try:
            await self._schedule_loop()
        except aio.CancelledError:
            self.logger.warning("Scheduler loop killed")
            self._drain_done_queue()
        finally:
            self.logger.debug("Scheduler loop exited")

    async def _shutdown_drain(self):
        """
        Drain the shutdown instance queue
        """
        while True:
            try:
                await self._shutdown_drain_wakeup.wait()
                while self._shutdown_instances:
                    instance = self._shutdown_instances[0]
                    await instance.wait(InstanceStatus.DEAD)
                    self.logger.debug("Drain dead instance %s(%s)", instance.uuid, instance.config.name)
                    self._shutdown_instances.pop(0)
                    self._instance_count -= 1
                    # Trigger request scanning
                    self._sched_try_next()
                self._shutdown_drain_wakeup.clear()
            except aio.CancelledError:
                break

    async def request_instance(self, owner: uuid.UUID, config: InstanceConfig) -> InstanceInfo:
        request = InstanceRequest(self.loop, owner, config)
        self.logger.debug("Enqueue instance request for %s benchmark=%s", config.name, owner)
        self._request_queue.put_nowait(request)
        instance = await request.instance
        self.logger.debug("Wait for %s to boot", instance)
        await instance.wait(InstanceStatus.READY)
        # Here we are guaranteed that the instance is booted and ready to use
        return instance.get_info()

    async def release_instance(self, owner: uuid.UUID):
        instance = self._active_instances[owner]
        self.logger.debug("Release %s benchmark=%s", instance, owner)
        del self._active_instances[owner]
        self._done_queue.put_nowait(instance)

    async def shutdown(self):
        # Stop the scheduler and shutdown drain
        self._shutdown_drain.cancel()
        await self._shutdown_drain
        self._sched_shutdown.set()
        await self._sched_task

        # await aio.gather([self._sched_task, self._shutdown_task], return_exceptions=True)
        # Force-release everything and wait for shutdown
        for instance in self._active_instances.values():
            instance.release()
        for instance in self._idle_instances:
            instance.release()
        await aio.gather(*[i.wait(InstanceStatus.DEAD) for i in self._active_instances.values()],
                         return_exceptions=True)
        await aio.gather(*[i.wait(InstanceStatus.DEAD) for i in self._idle_instances], return_exceptions=True)
        await aio.gather(*[i.wait(InstanceStatus.DEAD) for i in self._shutdown_instances], return_exceptions=True)
        # XXX possibly report any errors here
        self.logger.debug("Instances shutdown completed")

    async def kill(self):
        # Stop the scheduler and shutdown drain
        self._sched_task.cancel()
        self._shutdown_drain.cancel()
        await self._sched_task
        await self._shutdown_drain
        # Force kill all instances
        for instance in self._active_instances.values():
            instance.stop()
        for instance in self._idle_instances:
            instance.stop()
        for instance in self._shutdown_instances:
            instance.stop()
        await aio.gather(*[i.task for i in self._active_instances.values()], return_exceptions=True)
        await aio.gather(*[i.task for i in self._idle_instances], return_exceptions=True)
        await aio.gather(*[i.task for i in self._shutdown_instances], return_exceptions=True)
        self.logger.debug("Instances killed successfully")
