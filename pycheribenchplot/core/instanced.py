import logging
import asyncio as aio
import dbm
import os
import re
import signal
import uuid
import copy
import typing
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

import asyncssh
import zmq
import zmq.asyncio as zaio

from .config import Config, TemplateConfig, path_field

ctx = zaio.Context()


class InstanceDaemonError(Exception):
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
class QemuInstanceConfig(TemplateConfig):
    """QEMU-specific instance configuration"""
    qemu_trace_backend: str = None


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
    platform_options: typing.Union[QemuInstanceConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = f"{self.platform}-{self.cheri_target}-{self.kernelabi}"


@dataclass
class InstanceDaemonConfig(Config):
    concurrent_instances: int = 4
    verbose: bool = False
    ssh_key: Path = path_field("~/.ssh/id_rsa")
    sdk_path: Path = path_field("~/cheri/cherisdk")
    cheribuild_path: Path = path_field("~/cheri/cheribuild/cheribuild.py")
    instances: list[InstanceConfig] = field(default_factory=list)


class BenchmarkStatus(Enum):
    """
    Status of a benchmark run on an instance as seen by the clients
    QUEUED: enqueued in the instance run_queue
    PENDING: currently assigned to the instance but not yet ready to run
    READY: the benchmark driver can connect to the instance and run
    ERROR: an error occurred, the benchmark no longer owns the instance
    """
    QUEUED = "queued"
    PENDING = "pending"
    READY = "ready"
    ERROR = "error"


class InstanceStatus(Enum):
    """
    Status of an instance managed by the daemon
    INIT: instance initialized but not yet started
    BOOT: instance booting
    RUNNING: instance finished booting, the control connection is available
    RESET: instance is resetting after a release, may transition
    to BOOT or RUNNING
    IDLE: running but no benchmark is assigned, so we are waiting
    DEAD: instance is dead because of shutdown or because of an error
    """
    INIT = "initialized"
    BOOT = "booting"
    RUNNING = "running"
    RESET = "reset"
    IDLE = "idle"
    DEAD = "dead"


class InstanceOp(Enum):
    """
    Operations that can be requested to the instance daemon
    SUMMARY: return summary of tasks on the daemon
    REQUEST: schedule a benchmark to run on an instance
    RELEASE: release instance after a benchmark is done with it
    """
    SUMMARY = "summary"
    REQUEST = "request"
    RELEASE = "release"


@dataclass
class _Message:
    op: InstanceOp
    owner: typing.Optional[uuid.UUID] = None
    config: typing.Optional[InstanceConfig] = None


@dataclass
class BenchmarkInfo:
    # Status of the benchmark
    status: BenchmarkStatus
    # ID of the instance
    uuid: "typing.Optional[uuid.UUID]" = None
    # SSH host to reach the instance
    ssh_host: typing.Optional[str] = None
    # SSH port to reach the instance
    ssh_port: typing.Optional[int] = None
    # If the instance has qemu trace output, it will be
    # sent to this file
    qemu_trace_file: typing.Optional[Path] = None
    # If the instance has qemu-perfetto trace output, it will
    # be sent to this file
    qemu_pftrace_file: typing.Optional[Path] = None


@dataclass
class _Reply:
    benchmarks: list[BenchmarkInfo]


@dataclass
class _Notification:
    owner: uuid.UUID
    benchmark: BenchmarkInfo


@dataclass
class _InstanceMessage:
    op: InstanceOp
    owner: uuid.UUID
    config: typing.Optional[InstanceConfig] = None


class InstanceClient:
    """
    Client Interface to the zmq service.
    """
    class PendingBenchmark:
        def __init__(self):
            self.event = aio.Event()
            self.benchmark_info = None

        def dispatch(self, info: BenchmarkInfo):
            if self.benchmark_info is not None:
                raise InstanceDaemonError("Duplicate notification for pending benchmark")
            self.benchmark_info = info
            self.event.set()

        async def wait(self):
            await self.event.wait()

    def __init__(self, loop: aio.AbstractEventLoop):
        self.logger = logging.getLogger("cheri-instanced-client")
        self.loop = loop
        self.cmd_socket = ctx.socket(zmq.REQ)
        self.cmd_socket.connect("tcp://127.0.0.1:15555")
        self.cmd_socket_mutex = aio.Lock()
        self.notify_socket = ctx.socket(zmq.SUB)
        self.notify_socket.connect("tcp://127.0.0.1:15556")
        self.timeout = 5.0
        # Pending uuids -> received benchmark info
        self._pending_events = {}
        self._notifier_task = None

    def start(self):
        self._notifier_task = self.loop.create_task(self._notifier_loop())

    async def stop(self):
        if self._notifier_task:
            self._notifier_task.cancel()
            await aio.gather(self._notifier_task, return_exceptions=True)
        self.notify_socket.close()

    async def _notifier_loop(self):
        self.logger.debug("Start client notification listener")
        self.notify_socket.subscribe("")
        while True:
            msg = await self.notify_socket.recv_pyobj()
            if msg.owner not in self._pending_events:
                continue
            if (msg.benchmark.status == BenchmarkStatus.READY or msg.benchmark.status == BenchmarkStatus.ERROR):
                self._pending_events[msg.owner].dispatch(msg.benchmark)

    async def _cmd_msg(self, op: InstanceOp, owner: uuid.UUID, config: InstanceConfig = None, timeout=None) -> _Reply:
        if timeout is None:
            timeout = self.timeout
        req = _Message(op=op, owner=owner, config=config)
        async with self.cmd_socket_mutex:
            # We need to serialize command request-reply pairs as zmq REQ/REP sockets do
            # not like multiple in-flight messages
            await self.cmd_socket.send_pyobj(req)
            if timeout > 0:
                reply = await aio.wait_for(self.cmd_socket.recv_pyobj(), timeout=timeout)
            else:
                reply = await self.cmd_socket.recv_pyobj()
        return reply

    async def _summary_msg(self) -> list[BenchmarkInfo]:
        reply = await self._cmd_msg(InstanceOp.SUMMARY, None)
        return reply.benchmarks

    async def _request_msg(self, owner, config, timeout=0) -> BenchmarkInfo:
        reply = await self._cmd_msg(InstanceOp.REQUEST, owner, config, timeout)
        return reply.benchmarks[0]

    async def _release_msg(self, owner, timeout=None) -> BenchmarkInfo:
        reply = await self._cmd_msg(InstanceOp.RELEASE, owner, None, timeout=timeout)
        return reply.benchmarks[0]

    async def _wait_for_instance(self, owner: uuid.UUID):
        pending = InstanceClient.PendingBenchmark()
        self._pending_events[owner] = pending
        await pending.wait()
        self.logger.debug("Got pending instance %s status for %s", pending.benchmark_info.uuid, owner)
        del self._pending_events[owner]
        return pending.benchmark_info

    async def request_instance(self, owner: uuid.UUID, config: InstanceConfig) -> BenchmarkInfo:
        try:
            # Check that the daemon is up
            await self._summary_msg()
            # Request and instance, the reply will arrive after the instance is ready for us
            self.logger.debug("Request instance for %s", owner)
            info = await self._request_msg(owner, config)
            if info.status == BenchmarkStatus.READY:
                return info
            elif info.status == BenchmarkStatus.ERROR:
                self.logger.error("Failed to request instance for %s", owner)
            elif info.status == BenchmarkStatus.QUEUED or info.status == BenchmarkStatus.PENDING:
                self.logger.debug("Waiting for instance allocation")
                info = await self._wait_for_instance(owner)
                if info.status != BenchmarkStatus.READY:
                    self.logger.error("Failed to request instance for %s", owner)
                    return None
                return info
            else:
                self.logger.error("Invalid requested instance state")
                await self._release_msg(owner)
            return None
        except aio.TimeoutError:
            self.logger.error("Instance request timed out, is the daemon running?")

    async def release_instance(self, owner: uuid.UUID, inst: BenchmarkInfo):
        try:
            reply = await self._release_msg(owner)
        except aio.TimeoutError:
            self.logger.error("Instance request timed out, is the daemon running?")
        finally:
            self.logger.debug("Released instance: %s (%s:%d)", inst.uuid, inst.ssh_host, inst.ssh_port)


class Instance(ABC):
    last_ssh_port = 12000

    def __init__(self, daemon: "InstanceDaemon", config: InstanceConfig):
        self.daemon = daemon
        # Main daemon event loop
        self.event_loop = daemon.loop
        # Daemon configuration
        self.daemon_config: InstanceDaemonConfig = daemon.config
        # Instance configuration
        self.config = config
        # Unique ID of the instance
        self.uuid = uuid.uuid4()
        self.logger = logging.getLogger(f"{self.uuid}")
        # SSH port allocated for the benchmarks to connect to the instance
        self.ssh_port = self._get_ssh_port()
        # The task associated to this instance main loop
        self.task = None
        # Control connection to the instance
        self._ssh_ctrl_conn = None
        # When set, this instance is allocated to a benchmark identified by the ID
        # in the owner field
        self.owner = None
        # Queue of benchmark_id waiting to run on the instance
        self.run_queue = []
        # Event signaling the presence of items in the run_queue
        self.run_queue_avail = aio.Event()
        # Signal when the instance is released by the benchmark currently owning it
        self.release_event = aio.Event()
        # Signal when a new benchmark exits the queue and acquires the instance
        self.benchmark_acquired = aio.Event()
        # Status of the instance, should be updated via set_status()
        self.status = InstanceStatus.INIT

    def _get_ssh_port(self):
        port = Instance.last_ssh_port
        Instance.last_ssh_port += 1
        return port

    def get_client_info(self, benchmark_id: uuid.UUID):
        if self.status == InstanceStatus.DEAD:
            bench_status = BenchmarkStatus.ERROR
        elif self.owner == benchmark_id:
            if self.status == InstanceStatus.RUNNING:
                bench_status = BenchmarkStatus.READY
            else:
                bench_status = BenchmarkStatus.PENDING
        else:
            # should have a way to check if the ID is really in the queue
            bench_status = BenchmarkStatus.QUEUED
        info = BenchmarkInfo(status=bench_status, uuid=self.uuid, ssh_port=self.ssh_port)
        return info

    def schedule(self, benchmark_id: uuid.UUID):
        if self.status == InstanceStatus.INIT or self.status == InstanceStatus.IDLE:
            # We can fast-path and avoid passing via the queue
            pass
        self._run_queue_put(benchmark_id)

    def retire(self, benchmark_id: uuid.UUID):
        try:
            self.run_queue.remove(benchmark_id)
        except ValueError:
            self.logger.warning("Attempt to retire unscheduled benchmark %s on %s", benchmark_id, self.uuid)

    def set_status(self, next_status):
        self.logger.debug("STATUS %s -> %s", self.status, next_status)
        self.status = next_status
        self.daemon.notify_instance_update(self)

    async def _run_queue_get(self) -> uuid.UUID:
        await self.run_queue_avail.wait()
        benchmark_id = self.run_queue.pop(0)
        if len(self.run_queue) == 0:
            self.run_queue_avail.clear()
        return benchmark_id

    def _run_queue_put(self, benchmark_id: uuid.UUID):
        self.run_queue.append(benchmark_id)
        if not self.run_queue_avail.is_set():
            self.run_queue_avail.set()

    async def _next_scheduled_benchmark(self):
        assert self.owner is None, "Owner was not reset"
        next_benchmark = await self._run_queue_get()
        self.benchmark_acquired.set()
        self.owner = next_benchmark
        self.logger.info("Running benchmark %s on instance %s", next_benchmark, self.uuid)
        self.benchmark_acquired.clear()

    async def _run_cmd(self, prog, *args):
        cmdline = f"{prog}" + " ".join(args)
        self.logger.debug("exec %s", cmdline)
        result = await self._ssh_ctrl_conn.run(cmdline)
        if result.returncode != 0:
            self.logger.error("Command failed with %d: %s", result.returncode, result.stderr)
            raise InstanceDaemonError("Control command failed")
        self.logger.debug("%s", result.stdout)

    async def _make_control_connection(self):
        """
        Make sure that the cheribsd host accepts all environment variables we are going to send.
        To do this, update the configuration and restart sshd
        """
        ssh_keyfile = self.daemon_config.ssh_key.expanduser()
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
    async def _reset(self):
        """
        Reset the instance for another benchmark to run.
        This may reboot the instance if needed.
        """
        ...

    @abstractmethod
    async def _shutdown(self):
        """
        Shutdown the instance. This can be called when an error occurs or when
        the instance is not needed anymore.
        """
        ...

    async def main_loop(self):
        """
        Main loop entered when the instance is created.
        We first wait for at least a benchmark to request the instance to boot.
        Once we have booted, we wait for the benchmark to run. When the benchmark
        releases the instance, reset it and grab the next benchmark from the queue.
        """
        try:
            while True:
                await self._next_scheduled_benchmark()
                if self.status == InstanceStatus.INIT:
                    # We need to boot
                    self.set_status(InstanceStatus.BOOT)
                    await self._boot()
                    await self._make_control_connection()
                self.set_status(InstanceStatus.RUNNING)
                # Now we are running, wait for the benchmark to release the instance
                await self.release_event.wait()
                self.release_event.clear()
                self.set_status(InstanceStatus.RESET)
                await self._reset()
                self.run_queue.task_done()
                self.set_status(InstanceStatus.IDLE)
        except aio.CancelledError as ex:
            self.logger.debug("Instance loop cancelled")
            await self._shutdown()
            raise ex
        except Exception as ex:
            self.logger.error("Fatal error: %s - shutdown instance", ex)
            import traceback
            traceback.print_tb(ex.__traceback__)
            await self._shutdown()
        finally:
            self.set_status(InstanceStatus.DEAD)
            # Release all waiters on the instance
            self.benchmark_acquired.set()
            self.logger.debug("Exiting benchmark instance main loop")

    def release(self):
        """
        Release instance from the current benchmark. This indicates that the benchmark
        is done running and we can trigger the instance reset before advancing to the
        next benchmark in the run_queue.
        """
        self.logger.info("Benchmark %s release instance", self.owner)
        self.owner = None
        self.release_event.set()

    def stop(self):
        self.logger.info("Stopping instance")
        if self.task:
            self.task.cancel()

    def __str__(self):
        return (f"Instance {self.uuid} {self.config.platform} cheribsd:{self.config.cheribsd} " +
                f"kernel:{self.config.kernel}")


class CheribuildInstance(Instance):
    def __init__(self, daemon, config):
        super().__init__(daemon, config)
        self._cheribuild = self.daemon_config.cheribuild_path.expanduser()
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
        return Path(f"/tmp/trace-{self.uuid}.out")

    def _get_qemu_perfetto_sink(self):
        # This will be a binary file containing serialized perfetto protobufs.
        return Path(f"/tmp/pftrace-{self.uuid}.pb")

    def get_client_info(self, benchmark_id: uuid.UUID):
        info = super().get_client_info(benchmark_id)
        info.ssh_host = "localhost"
        info.qemu_trace_file = self._get_qemu_trace_sink()
        info.qemu_pftrace_file = self._get_qemu_perfetto_sink()
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
        # Extra qemu options and tracing tags?
        plat_opts = self.config.platform_options
        if (plat_opts and isinstance(plat_opts, QemuInstanceConfig) and plat_opts.qemu_trace):
            common_trace_sink = str(self._get_qemu_trace_sink())
            qemu_options = ["-D", common_trace_sink]
            # Other backends are not as powerful, so focus on this one
            qemu_options += [
                "--cheri-trace-backend", "perfetto", "--cheri-trace-perfetto-logfile",
                str(self._get_qemu_perfetto_sink()), "--cheri-trace-perfetto-categories",
                ",".join(plat_opts.qemu_trace_categories)
            ]
            run_cmd += [self._run_option("extra-options"), " ".join(qemu_options)]

        self.logger.debug("%s %s", self._cheribuild, run_cmd)
        self._cheribuild_task = await aio.create_subprocess_exec(self._cheribuild,
                                                                 *run_cmd,
                                                                 stdin=aio.subprocess.PIPE,
                                                                 stdout=aio.subprocess.PIPE,
                                                                 stderr=aio.subprocess.PIPE,
                                                                 start_new_session=True,
                                                                 loop=self.event_loop)
        self.logger.debug("Spawned cheribuild pid=%d pgid=%d", self._cheribuild_task.pid,
                          os.getpgid(self._cheribuild_task.pid))
        # Now wait for the boot to complete and start sshd
        sshd_pattern = re.compile("Starting sshd\.")
        ssh_keyfile = self.daemon_config.ssh_key.expanduser()
        while self._cheribuild_task.returncode is None:
            raw_out = await self._cheribuild_task.stdout.readline()
            if raw_out:
                out = raw_out.decode("ascii")
                if self.daemon_config.verbose:
                    self.logger.debug(out.rstrip())
                if sshd_pattern.match(out):
                    break
        if self._cheribuild_task.returncode is not None:
            self.logger.error("Unexpected shutdown")
            out, err = await self._cheribuild_task.communicate()
            raise InstanceDaemonError(f"Instance died before sshd could startup: {err}")
        # give qemu some time
        await aio.sleep(5)

    async def _reset(self):
        """Can reuse the qemu instance directly"""
        with open(self._get_qemu_trace_sink(), "w") as fd:
            fd.truncate(0)
        return

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


class InstanceDaemon:
    """
    The service managing instances for the benchmarks to run on
    """
    def __init__(self, config: InstanceDaemonConfig):
        self.config = config
        self.logger = logging.getLogger("cheri-instanced")
        # Socket used by clients to request things
        self.cmd_socket = ctx.socket(zmq.REP)
        # # Socket used by the daemon to push status updates to clients
        self.notify_socket = ctx.socket(zmq.PUB)
        try:
            self.cmd_socket.bind("tcp://127.0.0.1:15555")
            self.notify_socket.bind("tcp://127.0.0.1:15556")
            self.db = dbm.open(".cheri-instance-cache", "c")
        except zmq.ZMQError as e:
            self.logger.error("Can not start cheri-instanced: %s." + "Maybe the daemon is already running?", e)
            exit(1)
        self.loop = aio.get_event_loop()
        # Running instances
        self.active_instances = {}
        # Instances with status updates
        self.instance_update_queue = aio.Queue()
        # Benchmark ID mapped to the owned instance
        self.owners = {}
        # Protect owners registry against concurrent requests
        self.owners_mtx = aio.Lock()
        # I/O task for zmq socket
        self.zmq_task = None
        self.notifier_task = None

    def _create_fpga_instance(self, config: InstanceConfig):
        self.logger.warning("Running on fpga not yet supported")
        return None

    def create_instance(self, config: InstanceConfig):
        instance = None
        try:
            if config.platform == InstancePlatform.QEMU:
                instance = CheribuildInstance(self, config)
            elif config.platform == InstancePlatform.FPGA:
                instance = self._create_fpga_instance(config)
        except aio.CancelledError as ex:
            # Forward cancellations
            raise ex
        except Exception as ex:
            self.logger.error("Failed to create instance for %s: %s", config, ex)
            return

        if instance is None:
            raise InstanceDaemonError(f"Invalid instance platform {config.platform}")
        self.active_instances[instance.uuid] = instance
        instance.task = self.loop.create_task(instance.main_loop())
        instance.task.add_done_callback(lambda future: self._recycle_instance(instance))
        self.logger.debug("Created instance %s", instance.uuid)
        return instance

    def start(self):
        try:
            self.loop.run_until_complete(self._daemon_main())
        except KeyboardInterrupt:
            self.logger.info("User requested shutdown, cleanup")
            self._shutdown()
        except Exception as ex:
            self.logger.error("Fatal error: %s - killing daemon", ex)
            self._shutdown()
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            self.logger.info("Shutdown completed")

    def _shutdown(self):
        """Helper to run the _daemon_shutdown task"""
        try:
            self.loop.run_until_complete(self._daemon_shutdown())
        except Exception as ex:
            self.logger.error("Fatal error during shutdown %s", ex)
            self._kill()

    def _recycle_instance(self, instance):
        """
        Called whenever an instance shuts down completely.
        Remove it form the instances for now instead of potentially rebooting.
        """
        self.logger.debug("Cleanup dead instance %s", instance.uuid)
        self.active_instances.pop(instance.uuid, None)

    async def _daemon_shutdown(self):
        """
        Cancel and terminate all instances.
        1. stop the zmq task so that no new instance can be created.
        2. run the graceful stop task on all instances. This should cause the termination of
        the instance tasks
        3. wait for all instance tasks to complete
        """
        self.logger.debug("Global shutdown")
        self.zmq_task.cancel()
        for i in self.active_instances.values():
            i.stop()
        result = await aio.gather(*[i.task for i in self.active_instances.values()], return_exceptions=True)
        self.logger.debug("Instance tasks completed: errs=%s", result)
        for e in result:
            if isinstance(e, Exception):
                import traceback
                traceback.print_tb(e.__traceback__)
        # Garbage-collect all other service tasks
        self.notifier_task.cancel()
        await aio.gather(self.notifier_task, return_exceptions=True)
        self.logger.info("Global shutdown done")

    async def _daemon_main(self):
        try:
            self.zmq_task = self.loop.create_task(self._zmq_loop())
            self.notifier_task = self.loop.create_task(self._zmq_notifier_loop())
            # preload_task = self.loop.create_task(self._preload_from_config())
            # await preload_task
            await aio.gather(self.zmq_task, self.notifier_task)
        except aio.CancelledError:
            self.logger.info("ZMQ loop exited")

    async def _resurrect(self):
        """Check cached dead instances to see if we can restart them"""
        # XXX TODO
        return

    async def _preload_from_config(self):
        await self._resurrect()
        conf_instances = []
        for conf in self.config.instances:
            conf_instances.append(self.create_instance(conf))
        await aio.gather(*[i.task for i in conf_instances])

    def _schedule_instance(self, owner: uuid.UUID, config: InstanceConfig):
        inst = None
        for running_inst in self.active_instances.values():
            # Note: this compares all fields
            if config == running_inst.config and running_inst.status != InstanceStatus.DEAD:
                inst = running_inst
                break
        if inst is None:
            inst = self.create_instance(config)
        inst.schedule(owner)
        return inst

    def _find_instance_for(self, bench_id: uuid.UUID) -> typing.Optional[Instance]:
        """
        Find the instance we allocated the given benchmark
        """
        for instance in self.active_instances.values():
            if instance.owner == bench_id:
                return instance
            # Check the instance run queue
        return None

    def _error_reply(self):
        reply = _Reply(benchmarks=[BenchmarkInfo(status=BenchmarkStatus.ERROR)])
        return reply

    def notify_instance_update(self, instance: Instance):
        """
        Notify a change in an instance state to the subscribers. This is used
        to push the notification when a benchmark owns an instance to run on.
        This is done by pushing the instance to a queue that is drained by the
        notifier loop.
        """
        self.instance_update_queue.put_nowait(instance)

    async def _summary(self, msg):
        return _Reply(benchmarks=[])

    async def _request(self, msg):
        if msg.config is None:
            self.logger.error("No config in REQUEST message")
            return self._error_reply()
        if msg.owner in self.owners:
            self.logger.error("Duplicate scheduling request from %s", msg.owner)
            return self._error_reply()
        self.logger.info("Requested instance for benchmark %s", msg.owner)
        instance = self._schedule_instance(msg.owner, msg.config)
        info = instance.get_client_info(msg.owner)
        return _Reply(benchmarks=[info])

    async def _release(self, msg):
        instance = self._find_instance_for(msg.owner)
        if instance is None:
            self.logger.warning(f"No scheduled instance for %s, can not release", msg.owner)
        else:
            if instance.owner == msg.owner:
                self.logger.debug("Release locked instance for %s", msg.owner)
                instance.release()
            else:
                self.logger.debug("Release queued instance for %s", msg.owner)
                instance.retire(msg.owner)
        info = instance.get_client_info(msg.owner)
        return _Reply(benchmarks=[info])

    async def _process_message(self, msg: _InstanceMessage) -> BenchmarkInfo:
        if msg.op == InstanceOp.REQUEST:
            reply = await self._request(msg)
        elif msg.op == InstanceOp.SUMMARY:
            reply = await self._summary(msg)
        elif msg.op == InstanceOp.RELEASE:
            reply = await self._release(msg)
        else:
            reply = self._error_reply()
        return reply

    async def _zmq_notifier_loop(self):
        self.logger.info("Started cheri-instanced notifier zmq at localhost:15556")
        while True:
            instance = await self.instance_update_queue.get()
            n = _Notification(instance.owner, instance.get_client_info(instance.owner))
            await self.notify_socket.send_pyobj(n)

    async def _zmq_loop(self):
        self.logger.info("Started cheri-instanced zmq at localhost:15555")
        while True:
            msg = await self.cmd_socket.recv_pyobj()
            reply = await self._process_message(msg)
            await self.cmd_socket.send_pyobj(reply)
