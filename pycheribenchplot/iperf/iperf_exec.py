from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field, validate_dir_exists
from ..core.task import output
from ..core.tvrs import TVRSExecTask


class IPerfMode(Enum):
    #: Client sends, server receives.
    CLIENT_SEND = "client-send"
    #: Client receives, server sends.
    CLIENT_RECV = "client-recv"
    #: Bidirectional, both client and server send and receive data.
    BIDIRECTIONAL = "bidirectional"


class IPerfTransferLimit(Enum):
    #: Limit by number of bytes
    BYTES = "bytes"
    #: Limit by seconds
    TIME = "time"
    #: Limit by number of packets
    PACKETS = "pkts"


class IPerfProtocol(Enum):
    #: Use TCP (default)
    TCP = "tcp"
    #: Use UDP
    UDP = "udp"
    #: Use SCTP
    SCTP = "sctp"


@dataclass
class IPerfScenario(Config):
    """
    The iperf benchmark parameters are encoded in a scenario json file.

    This is the configuration schema that loads the iperf scenario to run.
    """
    protocol: IPerfProtocol = config_field(IPerfProtocol.TCP, desc="Protocol to use")
    transfer_mode: IPerfTransferLimit = config_field(IPerfTransferLimit.BYTES,
                                                     desc="How the transfer limit is interpreted")
    transfer_limit: int | str = config_field(2**32, desc="Transfer limit")
    remote_host: str = config_field(
        "localhost",
        desc="Hostname of the server. By default we operate on localhost. When this is set to anything other "
        "than localhost, the server setup is expected to be done manually")
    mode: IPerfMode = config_field(IPerfMode.CLIENT_SEND, desc="Stream mode")
    streams: int = config_field(1, desc="Number of parallel client streams")
    buffer_size: int = config_field(128 * 2**10, desc="Size of the send/recv buffer (bytes)")
    mss: Optional[int] = config_field(None, desc="Set MSS size")
    window_size: Optional[int] = config_field(None, desc="Set socket buffer size (bytes) (indirectly the TCP window)")
    warmup: Optional[int] = config_field(None, desc="Warmup seconds")
    cpu_affinity: Optional[str] = config_field(None, desc="CPU Affinity for send/receive sides")


@dataclass
class IPerfConfig(Config):
    """
    IPerf benchmark configuration.
    """
    scenario_path: ConfigPath = config_field(
        Path("scenarios"),
        desc="Scenarios directory where to find the scenarios named by the configuration",
        validate=validate_dir_exists)
    iperf_path: Optional[ConfigPath] = config_field(
        None, desc="Path of iperf executable in the remote host, appended to PATH")
    use_localhost_server: bool = config_field(
        True, desc="Spawn server on localhost, if False, the scenario must specify a remote_host")

    #: Enable hwpmc measurement
    hwpmc: bool = False


class IPerfExecTask(TVRSExecTask):
    """
    Generate the iperf benchmark scripts

    Note that we restart the server for every iteration.
    This is done to reduce the amount of state that carries over from one iteration to
    the other. This is particularly relevent when temporal safety is used.
    """
    public = True
    task_namespace = "iperf"
    task_name = "exec"
    task_config_class = IPerfConfig

    @output
    def stats(self):
        """IPerf json output"""
        return RemoteBenchmarkIterationTarget(self, "stats", ext="json")

    def hwpmc(self):
        """The remote profiling output target"""
        return RemoteBenchmarkIterationTarget(self, "hwpmc", ext="json")

    def run(self):
        self.script.set_template("iperf.sh.jinja")
        scenario = self.config.scenario_path / self.benchmark.parameters["scenario"]
        scenario_config = IPerfScenario.load_json(scenario.with_suffix(".json"))
        self.script.extend_context({
            "scenario_config": scenario_config,
            "iperf_config": self.config,
            "iperf_output_path": self.stats.remote_paths(),
        })
        self.script.register_global("IPerfProtocol", IPerfProtocol)
        self.script.register_global("IPerfMode", IPerfMode)
        self.script.register_global("IPerfTransferLimit", IPerfTransferLimit)
