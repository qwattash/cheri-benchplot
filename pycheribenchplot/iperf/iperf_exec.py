from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath
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
    #: Protocol to use
    protocol: IPerfProtocol = field(default=IPerfProtocol.TCP, metadata={"by_value": True})

    #: How the transfer limit is interpreted
    transfer_mode: IPerfTransferLimit = field(default=IPerfTransferLimit.BYTES, metadata={"by_value": True})
    #: Transfer limit
    transfer_limit: int | str = 2**32

    #: Hostname of the server. By default we operate on localhost; however
    #: it should be possible to run the server/client on two different CHERI machines.
    #: The server setup is expected to be manually done in this case.
    remote_host: str = "localhost"

    #: Stream mode
    mode: IPerfMode = field(default=IPerfMode.CLIENT_SEND, metadata={"by_value": True})

    #: Number of parallel client streams
    streams: int = 1

    #: Size of the send/recv buffer (bytes)
    buffer_size: int = 128 * 2**10

    #: Set MSS size
    mss: Optional[int] = None

    #: Set socket buffer size (bytes) (indirectly the TCP window)
    window_size: Optional[int] = None

    #: Warmup seconds
    warmup: Optional[int] = None

    #: CPU Affinity for send/receive
    cpu_affinity: Optional[str] = None


@dataclass
class IPerfConfig(Config):
    """
    IPerf benchmark configuration.
    """
    #: Scenarios directory where to find the scenarios named by the configuration
    scenario_path: ConfigPath = Path("scenarios")

    #: Iperf PATH in the remote host
    iperf_path: Optional[ConfigPath] = None

    #: Use local server
    use_localhost_server: bool = True

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
