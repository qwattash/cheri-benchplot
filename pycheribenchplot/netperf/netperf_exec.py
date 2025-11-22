import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from marshmallow import ValidationError
from marshmallow.validate import Regexp

from ..core.artefact import DataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import output
from ..core.tvrs import TVRSExecConfig, TVRSExecTask
from ..iperf.iperf_exec import IPerfProtocol, IPerfTransferLimit

# Reuse parameters from IPerf, there is no reason to repeat the same here
NetPerfProtocol = IPerfProtocol
NetPerfTransferLimit = IPerfTransferLimit


class NetPerfMode(Enum):
    STREAM = "stream"
    MAERTS = "stream-reverse"
    RR = "round-trip"


suffix_value = {"K": 2**10, "M": 2**20, "G": 2**30}


class NetperfLoader(DataFrameLoadTask):
    task_namespace = "netperf"
    task_name = "loader"

    def _load_one(self, path):
        # The first row is the test header output, skip it for the columns
        return self._load_one_csv(path, skip_rows=1)


@dataclass
class NetperfScenario(Config):
    """
    Configuration for a single netperf scenario.

    This encodes netperf and netserver parameters.
    """

    protocol: NetPerfProtocol = config_field(
        NetPerfProtocol.TCP, desc="Protocol to use"
    )
    transfer_mode: NetPerfTransferLimit = config_field(
        NetPerfTransferLimit.BYTES, desc="How the transfer limit is interpreted"
    )
    mode: NetPerfMode = config_field(NetPerfMode.STREAM, desc="Test mode")
    transfer_limit: int | str = config_field(
        "1G", desc="Transfer limit, format depends on 'transfer_mode'"
    )
    remote_host: str = config_field(
        "localhost",
        desc="Hostname of the server. By default we operate on localhost. "
        "When this is set to anything other than localhost, the server setup "
        "is expected to be done manually",
    )
    use_ipv4: bool = config_field(True, desc="Force use of IPv4 vs IPv6")
    cpu_affinity: Optional[str] = config_field(
        None,
        desc="CPU Affinity for send/receive sides",
        validate=Regexp(
            r"[0-9]+(,[0-9+])?", error="CPU Affinty must be of the form 'N[,M]'"
        ),
    )
    buffer_size: int | str = config_field(
        "128K", desc="Size of the send/recv buffer (bytes)"
    )
    window_size: Optional[int | str] = config_field(
        None, desc="Set socket buffer size (bytes) (indirectly the TCP window)"
    )
    nodelay: bool = config_field(
        False, desc="Disable Nagle algorithm to send small packets immediately"
    )

    def __post_init__(self):
        super().__post_init__()
        # Ensure that the transfer limit is a valid integer
        m = re.match(r"([0-9]+)([KMG])?", str(self.transfer_limit))
        if not m:
            raise ValidationError("Must be formatted as <N>{KMG}")
        value = int(m.group(1))
        suffix = m.group(2)
        if suffix:
            value = value * suffix_value[suffix]
        self.transfer_limit = value


@dataclass
class NetperfRunConfig(TVRSExecConfig):
    """
    Configuration for a netperf benchmark instantiation.
    """

    netperf_path: Optional[ConfigPath] = config_field(
        None,
        desc="Path of netserver/netperf executables in the remote host, appended to PATH",
    )
    use_localhost_server: bool = config_field(
        True,
        desc="Spawn netserver on localhost. If False, the scenario msut specify a remote host",
    )


class NetperfExecTask(TVRSExecTask):
    """
    Run netperf tests.

    Not that this assumes that the netperf and netserver executables are available
    on the remote host.
    """

    task_namespace = "netperf"
    task_name = "exec"
    task_config_class = NetperfRunConfig
    script_template = "netperf.sh.jinja"
    scenario_config_class = NetperfScenario
    public = True

    @output
    def results(self):
        return RemoteBenchmarkIterationTarget(
            self, "stats", ext="csv", loader=NetperfLoader
        )

    def run(self):
        super().run()
        self.script.extend_context(
            {
                "netperf_config": self.config,
                "netperf_gen_output_path": self.results.shell_path_builder(),
            }
        )
        self.script.register_global("NetPerfProtocol", NetPerfProtocol)
        self.script.register_global("NetPerfMode", NetPerfMode)
        self.script.register_global("NetPerfTransferLimit", NetPerfTransferLimit)
