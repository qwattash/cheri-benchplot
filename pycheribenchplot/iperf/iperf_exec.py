import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import polars as pl
import polars.selectors as cs
from marshmallow.validate import Regexp

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import output
from ..core.tvrs import TVRSExecConfig, TVRSExecTask


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

    In the standard configuration, the client sends data to the server. The
    `mode` parameter can be used to modify this behaviour.
    Only one stream is used by default. For multiple parellel streams, increase
    the `streams` configuration key.
    The CPU affinity is a comma-separated tuple of core IDs (0 to MAXCPU) that determine
    the sender/receiver affinity. This sets both the client and server affinity.
    """
    # yapf: disable
    protocol: IPerfProtocol = config_field(IPerfProtocol.TCP, desc="Protocol to use")
    transfer_mode: IPerfTransferLimit = config_field(
        IPerfTransferLimit.BYTES,
        desc="How the transfer limit is interpreted"
    )
    transfer_limit: str = config_field(
        "1G",
        desc="Transfer limit, format depends on `transfer_mode`",
        validate=Regexp(r"[0-9]+[KMG]?", error="Must be formatted as <N>{KMG}")
    )
    remote_host: str = config_field(
        "localhost",
        desc="Hostname of the server. By default we operate on localhost. "
        "When this is set to anything other than localhost, the server setup "
        "is expected to be done manually"
    )
    mode: IPerfMode = config_field(IPerfMode.CLIENT_SEND, desc="Stream mode")
    streams: int = config_field(1, desc="Number of parallel client streams")
    buffer_size: int|str = config_field("128K", desc="Size of the send/recv buffer (bytes)")
    mss: Optional[int] = config_field(None, desc="Set MSS size")
    window_size: Optional[int|str] = config_field(
        None,
        desc="Set socket buffer size (bytes) (indirectly the TCP window)"
    )
    warmup: Optional[int] = config_field(None, desc="Warmup seconds")
    cpu_affinity: Optional[str] = config_field(
        None,
        desc="CPU Affinity for send/receive sides",
        validate=Regexp(r"[0-9]+(,[0-9+])?", error="CPU Affinty must be of the form 'N[,M]'")
    )
    nodelay: bool = config_field(False, desc="Disable Nagle algorithm to send small packets immediately")
    use_ipv4: bool = config_field(True, desc="Force use of IPv4 vs IPv6")
    # yapf: enable


@dataclass
class IPerfConfig(TVRSExecConfig):
    """
    IPerf benchmark configuration.
    """
    iperf_path: Optional[ConfigPath] = config_field(
        None, desc="Path of iperf executable in the remote host, appended to PATH")
    use_localhost_server: bool = config_field(
        True, desc="Spawn server on localhost, if False, the scenario must specify a remote_host")


class IngestIPerfStats(PLDataFrameLoadTask):
    """
    Loader for stats data that produces a standard polars dataframe.
    """
    task_namespace = "iperf"
    task_name = "ingest-stats"

    def _load_one(self, path: Path) -> pl.DataFrame:
        """
        Load data for a benchmark run from the given target file.

        We extract the summary information from the "end" record.
        RTT measurements for latency are taken for each stream, so we
        generate one row for each stream for each iteration.
        """
        data = json.load(open(path, "r"))
        end_info = data["end"]
        df = pl.DataFrame(end_info["streams"])
        df = df.with_row_index("stream")
        # yapf: disable
        snd_df = (
            df.select("stream", "sender")
            .unnest("sender")
            .select("stream", cs.all().exclude("stream").name.prefix("snd_"))
        )
        rcv_df = (
            df.select("stream", "receiver")
            .unnest("receiver")
            .select("stream", cs.all().exclude("stream").name.prefix("rcv_"))
        )
        # yapf: enable
        df = snd_df.join(rcv_df, on="stream")

        assert end_info["sum_sent"]["bits_per_second"] == snd_df["snd_bits_per_second"].sum()
        assert end_info["sum_received"]["bits_per_second"] == rcv_df["rcv_bits_per_second"].sum()
        return df


class IPerfExecTask(TVRSExecTask):
    """
    Generate the iperf benchmark scripts

    Note that we restart the server for every iteration.
    This is done to reduce the amount of state that carries over from one iteration to
    the other. This is particularly relevent when temporal safety is used.
    """
    task_namespace = "iperf"
    task_name = "exec"
    task_config_class = IPerfConfig
    scenario_config_class = IPerfScenario
    script_template = "iperf.sh.jinja"
    public = True

    @output
    def stats(self):
        """IPerf json output"""
        return RemoteBenchmarkIterationTarget(self, "stats", ext="json", loader=IngestIPerfStats)

    def run(self):
        super().run()
        self.script.extend_context({
            "iperf_config": self.config,
            "iperf_gen_output_path": self.stats.shell_path_builder()
        })
        self.script.register_global("IPerfProtocol", IPerfProtocol)
        self.script.register_global("IPerfMode", IPerfMode)
        self.script.register_global("IPerfTransferLimit", IPerfTransferLimit)
