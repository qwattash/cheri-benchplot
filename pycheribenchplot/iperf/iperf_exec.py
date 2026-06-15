import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl
import polars.selectors as cs
from marshmallow.validate import Regexp

from ..core.artefact import DataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, ConfigPath, config_field
from ..core.task import ExecutionTask, output


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
class IPerfConfig(Config):
    """
    The iperf benchmark parameters.

    In the standard configuration, the client sends data to the server. The
    `mode` parameter can be used to modify this behaviour.
    Only one stream is used by default. For multiple parellel streams, increase
    the `streams` configuration key.
    The CPU affinity is a comma-separated tuple of core IDs (0 to MAXCPU) that determine
    the sender/receiver affinity. This sets both the client and server affinity.
    """

    iperf_path: ConfigPath | None = config_field(
        None, desc="Path of iperf executable in the remote host, appended to PATH"
    )
    spawn_server: bool = config_field(
        True,
        desc="Spawn server, if False, must specify a different server_addr",
    )
    server_addr: str = config_field(
        "localhost",
        desc="Hostname of the server. By default we operate on localhost. "
        "When this is set to anything other than localhost, the server setup "
        "is expected to be done manually",
    )
    protocol: IPerfProtocol = config_field(IPerfProtocol.TCP, desc="Protocol to use")
    transfer_mode: IPerfTransferLimit = config_field(
        IPerfTransferLimit.BYTES, desc="How the transfer limit is interpreted"
    )
    transfer_limit: str = config_field(
        "1G",
        desc="Transfer limit, format depends on `transfer_mode`",
        validate=Regexp(r"[0-9]+[KMG]?", error="Must be formatted as <N>{KMG}"),
    )
    mode: IPerfMode = config_field(IPerfMode.CLIENT_SEND, desc="Stream mode")
    streams: int = config_field(1, desc="Number of parallel client streams")
    buffer_size: str = config_field("128K", desc="Size of the send/recv buffer (bytes)")
    mss: int | None = config_field(None, desc="Set MSS size")
    window_size: int | str | None = config_field(
        None, desc="Set socket buffer size (bytes) (indirectly the TCP window)"
    )
    warmup: int | None = config_field(None, desc="Warmup seconds")
    server_cpu: int | None = config_field(
        None, desc="Server CPU affinity, when running on localhost"
    )
    client_cpu: int | None = config_field(None, desc="Client CPU affinity")
    nodelay: bool = config_field(
        False, desc="Disable Nagle algorithm to send small packets immediately"
    )
    use_ipv4: bool = config_field(True, desc="Force use of IPv4 vs IPv6")


class LoadIPerfStats(DataFrameLoadTask):
    """
    Loader for stats data that produces a standard polars dataframe.
    """

    task_namespace = "iperf"
    task_name = "ingest-stats"

    @property
    def data_columns(self) -> list[str]:
        exec_task = self.target.task

        cols = ["seconds", "bytes", "bits_per_second", "side"]
        if exec_task.config.protocol == IPerfProtocol.UDP:
            cols += ["jitter_ms", "packets", "lost_packets", "lost_percent"]
        return cols

    def _load_one(self, path: Path) -> pl.DataFrame:
        """
        Load data for a benchmark run from the given target file.

        We extract the summary information from the "end" record.
        RTT measurements for latency are taken for each stream, so we
        generate one row for each stream for each iteration.
        """

        # Expect a single json data entry
        data = json.load(open(path, "r"))
        end_info = data["end"]

        # Double check that there are no errors
        if "error" in data:
            self.logger.fatal("Detected error in iperf data: %s", data["error"])
            raise RuntimeError("Dataset corruption")

        snd = (
            pl.DataFrame(end_info["sum_sent"])
            .with_columns(
                cs.numeric().exclude("packets").cast(pl.Float64),
                pl.lit("sender").alias("side"),
            )
            .select(self.data_columns)
        )
        rcv = (
            pl.DataFrame(end_info["sum_received"])
            .with_columns(
                cs.numeric().exclude("packets").cast(pl.Float64),
                pl.lit("receiver").alias("side"),
            )
            .select(self.data_columns)
        )
        df = pl.concat([snd, rcv], how="vertical", rechunk=True)

        test_info = data["start"]["test_start"]
        df = df.with_columns(
            pl.lit(data["start"]["sndbuf_actual"]).alias("sndbuf_bytes"),
            pl.lit(data["start"]["rcvbuf_actual"]).alias("rcvbuf_bytes"),
            pl.lit(test_info["blksize"]).alias("block_size_bytes"),
        )
        return df


class IPerfExecTask(ExecutionTask):
    """
    Generate the iperf benchmark scripts

    Note that we restart the server for every iteration.
    This is done to reduce the amount of state that carries over from one iteration to
    the other. This is particularly relevent when temporal safety is used.
    """

    task_namespace = "iperf"
    task_name = "exec"
    task_config_class = IPerfConfig
    public = True

    @output
    def stats(self):
        return RemoteBenchmarkIterationTarget(
            self, "stats", ext="json", loader=LoadIPerfStats
        )

    def run(self):
        super().run()
        self.script.set_template("iperf.sh.jinja")
        self.script.extend_context(
            {
                "iperf_config": self.config,
                "iperf_gen_output_path": self.stats.shell_path_builder(),
            }
        )
        self.script.register_global("IPerfProtocol", IPerfProtocol)
        self.script.register_global("IPerfMode", IPerfMode)
        self.script.register_global("IPerfTransferLimit", IPerfTransferLimit)
