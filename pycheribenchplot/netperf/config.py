from dataclasses import dataclass, field
from pathlib import Path

from ..core.options import TemplateConfig


@dataclass
class NetperfBenchmarkRunConfig(TemplateConfig):
    netperf_path: Path = Path("opt/{cheri_target}/netperf/bin")
    netperf_prime_options: list[str] = field(default_factory=list)
    netperf_options: list[str] = field(default_factory=list)
    netserver_options: list[str] = field(default_factory=list)
    qemu_log_output: str = "netperf-qemu-{uuid}.csv"
