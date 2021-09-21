from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import TemplateConfig, path_field


@dataclass
class NetperfBenchmarkRunConfig(TemplateConfig):
    netperf_path: Path = path_field("opt/{cheri_target}/netperf/bin")
    netperf_ktrace_options: list[str] = field(default_factory=list)
    netperf_prime_options: list[str] = field(default_factory=list)
    netperf_options: list[str] = field(default_factory=list)
    netserver_options: list[str] = field(default_factory=list)
    qemu_log_output: str = "netperf-qemu-{uuid}.pb"
