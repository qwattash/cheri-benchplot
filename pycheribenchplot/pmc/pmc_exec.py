from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ..core.artefact import PLDataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import Config, config_field
from ..core.task import ExecutionTask, output


class PMCSet(Enum):
    """
    Pre-defined sets of counters
    """
    Instr = "instr"
    L1Cache = "l1cache"
    L2Cache = "l2cache"
    Branch = "branch"


PMC_SET_COUNTERS = {
    PMCSet.Instr:
    ["CPU_CYCLES", "INST_RETIRED", "INST_SPEC", "EXECUTIVE_ENTRY", "EXECUTIVE_EXIT", "INST_SPEC_RESTRICTED"],
    PMCSet.L1Cache: [
        "CPU_CYCLES", "INST_RETIRED", "L1D_CACHE_REFILL", "L1D_CACHE", "L1D_CACHE_WB_VICTIM", "L1I_CACHE",
        "L1I_TLB_REFILL", "L1I_CACHE_REFILL", "L1D_TLB_REFILL"
    ],
    PMCSet.L2Cache:
    ["CPU_CYCLES", "INST_RETIRED", "L2D_CACHE_REFILL", "L2D_CACHE", "L2D_CACHE_WB_VICTIM", "BUS_ACCESS"],
    PMCSet.Branch: [
        "CPU_CYCLES", "INST_RETIRED", "BR_MIS_PRED", "BR_PRED", "BR_MIS_PRED_RS", "BR_RETIRED", "BR_MIS_PRED_RETIRED",
        "BR_RETURN_SPEC"
    ]
}


@dataclass
class PMCExecConfig(Config):
    """
    HWPMC configuration.
    """
    system_mode: bool = config_field(False, desc="Use system mode counters")
    sampling_mode: bool = config_field(False, desc="Use sampling vs counting mode")
    sampling_rate: int = config_field(97553, desc="Counter sampling rate, only relevant in sampling mode")
    counters: List[str] = config_field(list, desc="List of PMC counters to use")
    group: Optional[str] = config_field(None, desc="Pre-defined group of counters, overrides 'counters' option")

    @property
    def counters_list(self):
        if self.group:
            try:
                pmc_set = PMCSet(self.group)
            except ValueError:
                raise RuntimeError("Invalid configuration")
            return PMC_SET_COUNTERS[pmc_set]
        else:
            return self.counters


class PMCExec(ExecutionTask):
    """
    Configure hwpmc run for the current benchmark.

    If the benchmark does not support hwpmc, the script context will be ignored
    and the data ingest phase will fail.
    """
    task_namespace = "pmc"
    task_name = "exec"
    task_config_class = PMCExecConfig
    public = True

    @output
    def pmc_data(self):
        return RemoteBenchmarkIterationTarget(self, "hwpmc", ext="out")

    def run(self):
        self.script.extend_context({
            "hwpmc_config": self.config,
            "hwpmc_counters": self.config.counters_list,
            "hwpmc_output": self.pmc_data.remote_paths()
        })
