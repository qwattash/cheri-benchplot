from ..generic.timing import TimingSlicePlotTask
from .build_exec import KernelBuildBenchmarkExec


class KernelBuildTimingPlot(TimingSlicePlotTask):
    exec_task_class = KernelBuildBenchmarkExec
    task_namespace = "kernel-build"
    task_name = "timing"
    public = True
