from ..generic.timing import TimingPlotTask
from .build_exec import KernelBuildBenchmarkExec


class KernelBuildTimingPlot(TimingPlotTask):
    exec_task_class = KernelBuildBenchmarkExec
    task_namespace = "kernel-build"
    task_name = "timing"
    public = True
