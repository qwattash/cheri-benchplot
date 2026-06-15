from ..generic.timing import TimingSlicePlotTask
from .unixbench_exec import UnixBenchExec


class UnixBenchTimingPlot(TimingSlicePlotTask):
    exec_task_class = UnixBenchExec
    task_namespace = "unixbench"
    task_name = "timing"
    public = True
