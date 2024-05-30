# Required for dataset registration hooks
import matplotlib

matplotlib.use("pdf")
# import pycheribenchplot.analysis
# import pycheribenchplot.core.pidmap
# import pycheribenchplot.core.procstat
# import pycheribenchplot.drcachesim.analysis
# import pycheribenchplot.drcachesim.plot
from . import (c18n, cloc, generic, iperf, kernel_history, kernel_static, pmc, qps, subobject, unixbench, wrk)
from .kernel_vuln import analysis

# from .netperf import analysis, plot, task

# import pycheribenchplot.netstat.dataset
# import pycheribenchplot.pmc.analysis
# import pycheribenchplot.pmc.dataset
# import pycheribenchplot.pmc.plot
# import pycheribenchplot.qemu.cheribsd_counters
# import pycheribenchplot.qemu.cheribsd_counters_plot
# import pycheribenchplot.qemu.dataset
# import pycheribenchplot.qemu.plot
# import pycheribenchplot.spec.dataset
# import pycheribenchplot.vmstat.dataset
# import pycheribenchplot.vmstat.plot
