# Required for dataset registration hooks
import matplotlib

matplotlib.use("pdf")
# import pycheribenchplot.core.pidmap
# import pycheribenchplot.core.procstat
# import pycheribenchplot.drcachesim.analysis
# import pycheribenchplot.drcachesim.plot
from . import (c18n, cloc, generic, iperf, kernel_advisories, kernel_history, kernel_static, netperf, pmc, qps,
               subobject, unixbench, wrk)

# import pycheribenchplot.netstat.dataset
# import pycheribenchplot.qemu.cheribsd_counters
# import pycheribenchplot.qemu.cheribsd_counters_plot
# import pycheribenchplot.qemu.dataset
# import pycheribenchplot.qemu.plot
# import pycheribenchplot.spec.dataset
# import pycheribenchplot.vmstat.dataset
# import pycheribenchplot.vmstat.plot
