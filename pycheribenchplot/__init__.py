# Required for dataset registration hooks
import matplotlib

matplotlib.use("pdf")

from . import (c18n, cloc, compile_db, generic, iperf, kernel_advisories, kernel_build, netperf, nginx, pmc, qps,
               subobject, unixbench)
