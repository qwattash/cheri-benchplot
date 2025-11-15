# Required for dataset registration hooks
import matplotlib

matplotlib.use("pdf")

# compile_db, iperf, kernel_build, netperf, nginx, unixbench
from . import (c18n, cloc, generic, kernel_advisories, pmc, qps, subobject)
