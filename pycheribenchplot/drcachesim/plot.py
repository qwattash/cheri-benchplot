from pycheribenchplot.core.config import DatasetName
from pycheribenchplot.core.plot.plot_base import BenchmarkPlot


class CacheSizesPlot(BenchmarkPlot):
    require = {DatasetName.QEMU_DYNAMORIO}
    name = "cache-sizes"
    description = "Plot cache sizes"