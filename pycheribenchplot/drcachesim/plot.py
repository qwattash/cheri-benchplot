from dataclasses import dataclass
from pathlib import Path
from pycheribenchplot.core.config import DatasetName, TemplateConfig
from pycheribenchplot.core.plot.plot_base import BenchmarkPlot
import os, numpy as np
import matplotlib.pyplot as plt

@dataclass
class CachePlotConfig(TemplateConfig):
    cache_level: str = "LL" 
class CacheSizesPlot(BenchmarkPlot):
    require = {DatasetName.QEMU_DYNAMORIO}
    name = "cache-plot"
    description = "Plot miss rate against cache sizes"
    analysis_options_class = CachePlotConfig
    cross_analysis: bool = True

    def convert_prefix(self, prefix):
        if prefix == 'K':
            return 1024
        elif prefix == 'M':
            return 1024**2
        elif prefix == 'G':
            return 1024**3
        else:
            return 1

    def plot_result_internal(self, outdir, run_name):
        files = os.listdir(outdir)
        cache_sizes = np.array([])
        miss_rates = np.array([])
        for file in files:
            if file.startswith('LL_size_') and file.endswith('.txt'):
                cache_size = file[8:-4]
                with open(os.path.join(outdir, file), 'r') as f:
                    buf = f.read()
                    ind = buf.find('Local miss rate:')
                    if ind == -1:
                        continue
                    else:
                        ind += len('Local miss rate:')
                        miss_rate = float(buf[ind:].split()[0].strip('%'))/100
                        cache_sizes = np.append(cache_sizes, cache_size)
                        miss_rates = np.append(miss_rates, miss_rate)

        cache_sizes_bytes = np.array([int(x[:-1])*self.convert_prefix(x[-1]) for x in cache_sizes])
        ind = np.argsort(cache_sizes_bytes)
        cache_sizes_bytes = cache_sizes_bytes[ind]
        miss_rates = miss_rates[ind]
        cache_sizes = cache_sizes[ind]

        plt.plot(cache_sizes_bytes, miss_rates, 'o-', label=run_name)
        plt.xticks(cache_sizes_bytes, cache_sizes)

    def plot_result(self, outdirs, variant_name):
        plt.xscale('log')
        for outdir, spec_variant in outdirs.items():
            self.plot_result_internal(outdir, spec_variant)
        plt.xlabel('LL cache size')
        plt.ylabel('Local miss rate')
        # plt.ylim(0, 1)
        plt.title('Local miss rate vs LL cache size: ' + variant_name)
        plt.legend(loc='upper right')
        file_path = self.get_plot_root_path() / "cache_plots" / "LL_cache_sizes" / (variant_name + '.png') 
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path, format='png')
        plt.close()

    async def process_datasets(self):
        dset = self.get_dataset(DatasetName.QEMU_DYNAMORIO)
        trace_info = dset.merged_tracefiles
        groups = trace_info.groupby('variant').groups
        for variant, group in groups.items():
            self.logger.info(f"Plotting cache sizes for variant {variant}")
            self.plot_result({k:v for k,v in zip(trace_info.iloc[group]['cachesim_dir'], trace_info.iloc[group]['spec_variant'])}, variant)
        