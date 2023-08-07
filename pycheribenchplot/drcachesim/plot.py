import os
import typing
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pycheribenchplot.core.config import Config, DatasetName
from pycheribenchplot.core.plot.plot_base import BenchmarkPlot


@dataclass
class CachePlotConfig(Config):
    plot_cache_levels: typing.List[str] = field(default_factory=list)


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

    def plot_result_internal(self, cache_level, outdir, run_name):
        if cache_level == 'LL':
            file_path = outdir / "LL_size"
            key_str = "Local miss rate:"
            start_str = "LL stats:"
        elif cache_level == 'L1D':
            file_path = outdir / "L1D_size"
            key_str = "Miss rate:"
            # for now assume there is only one core
            start_str = "L1D stats:"
        elif cache_level == 'L1I':
            file_path = outdir / "L1I_size"
            key_str = "Miss rate:"
            start_str = "L1I stats:"
        if not file_path.exists():
            raise Exception(f"No cache size file found for {file_path}")
        files = os.listdir(file_path)
        cache_sizes = np.array([])
        miss_rates = np.array([])
        for file in files:
            if file.endswith('.txt'):
                cache_size = file.split('.')[0]
                with open(file_path / file, 'r') as f:
                    buf = f.read()
                    start_ind = buf.find(start_str)
                    if start_ind == -1:
                        continue
                    else:
                        start_ind += len(start_str)
                    ind = buf.find(key_str, start_ind)
                    if ind == -1:
                        continue
                    else:
                        ind += len(key_str)
                        miss_rate = float(buf[ind:].split()[0].strip('%')) / 100
                        cache_sizes = np.append(cache_sizes, cache_size)
                        miss_rates = np.append(miss_rates, miss_rate)

        cache_sizes_bytes = np.array([int(x[:-1]) * self.convert_prefix(x[-1]) for x in cache_sizes])
        ind = np.argsort(cache_sizes_bytes)
        cache_sizes_bytes = cache_sizes_bytes[ind]
        miss_rates = miss_rates[ind]
        cache_sizes = cache_sizes[ind]

        plt.plot(cache_sizes_bytes, miss_rates, 'o-', label=run_name)
        plt.xticks(cache_sizes_bytes, cache_sizes)

    def plot_result(self, cache_level, outdirs, variant_name):
        plt.xscale('log')
        if cache_level == 'LL':
            plt.xlabel('LL cache size')
        elif cache_level == 'L1D':
            plt.xlabel('L1D cache size')
        elif cache_level == 'L1I':
            plt.xlabel('L1I cache size')
        for outdir, spec_variant in outdirs.items():
            outdir = Path(outdir)
            try:
                self.plot_result_internal(cache_level, outdir, spec_variant)
            except Exception as e:
                self.logger.warning(f"Failed to plot cache sizes for {variant_name}: {e}")
                return
        plt.ylabel('Local miss rate')
        # plt.ylim(0, 1)
        plt.title(f'Local miss rate vs {cache_level} cache size: {variant_name}')
        plt.legend(loc='upper right')
        file_path = self.get_plot_root_path() / "cache_plots" / (cache_level + "_cache_sizes") / (variant_name + '.png')
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path, format='png')
        plt.close()

    async def process_datasets(self):
        dset = self.get_dataset(DatasetName.QEMU_DYNAMORIO)
        trace_info = dset.merged_tracefiles
        groups = trace_info.groupby('variant').groups
        for variant, group in groups.items():
            for level in self.config.plot_cache_levels:
                self.logger.info(f"Plotting cache sizes for variant {variant}")
                self.plot_result(level, {
                    k: v
                    for k, v in zip(trace_info.iloc[group]['cachesim_dir'], trace_info.iloc[group]['spec_variant'])
                }, variant)
