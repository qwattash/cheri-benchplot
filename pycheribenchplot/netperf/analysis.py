from ..core.dataset import DatasetID
from ..core.analysis import BenchmarkAnalysis


class NetperfSanityCheck(BenchmarkAnalysis):
    """
    Sanity check for netperf stats
    """
    @classmethod
    def check_required_datasets(cls, dsets):
        return DatasetID.NETPERF_DATA in set(dsets)

    def process_datasets(self):
        self.logger.info("Verify integrity of netperf datasets")
        dset = self.get_dataset(DatasetID.NETPERF_DATA)
        # Check that all benchmarks report the same number of iterations
        if "Confidence Iterations Run" in dset.agg_df.columns:
            if len(dset.agg_df["Confidence Iterations Run"].unique()) > 1:
                self.logger.error("Benchmark iteration count does not match across samples")
        else:
            self.logger.warning(
                "Can not verify netperf iteration count, consider enabling the CONFIDENCE_ITERATION output")
        # Check that all benchmarks ran a consistent amount of sampling
        # functions in libstatcounters
        dset = self.get_dataset(DatasetID.QEMU_STATS_CALL_HIST)
        if dset:
            syms_index = dset.agg_df.index.get_level_values("symbol")
            cpu_start = syms_index == "cpu_start"
            cpu_stop = syms_index == "cpu_stop"
            statcounters_sample = syms_index == "statcounters_sample"
            check = dset.agg_df.loc[cpu_start, "call_count"].unique()
            if len(check) > 1:
                self.logger.error("netperf::cpu_start anomalous #calls %s", check)
            check = dset.agg_df.loc[cpu_stop, "call_count"].unique()
            if len(check) > 1:
                self.logger.error("netperf::cpu_stop anomalous #calls %s", check)
            check = dset.agg_df.loc[statcounters_sample, "call_count"].unique()
            if len(check) > 1:
                self.logger.error("libstatcounters::statcounters_sample anomalous #calls %s", check)
