from ..core.analysis import BenchmarkAnalysisTask

# class NetperfSanityCheck(BenchmarkAnalysis):
#     """
#     Sanity check for netperf stats
#     """
#     name = "netperf-sanity-check"
#     require = {DatasetName.NETPERF_DATA}

#     async def process_datasets(self):
#         self.logger.info("Verify integrity of netperf datasets")
#         dset = self.get_dataset(DatasetName.NETPERF_DATA)
#         # Check that all benchmarks report the same number of iterations
#         if "Confidence Iterations Run" in dset.agg_df.columns:
#             if len(dset.agg_df["Confidence Iterations Run"].unique()) > 1:
#                 self.logger.error("Benchmark iteration count does not match across samples")
#         else:
#             self.logger.warning(
#                 "Can not verify netperf iteration count, consider enabling the CONFIDENCE_ITERATION output")
#         # Check that all benchmarks ran a consistent amount of sampling
#         # functions in libstatcounters
#         dset = self.get_dataset(DatasetName.QEMU_STATS_CALL_HIT)
#         if dset:
#             syms_index = dset.ctx_agg_df.index.get_level_values("symbol")
#             cpu_start = syms_index == "cpu_start"
#             cpu_stop = syms_index == "cpu_stop"
#             statcounters_sample = syms_index == "statcounters_sample"
#             check = dset.ctx_agg_df.loc[cpu_start, "call_count"].groupby("dataset_id").sum().unique()
#             if len(check) > 1:
#                 self.logger.error("netperf::cpu_start anomalous #calls %s", check)
#             check = dset.ctx_agg_df.loc[cpu_stop, "call_count"].groupby("dataset_id").sum().unique()
#             if len(check) > 1:
#                 self.logger.error("netperf::cpu_stop anomalous #calls %s", check)
#             check = dset.ctx_agg_df.loc[statcounters_sample, "call_count"].groupby("dataset_id").sum().unique()
#             if len(check) > 1:
#                 self.logger.error("libstatcounters::statcounters_sample anomalous #calls %s", check)


class BenchmarkDataLoadTask(BenchmarkAnalysisTask):
    """
    General-purpose data loading and pre-processing task for benchmarks.

    This task will load some data from a target of a benchmark exec task.
    The load task needs to be pointed to the provider of the target, from which it
    can extract the path information.
    The data is loaded to a dataframe, according to a :class:`DataModel`.
    The input data model must be specified so that the input data is validated and
    the columns of interest are filtered.
    This task generates a DataFrameTarget() that identifies the task result.
    """
    #: The exec task from which to fetch the target
    exec_task: ExecutionTask = None
    #: The name of the target file to load
    target: str = None
    #: Input data model
    model: DataModel = None

    def run(self):
        pass


class NetperfStatsLoadTask(BenchmarDataLoadTask):
    exec_task = NetperfExecTask
    target = "stats"
    model = NetperfInputModel


class NetperfStatsPipeline():
    pass
