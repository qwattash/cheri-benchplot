from pycheribenchplot.core.analysis import DatasetTransform
from pycheribenchplot.core.dataset import DataSetParser, subset_xs


class QEMUCombineHistStats(DatasetTransform):
    @classmethod
    def get_required_datasets(cls):
        return [DataSetParser.QEMU_STATS_BB_HIST, DataSetParser.QEMU_STATS_CALL_HIST]

    def compute(self):
        bb_df = self.benchmark.get_dataset(DataSetParser.QEMU_STATS_BB_HIST).agg_df
        call_df = self.benchmark.get_dataset(DataSetParser.QEMU_STATS_CALL_HIST).agg_df
        self.computed_df = bb_df.join(call_df, how="inner", lsuffix="_bb", rsuffix="_call")


class QEMUHistStatsFilterIdxCommonSymbols(DatasetTransform):
    @classmethod
    def get_required_datasets(cls):
        return [DataSetParser.QEMU_STATS_BB_HIST, DataSetParser.QEMU_STATS_CALL_HIST]

    def compute(self):
        """
        Return a dataframe containing the cross section of the (joined) qemu stats dataframes
        containing the symbols that are common to all runs (i.e. across all __dataset_id values).
        We consider valid common symbols those for which we were able to resolve the (file, sym_name)
        pair and have sensible BB count and call_count values.
        Care must be taken to keep the multi-index levels aligned.
        """
        super().compute()
        df = self.benchmark.find_dset_transform(QEMUCombineHistStats).computed_df
        # Isolate the file:symbol pairs for each symbol marked valid in all datasets.
        # Since the filter is the same for all datasets, the cross-section will stay aligned.
        valid = (df["valid_symbol"] == "ok") & (df["bb_count"] != 0)
        valid_syms = valid.groupby(["file", "symbol"]).all()
        self.computed_df = subset_xs(df, valid_syms)


class QEMUHistStatsFilterIdxExtraSymbols(DatasetTransform):
    @classmethod
    def get_required_datasets(cls):
        return [DataSetParser.QEMU_STATS_BB_HIST, DataSetParser.QEMU_STATS_CALL_HIST]

    def compute(self):
        """
        This is complementary to QEMUHistStatsFilterIdxCommonSymbols
        """
        super().compute()
        df = self.benchmark.find_dset_transform(QEMUCombineHistStats).computed_df
        # Isolate the file:symbol pairs for each symbol marked valid in at least one dataset,
        # but not all datasets.
        valid = df["valid_symbol"] == "ok"
        # bb_count is valid if:
        # symbol is valid and bb_count != 0
        # symbol is invalid and bb_count == 0
        bb_count_ok = (df["bb_count"] == 0) ^ valid
        # Here we only select symbols that have no issues in the bb_count column
        all_bb_count_ok = bb_count_ok.groupby(["file", "symbol"]).all()
        all_valid = valid.groupby(["file", "symbol"]).all()
        some_valid = valid.groupby(["file", "symbol"]).any()
        unique_syms = all_bb_count_ok & some_valid & ~all_valid
        self.computed_df = subset_xs(df, unique_syms)
