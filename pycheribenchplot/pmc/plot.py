from ..core.dataset import DatasetName, scale_to_std_notation
from ..core.plot import (BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, LegendInfo, Mosaic, Symbols)


class PMCMetricBar(BenchmarkSubPlot):
    """
    Draw a set of bars for each PMC metric
    """
    def __init__(self, plot, dataset, metric: str):
        super().__init__(plot)
        self.ds = dataset
        self.metric = metric
        self.col = (self.metric, "median", "sample")
        # We want the direct value of quartiles, not their delta here
        self.err_hi_col = (self.metric, "q75", "sample")
        self.err_lo_col = (self.metric, "q25", "sample")

    def get_legend_info(self):
        base = self.build_legend_by_dataset()
        legend = base.map_label(lambda l: f"{self.metric} " + l)
        return legend.assign_colors_hsv("dataset_id", h=(0.2, 1), s=(0.7, 0.7), v=(0.6, 0.9))

    def get_cell_title(self):
        return f"PMC {self.metric}"

    def get_df(self):
        """Generate base dataset with any auxiliary columns we need."""
        return self.ds.agg_df.copy()

    def generate(self, fm, cell):
        self.logger.debug("extract PMC metric %s", self.metric)

        # Note that we have a single bar group for each metric
        df = self.get_df()
        df["x"] = 0

        # Scale y column
        mag = scale_to_std_notation(df[self.col])
        df[self.err_hi_col] = df[self.err_hi_col] - df[self.col]
        df[self.err_lo_col] = df[self.col] - df[self.err_lo_col]
        if mag != 0:
            df.loc[:, self.col] /= mag
            df.loc[:, self.err_hi_col] /= mag
            df.loc[:, self.err_lo_col] /= mag
            mag_suffix = f"1e{mag}"
        else:
            mag_suffix = ""

        view = BarPlotDataView(df, x="x", yleft=self.col, err_hi=self.err_hi_col, err_lo=self.err_lo_col)
        view.bar_group = "dataset_id"
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        cell.x_config.ticks = [0]
        cell.x_config.tick_labels = [f"PMC {self.metric}"]
        cell.x_config.padding = 0.4
        cell.yleft_config.label = f"{self.metric} {mag_suffix}"


class RV64PMCOverviewPlot(BenchmarkPlot):
    """
    Overview of all PMC data we have for this benchmark
    """
    @classmethod
    def check_enabled(cls, datasets, config):
        required = {DatasetName.PMC}
        return required.issubset(datasets)

    def _make_subplots_mosaic(self):
        """
        Make mosaic with 2 columns: Left is the raw metric delta, right is the normalized % delta
        """
        subplots = {}
        layout = []
        pmc_stats = self.get_dataset(DatasetName.PMC)
        for idx, metric in enumerate(pmc_stats.data_columns()):
            name = f"subplot-pmc-stats-{idx}"
            if pmc_stats.agg_df[(metric, "median", "sample")].isna().all():
                continue
            subplots[name] = PMCMetricBar(self, pmc_stats, metric)
            layout.append([name])
        return Mosaic(layout, subplots)

    def get_plot_name(self):
        return "PMC data overview"

    def get_plot_file(self):
        return self.benchmark.get_plot_path() / "pmc-overview"


class PMCDeltaAll(BenchmarkSubPlot):
    """
    Combine all the PMC on the X axis with relative deltas on the Y axis
    """
    @classmethod
    def get_required_datasets(cls):
        dsets = super().get_required_datasets()
        dsets += [DatasetName.PMC]
        return dsets

    def __init__(self, plot):
        super().__init__(plot)
        self.ds = self.get_dataset(DatasetName.PMC)

    def get_legend_info(self):
        base = self.build_legend_by_dataset()
        return base.assign_colors_hsv("dataset_id", h=(0.2, 1), s=(0.7, 0.7), v=(0.6, 0.9))

    def get_cell_title(self):
        return "PMC relative overhead"

    def get_df(self):
        """Filter out baseline values as we are showing deltas"""
        sel = self.ds.agg_df.index.get_level_values("dataset_id") != self.benchmark.uuid
        df = self.ds.agg_df[sel].copy()
        # Make normalized delta columns a percentage
        df.loc[:, (slice(None), "median", "norm_delta_baseline")] *= 100
        return df

    def generate(self, fm, cell):
        df = self.get_df()
        df = df.dropna(axis=1)

        median = df.columns.get_level_values("aggregate") == "median"
        norm_delta = df.columns.get_level_values("delta") == "norm_delta_baseline"
        df = df[df.columns[(median & norm_delta)]]
        # Drop extra column levels as they are no longer really needed
        df = df.droplevel(["aggregate", "delta"], axis=1)
        # Unpivot the PMC columns into rows for the bar plot
        df = df.melt(ignore_index=False)
        # Assign seed X coordinates
        df["x"] = df.groupby("dataset_id").cumcount()
        x = df["x"].unique()
        # Keep index unique after melting
        df.set_index("x", append=True, inplace=True)

        view = BarPlotDataView(df, x="x", yleft=["value"])
        view.bar_group = "dataset_id"
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        cell.x_config.ticks = x
        cell.x_config.tick_labels = df["metric"].unique()
        cell.x_config.tick_rotation = 90
        cell.yleft_config.label = f"% {Symbols.DELTA}metric"


class RV64PMCDeltaOverviewPlot(BenchmarkPlot):
    """
    Overview of all PMC data we have for this benchmark
    """
    subplots = [PMCDeltaAll]

    def get_plot_name(self):
        return "PMC delta overview"

    def get_plot_file(self):
        return self.benchmark.get_plot_path() / "pmc-delta-overview"
