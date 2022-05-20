from ..core.dataset import DatasetName, scale_to_std_notation
from ..core.plot import (BarPlotDataView, BenchmarkPlot, BenchmarkSubPlot, LegendInfo, LinePlotDataView, Mosaic,
                         Symbols)


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
        # bar plots want relative errors
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


class PMCDeltaBar(BenchmarkSubPlot):
    """
    Draw a set of bar for each PMC delta metric
    """
    def __init__(self, plot, dataset, metric: str):
        super().__init__(plot)
        self.ds = dataset
        self.metric = metric
        self.col = (self.metric, "median", "norm_delta_baseline")
        self.err_hi_col = (self.metric, "q75", "norm_delta_baseline")
        self.err_lo_col = (self.metric, "q25", "norm_delta_baseline")

    def get_legend_info(self):
        base = self.build_legend_by_dataset()
        legend = base.map_label(lambda l: f"{Symbols.DELTA}{self.metric} " + l)
        return legend.assign_colors_hsv("dataset_id", h=(0.2, 1), s=(0.7, 0.7), v=(0.6, 0.9))

    def get_cell_title(self):
        return f"PMC {Symbols.DELTA}{self.metric} w.r.t baseline"

    def get_df(self):
        df = self.ds.agg_df
        sel = df.index.get_level_values("dataset_id") != self.benchmark.uuid
        return df[sel].copy()

    def generate(self, fm, cell):
        self.logger.debug("extract PMC Delta metric %s", self.metric)
        # We have a single bar group for each metric, so just one X coordinate
        df = self.get_df()
        df["x"] = 0

        # bar plots want relative errors and we convert things to a percentage
        df[self.err_hi_col] = (df[self.err_hi_col] - df[self.col]) * 100
        df[self.err_lo_col] = (df[self.col] - df[self.err_lo_col]) * 100
        df[self.col] = df[self.col] * 100

        # import code
        # code.interact(local=locals())
        view = BarPlotDataView(df, x="x", yleft=self.col, err_hi=self.err_hi_col, err_lo=self.err_lo_col)
        view.bar_group = "dataset_id"
        view.legend_info = self.get_legend_info()
        view.legend_level = ["dataset_id"]
        cell.add_view(view)

        cell.x_config.ticks = [0]
        cell.x_config.tick_labels = [f"% {Symbols.DELTA}{self.metric}"]
        cell.x_config.padding = 0.4
        cell.yleft_config.label = f"% {Symbols.DELTA}{self.metric}"


class PMCOverviewPlot(BenchmarkPlot):
    """
    Overview of all PMC data we have for this benchmark
    """
    require = {DatasetName.PMC}
    name = "pmc-overview"
    description = "PMC data overview"

    def _make_subplots_mosaic(self):
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


class PMCDeltaAll(BenchmarkSubPlot):
    """
    Combine all the PMC on the X axis with relative deltas on the Y axis
    """
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


class PMCDeltaOverviewPlot(BenchmarkPlot):
    """
    Overview of all PMC data we have for this benchmark
    """
    require = {DatasetName.PMC}
    name = "pmc-delta-overview"
    description = "PMC delta overview"

    def _make_subplots_mosaic(self):
        """
        Make mosaic with 2 columns: Left is the raw metric delta, right is the normalized % delta
        """
        subplots = {}
        layout = []
        subplots["pmc-delta-all"] = PMCDeltaAll(self)
        layout.append(["pmc-delta-all"])

        pmc_stats = self.get_dataset(DatasetName.PMC)
        for idx, metric in enumerate(pmc_stats.data_columns()):
            name = f"subplot-pmc-stats-{idx}"
            if pmc_stats.agg_df[(metric, "median", "sample")].isna().all():
                continue
            subplots[name] = PMCDeltaBar(self, pmc_stats, metric)
            layout.append([name])

        return Mosaic(layout, subplots)


class PMCParamScalingView(BenchmarkSubPlot):
    """
    Generate line plot to show PMC variation along a parameterisation axis.
    """
    def __init__(self, plot, pmc_metric, param):
        super().__init__(plot)
        self.pmc = pmc_metric
        self.param = param
        self.col = (self.pmc, "median", "sample")
        self.err_hi = (self.pmc, "q75", "sample")
        self.err_lo = (self.pmc, "q25", "sample")

    def get_cell_title(self):
        return f"PMC {self.pmc} trend w.r.t. {self.param}"

    def get_df(self):
        ds = self.get_dataset(DatasetName.PMC)
        return ds.cross_merged_df.copy()

    def generate(self, fm, cell):
        df = self.get_df()
        self.logger.debug("Extract XPMC %s along %s", self.pmc, self.param)

        df[self.err_hi] = df[self.err_hi] - df[self.col]
        df[self.err_lo] = df[self.col] - df[self.err_lo]

        view = LinePlotDataView(df, x=self.param, yleft=self.col, err_hi=self.err_hi, err_lo=self.err_lo)
        view.line_group = ["dataset_gid"]
        view.legend_info = self.build_legend_by_gid()
        view.legend_level = ["dataset_gid"]
        cell.add_view(view)

        cell.x_config.label = self.param
        cell.x_config.ticks = sorted(df.index.unique(self.param))
        cell.yleft_config.label = self.pmc


class PMCParamScaling(BenchmarkPlot):
    """
    Generate plot to show PMC variation for each benchmark parameterisation.
    """
    require = {DatasetName.PMC}
    name = "pmc-param-scaling"
    description = "PMC parameterisation scaling"
    cross_analysis = True

    def _make_subplots_mosaic(self):
        """
        We create a matrix of plots. Rows are PMC, columns are benchmark
        parameters.
        """
        ds = self.get_dataset(DatasetName.PMC)
        layout = []
        subplots = {}

        for p in ds.parameter_index_columns():
            for j, pmc in enumerate(ds.data_columns()):
                if ds.agg_df[(pmc, "median", "sample")].isna().all():
                    # Skip missing metrics
                    continue
                name = f"pmc-param-{p}-{pmc}"
                subplots[name] = PMCParamScalingView(self, pmc, p)
                if j < len(layout):
                    layout[j].append(name)
                else:
                    layout.append([name])
        return Mosaic(layout, subplots)
