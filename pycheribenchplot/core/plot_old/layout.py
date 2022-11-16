from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..analysis import BenchmarkAnalysis
from ..benchmark import Benchmark


class AnalysisSubplot(ABC):
    """
    Handle rendering of a subplot on a set of axes.
    This is used to more easily compose plots.
    """
    def __init__(self, plot: "AnalysisPlot"):
        self.plot = plot
        self.logger = plot.logger
        self.benchmark = plot.benchmark

    def get_dataset(self, name: DatasetName):
        """
        Helper to access a dataset in the benchmark
        """
        dset = self.plot.get_dataset(name)
        assert dset is not None, f"Subplot missing dependency {name}"
        return dset

    @abstractmethod
    def draw(self, figure, ax):
        """
        Draw the subplot in the given axes.
        """
        ...


class PlotLayout:
    """
    Base class for plot layout helpers.
    This is responsible for collecting subplots generators and
    produce a figure and a set of axes for each of the subplots.
    """
    def __init__(self):
        self.mosaic = np.array([[]])
        self.subplots = {}

    def figsize(self):
        """
        A mosaic is always at least 2-D. It must be a square matrix.
        """
        m = np.atleast_2d(np.array(self.mosaic, dtype=object))
        nrows, ncols = m.shape
        return (10 * ncols, min(7 * nrows, 300))

    def mosaic_array(self):
        """
        A mosaic is always at least 2-D. It must be a square matrix.
        """
        m = np.atleast_2d(np.array(self.mosaic, dtype=object))
        return m

    def extract(self, name: str):
        """
        Extract the subset of the mosaic matrix with the given subplot name.
        Set everything else to BLANK
        """
        r, c = np.where(self.mosaic == name)
        cut = self.mosaic[min(r):max(r) + 1, min(c):max(c) + 1]
        blank = np.where(cut != name)
        cut[blank] = "BLANK"
        return cut

    def add_subplot(self, subplot: AnalysisSubplot):
        pass

    def __iter__(self):
        for key, sub in self.subplots.items():
            yield (key, sub)


class SimplePlotLayout(PlotLayout):
    pass


class AnalysisPlot(BenchmarkAnalysis):
    """
    Helper class to produce combined plots as analysis passes.
    This only takes care of automatic layout generation given a set
    of subplot classes.
    """
    def _write_figure(self, fig, suffix=None):
        """
        Save the given figure to the file formats specified by the configuration.
        If a suffix is given, it is appended to the plot file name
        """
        plot_output_base = self.get_plot_file()
        if suffix:
            plot_output_base.with_stem(plot_output_base.stem + f"-{suffix}")
        for ext in self.analysis_config.plot_output_format:
            plot_path = plot_output_base.with_suffix(f".{ext}")
            self.logger.debug("Writing plot %s -> %s", self.name, plot_path)
            fig.savefig(plot_path)
        plt.close(fig)

    def get_plot_title(self):
        """
        Main title for the plot.
        """
        if self.description is not None:
            return self.description
        return self.name

    def get_plot_file(self):
        """
        Generate output file for the plot.
        Note that the filename should not have an extension as the surface will
        append it later.
        """
        if self.cross_analysis:
            return self.benchmark.session.get_plot_root_path() / self.name
        else:
            return self.benchmark.get_plot_path() / self.name

    def setup_layout(self) -> PlotLayout:
        """
        Return the plot layout instance to use for figure rendering
        """
        return SimplePlotLayout()

    def setup_figure(self, fig, axes: dict):
        """
        Perform any extra figure initialization.
        """
        fig.suptitle(self.get_plot_title())

    def process_datasets(self):
        """
        First pass setup the layout, second pass invoke subplots drawing.
        """
        self.logger.info("Setup plot %s", self.name)
        layout = self.setup_layout()

        if self.config.split_subplots:
            for name, subplot in layout:
                fig = plt.figure(constrained_layout=True)
                axes = fig.subplot_mosaic(layout.extract(name), empty_sentinel="BLANK")
                subplot.draw(fig, axes[name])
                self._write_figure(fig, suffix=name)
        else:
            fig = plt.figure(constrained_layout=True, figsize=layout.figsize())
            axes = fig.subplot_mosaic(layout.mosaic_array(), empty_sentinel="BLANK")
            for name, subplot in layout:
                subplot.draw(fig, axes[name])
            self._write_figure(fig)
