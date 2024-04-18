import json
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from ..core.analysis import AnalysisTask
from ..core.artefact import (DataFrameTarget, HTMLTemplateTarget, Target, ValueTarget, make_dataframe_loader)
from ..core.config import Config, ConfigPath, InstanceKernelABI
from ..core.dwarf import DWARFManager, StructLayoutGraph
from ..core.plot import PlotTarget, PlotTask, new_facet, new_figure
from ..core.task import DataGenTask, DatasetTask, dependency, output
# from ..ext import pychericap, pydwarf
from .model import (ImpreciseSubobjectInfoModel, ImpreciseSubobjectInfoModelRecord, ImpreciseSubobjectLayoutModel)


def legend_on_top(fig, ax=None, **kwargs):
    """
    Fixup the legend to appear at the top of the axes
    """
    if not fig.legends:
        return
    # Hack to remove the legend as we can not easily move it
    # Not sure why seaborn puts the legend in the figure here
    legend = fig.legends.pop()
    if ax is None:
        owner = fig
    else:
        owner = ax
    kwargs.setdefault("loc", "center")
    owner.legend(legend.legend_handles,
                 map(lambda t: t.get_text(), legend.texts),
                 bbox_to_anchor=(0, 1.02),
                 ncols=4,
                 **kwargs)


@dataclass
class PathMatchSpec(Config):
    path: ConfigPath
    match: str | None = None


@dataclass
class ExtractImpreciseSubobjectConfig(Config):
    """
    Configure the imprecise sub-object extractor.
    """
    #: List of paths that are used to extract DWARF information.
    #: Note that multiple paths will be considered as belonging to the same "object",
    #: if multiple objects are being compared, we should use parameterization.
    #: Note that relative paths will be interpreted as relative paths into a cheribuild rootfs.
    dwarf_data_sources: list[PathMatchSpec]

    # def _check_flex_array(self, g: nx.DiGraph, n: StructLayoutGraph.NodeID) -> bool:
    #     """
    #     Check whether the given structure contains a flexible array member at the end.

    #     Note that this uses an heuristic to skip structures with zero-length padding
    #     arrays at the end that are generated for CheriBSD syscall arguments.
    #     """
    #     # Special case for kernel system call structures that may have 0 padding members.
    #     # We know that there are no flex arrays in there
    #     if n.file.endswith("sys/sys/sysproto.h") or re.search(r"sys/compat/.*_proto\.h$", n.file):
    #         self.logger.warning("Note: %s @ %s:%d ignores flex array matching due to CheriBSD heuristic",
    #                             g.nodes[n]["type_name"], n.file, n.line)
    #         return False
    #     top_level_members = sorted(g.successors(n), key=lambda m: g.nodes[m]["offset"])
    #     if not top_level_members:
    #         return False

    #     last = top_level_members[-1]
    #     if g.nodes[last]["flags"] & pydwarf.TypeInfoFlags.kIsArray == 0 or g.nodes[last]["nitems"] != 0:
    #         return False

    #     # This is likely a flex array.
    #     self.logger.debug("Found flex array structure %s", n)
    #     return True


class RequiredSubobjectPrecision(ImpreciseMembersPlotBase):
    """
    Plot the number of bits required to ensure that imprecise sub-objects would be
    representable.
    """
    public = True
    task_name = "imprecise-subobject-bits-plot"

    @output
    def imprecise_bits_plot(self):
        return PlotTarget(self, "all")

    @output
    def imprecise_bits_cdf(self):
        return PlotTarget(self, "cdf")

    @output
    def imprecise_abs_bits_cdf(self):
        return PlotTarget(self, "cdf-abs")

    @output
    def imprecise_common_bits_plot(self):
        return PlotTarget(self, "common")

    @output
    def imprecise_common_bits_cdf(self):
        return PlotTarget(self, "common-cdf")

    @output
    def imprecise_common_abs_bits_cdf(self):
        return PlotTarget(self, "common-cdf-abs")

    def _compute_precision(self, base: float, top: float):
        """
        This should produce the mantissa size required for a specific (base, top) pair
        to be representable.
        """
        assert base <= top, "Invalid base > top"
        base = int(base)
        top = int(np.ceil(top))

        def lsb(x):
            if x == 0:
                return 0
            return int(np.log2(x & -x))

        len_msb = np.floor(np.log2(top - base)) if top != base else 0
        if top == 0:
            # Base must also be 0
            exponent = 0
        elif base == 0:
            exponent = lsb(top)
        else:
            exponent = min(lsb(base), lsb(top))
        return len_msb - exponent + 1

    def _platform_cap_format(self, g_uuid: uuid.UUID):
        instance_config = self.get_instance_config(g_uuid)
        if instance_config.cheri_target.is_riscv():
            # XXX support/detect riscv32
            cap_format = pychericap.CompressedCap128
        elif instance_config.cheri_target.is_morello():
            cap_format = pychericap.CompressedCap128m
        else:
            self.logger.error("DWARF TypeInfo extraction unsupported for %s", instance_config.cheri_target)
            raise RuntimeError(f"Unsupported instance target {instance_config.cheri_target}")
        return cap_format

    def _platform_mantissa_and_exp_width(self, g_uuid: uuid.UUID) -> tuple[int, int]:
        """
        Return the mantissa and exponent width of a given platform
        """
        cap_format = self._platform_cap_format(g_uuid)
        mantissa_width = cap_format.get_mantissa_width()
        exponent_width = 3
        return mantissa_width, exponent_width

    def _compute_platform_precision(self, g_uuid: uuid.UUID, base: float, top: float):
        """
        Compute the precision for a given [base, top] region on the platform identified
        by the g_uuid.

        Note that base and top may be floats if the region involves bit fields.
        """
        base = int(base)
        top = int(np.ceil(top))
        mantissa_width, exponent_width = self._platform_mantissa_and_exp_width(g_uuid)

        ie_threshold = mantissa_width - 2
        len_msb = np.floor(np.log2(top - base)) if top != base else 0
        if len_msb + 1 <= ie_threshold:
            precision = ie_threshold
        else:
            precision = mantissa_width - (1 + exponent_width)
        return precision

    def _plot_precision_bars(self, df, target):
        """
        Produce a bar plot showing the extra number of precision bits required
        for each imprecise sub-object member.
        """
        show_df = df.reset_index()
        show_df["legend"] = show_df["dataset_gid"].map(self.g_uuid_to_label)
        show_df["label"] = show_df["file"] + ": " + show_df["base_name"] + "::" + show_df["member_name"]
        show_df = show_df.sort_values(["member_size", "label", "dataset_gid"])

        with new_figure(target.paths(), figsize=(10, 50)) as fig:
            ax = fig.subplots()

            (so.Plot(show_df, y="label", x="additional_precision",
                     color="legend").on(ax).add(so.Bar(),
                                                so.Dodge(by=["dataset_gid"]),
                                                dataset_gid=show_df["dataset_gid"],
                                                orient="y").plot())

            legend_on_top(fig, ax)
            ax.set_xlabel("Increased precision (bits) required")
            ax.set_ylabel("Sub-object field")
            plt.setp(ax.get_yticklabels(), ha="right", va="center", fontsize="xx-small")

    def _plot_precision_cdf(self, df, target: PlotTarget, absolute: bool):
        """
        Produce a CDF plot showing the amount of imprecise sub-object members that
        can be "fixed" by adding a number of precision bits.
        """
        show_df = df.reset_index()
        show_df["legend"] = show_df["dataset_gid"].map(self.g_uuid_to_label)
        show_df["label"] = show_df["file"] + ": " + show_df["base_name"] + "::" + show_df["member_name"]
        show_df = show_df.sort_values(["member_size", "label", "dataset_gid"])

        with new_figure(target.paths()) as fig:
            ax = fig.subplots()

            if absolute:
                sns.ecdfplot(data=show_df, x="required_precision", hue="legend", ax=ax)
                ax.set_xlabel("Precision bits")
                ax.set_ylabel("Proportion of sub-objects that become representable")
            else:
                sns.ecdfplot(data=show_df, x="additional_precision", hue="legend", ax=ax)
                ax.set_xlabel("Additional precision bits")
                ax.set_ylabel("Proportion of sub-objects that become representable")

    def _plot_precision(self, df, bars_target, cdf_target, abs_cdf_target):
        """
        Generate the following plots for the given dataset:
        - A bar plot that shows the number of precision bits required to make an imprecise sub-object
          representable.
        - A CDF plot showing the proportion of imprecise sub-objects that become representable as
        the precision increases.
        - A CDF plot as above, but the X axis uses the absolute number of precision bits instead of the
        offset from the platform-specific precision.
        """
        sns.set_theme()
        # Compute imprecision bits
        # XXX just validate the model here?
        member_offset_index = df.index.names.index("member_offset")
        assert member_offset_index >= 0, "Missing member_offset from index"
        gid_index = df.index.names.index("dataset_gid")
        assert gid_index >= 0, "Missing dataset_gid from index"

        def _calc_precision(r):
            return self._compute_precision(r.name[member_offset_index], r.name[member_offset_index] + r["member_size"])

        df["required_precision"] = df.apply(_calc_precision, axis=1)

        def _calc_plat_precision(r):
            return self._compute_platform_precision(r.name[gid_index], r.name[member_offset_index],
                                                    r.name[member_offset_index] + r["member_size"])

        df["platform_precision"] = df.apply(_calc_plat_precision, axis=1)

        df["additional_precision"] = df["required_precision"] - df["platform_precision"]
        assert (df["additional_precision"] >= 0).all(), \
            "Something is wrong, these must be unrepresentable"

        self._plot_precision_bars(df, bars_target)
        self._plot_precision_cdf(df, cdf_target, absolute=False)
        self._plot_precision_cdf(df, abs_cdf_target, absolute=True)

    def run_plot(self):
        self._plot_precision(self._prepare_dataset(), self.imprecise_bits_plot, self.imprecise_bits_cdf,
                             self.imprecise_abs_bits_cdf)
        self._plot_precision(self._prepare_dataset_filter_common(), self.imprecise_common_bits_plot,
                             self.imprecise_common_bits_cdf, self.imprecise_common_abs_bits_cdf)


class ImpreciseSetboundsSecurity(PlotTask):
    """
    Produce plots to display the impact of imprecise sub-object
    """
    public = True
    task_namespace = "subobject"
    task_name = "imprecise-subobject-security"

    @dependency
    def data(self):
        return ImpreciseSubobjectBoundsUnion(self.session, self.analysis_config)

    @output
    def categories(self):
        return PlotTarget(self, "categories")

    @output
    def categories_percent(self):
        return PlotTarget(self, "categories-percent")

    @output
    def categories_alias_ptr(self):
        return PlotTarget(self, "alias-ptr")

    @output
    def categories_sizes_per_alias_kind(self):
        return PlotTarget(self, "sizes-per-alias-kind")

    def run_plot(self):
        containers = self.data.all_layouts.get()
        records = []

        for c in containers:
            gid = uuid.UUID(c.layouts.graph["dataset_gid"])
            # Count categories for this data source
            for node, attrs in c.layouts.nodes(data=True):
                if attrs.get("alias_group_id") is None:
                    # Nothing else to do
                    continue

                if attrs["flags"] & pydwarf.TypeInfoFlags.kIsStruct:
                    category = "struct"
                elif attrs["flags"] & pydwarf.TypeInfoFlags.kIsUnion:
                    category = "union"
                elif attrs["flags"] & pydwarf.TypeInfoFlags.kIsArray:
                    category = "array"
                else:
                    self.logger.warning("Uncategorised sub-object node %s %s", node.member, attrs)
                    continue

                r = dict(dataset_gid=gid,
                         category=category,
                         alias_kind="alias-ptr" if attrs["alias_pointer_members"] else "other",
                         size=node.member_size)
                records.append(r)
        df = pd.DataFrame(records)
        df["platform"] = df["dataset_gid"].apply(self.g_uuid_to_label)
        sns.set_theme()

        # Plot the absolute number of imprecise members in each category grouped by platform
        with new_figure(self.categories.paths()) as fig:
            ax = fig.subplots()
            count_df = df.groupby(["platform", "category"], as_index=False).size().rename(columns={"size": "count"})
            sns.barplot(count_df, x="category", y="count", hue="platform", ax=ax)

        # Plot the % of imprecise members in each category grouped by platform
        with new_figure(self.categories_percent.paths()) as fig:
            ax = fig.subplots()
            total_df = df.groupby(["platform"], as_index=False).size().rename(columns={"size": "total"})
            count_df = df.groupby(["platform", "category"], as_index=False).size()
            percent_df = total_df.merge(count_df, on="platform")
            percent_df["% of all imprecise"] = 100 * percent_df["size"] / percent_df["total"]
            sns.barplot(percent_df, x="category", y="% of all imprecise", hue="platform", ax=ax)

        # Plot the % of imprecise members in each category that alias with pointer members,
        # grouped by platform
        with new_figure(self.categories_alias_ptr.paths()) as fig:
            ax = fig.subplots()
            count_df = df.groupby(["platform", "category"], as_index=False).size()
            alias_ptr = df[df["alias_kind"] == "alias-ptr"].groupby(["platform", "category"], as_index=False).size()
            percent_df = alias_ptr.merge(count_df, on=["platform", "category"], suffixes=["_ptr", "_total_by_kind"])
            percent_df["% aliasing with ptr"] = 100 * percent_df["size_ptr"] / percent_df["size_total_by_kind"]
            sns.barplot(percent_df, x="category", y="% aliasing with ptr", hue="platform", ax=ax)

        # Plot distribution of imprecise sub-object sizes grouped by platform, with one faced for
        # each aliasing kind.
        with new_facet(self.categories_sizes_per_alias_kind.paths(), df, row="platform", height=5, aspect=2.5) as facet:
            min_size = int(np.log2(df["size"].min()))
            max_size = int(np.log2(df["size"].max())) + 1
            hist_bins = [2**x for x in range(min_size, max_size)]
            # XXX can not use log_scale=2 with static bin widths
            facet.map_dataframe(sns.histplot, x="size", hue="category", bins=hist_bins, element="step")
            for ax in facet.axes.ravel():
                ax.set_xscale("log", base=2)
            facet.add_legend()


class ImpreciseSubobjectPaddingEstimator(PlotTask):
    """
    Determine the cost in terms of padding to fix the sub-object imprecision.
    """
    pass


class ImpreciseSubobjectLayoutReorderingEstimator(PlotTask):
    """
    Determine whether it is possible to reorder fields to fix the sub-object imprecision.
    """
    pass
