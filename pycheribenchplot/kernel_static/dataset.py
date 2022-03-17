import asyncio as aio
from enum import Enum
from shutil import copyfileobj

import numpy as np
import pandas as pd

from ..core.csv import CSVDataSetContainer
from ..core.dataset import (DatasetArtefact, DatasetName, DatasetProcessingError, Field, align_multi_index_levels)
from ..core.instance import InstanceKernelABI


class SetboundsKind(Enum):
    STACK = "s"
    HEAP = "h"
    SUBOBJECT = "o"
    GLOBAL = "g"
    CODE = "c"
    UNKNOWN = "?"

    def __lt__(self, o):
        return self.value < o.value

    def __le__(self, o):
        return self.value <= o.value


class KernelSubobjectBoundsDataset(CSVDataSetContainer):
    """
    Extract subobject bounds from kernel builds.
    For the current benchmark instance, we grab the instance configuration and determine
    which kernel config we should look for
    """
    dataset_config_name = DatasetName.KERNEL_CSETBOUNDS_STATS
    dataset_source_id = DatasetArtefact.KERNEL_CSETBOUNDS_STATS

    fields = [
        Field("alignment_bits", dtype=float),
        Field("size", dtype=float),
        Field.str_field("src_module"),
        Field("kind", dtype="object", importfn=lambda v: SetboundsKind(v)),
        Field.str_field("source_loc"),
        Field.str_field("compiler_pass"),
        Field.str_field("details"),
    ]

    def _kernel_build_path(self):
        mconfig = self.benchmark.manager_config
        iconfig = self.benchmark.instance_config
        build_base = mconfig.build_path / f"cheribsd-{iconfig.cheri_target}-build"
        path_match = list(build_base.glob(f"**/sys/{iconfig.kernel}"))
        if len(path_match) == 0:
            self.logger.error("No kernel build directory for %s in %s", iconfig.kernel, build_base)
            raise DatasetProcessingError("Missing kernel build directory")
        assert len(path_match) == 1
        return path_match[0]

    def _llvm_output_kernel(self):
        build_path = self._kernel_build_path()
        path = build_path / "kernel-subobject-bounds-stats.csv"
        if not path.exists():
            self.logger.error(
                "Missing csetbounds stats for %s, you may need to" +
                " build the kernels with --collect-csetbounds-stats", self.benchmark.instance_config.kernel)
            raise DatasetProcessingError("Missing kernel csetbounds stats")
        return path

    def _llvm_output_modules(self):
        build_path = self._kernel_build_path()
        return list(build_path.glob("modules/**/kernel-subobject-bounds-stats.csv"))

    def output_file(self):
        base = super().output_file()
        return base.with_suffix(".csv")

    async def after_extract_results(self):
        await super().after_extract_results()
        iconfig = self.benchmark.instance_config
        if iconfig.kernelabi == InstanceKernelABI.PURECAP:
            with open(self.output_file(), "w+") as outfd:
                df = pd.read_csv(self._llvm_output_kernel())
                df["src_module"] = "kernel"
                df.to_csv(outfd)
                for stats_file in self._llvm_output_modules():
                    df = pd.read_csv(stats_file)
                    df["src_module"] = stats_file.parent.name
                    df.to_csv(outfd, header=False)
        else:
            # Just make an empty file for load() to find it
            pd.DataFrame({}, columns=self.input_all_columns()).to_csv(self.output_file())

    def load(self):
        conv = {
            "alignment_bits": lambda v: np.nan if str(v).startswith("<unknown") else int(v),
            "size": lambda v: np.nan if str(v).startswith("<unknown") else int(v)
        }
        df = self._load_csv(self.output_file(), converters=conv)
        self._append_df(df)


class KernelStructDWARFInfo(CSVDataSetContainer):
    """
    Extract kernel struct information from kernel DWARF info.
    This is a base dataset that is used to build different tables.
    """
    dataset_source_id = DatasetArtefact.KERNEL_STRUCT_STATS
    fields = [
        Field.index_field("name", dtype=str),
        Field.index_field("src_file", dtype=str),
        Field.index_field("src_line", dtype=int),
        Field.index_field("member_name", dtype=str),
        Field.data_field("size", dtype=float),
        Field.str_field("from_path"),
        Field.data_field("member_offset", dtype=float),
        Field.data_field("member_bit_offset", dtype=float),
        Field.data_field("member_size", dtype=float),
        Field("member_bit_size", dtype=float),
        Field.data_field("member_pad", dtype=float),
        Field("member_bit_pad", dtype=float),
        Field.str_field("member_type_name"),
        Field.str_field("member_type_base_name"),
        Field("member_type_is_ptr", dtype=bool),
        Field("member_type_is_struct", dtype=bool),
        Field("member_type_is_typedef", dtype=bool),
        Field("member_type_is_array", dtype=bool),
        Field("member_type_array_items", dtype=float),
        Field("member_type_size", dtype=float),
        Field("member_type_pad", dtype=float),
        Field.str_field("member_type_src_file"),
        Field("member_type_src_line", dtype=float),
        Field("member_type_params", dtype=object)
    ]

    def full_kernel_path(self):
        mconfig = self.benchmark.manager_config
        iconfig = self.benchmark.instance_config
        build_base = mconfig.build_path / f"cheribsd-{iconfig.cheri_target}-build"
        kern_match = list(build_base.glob(f"**/sys/{iconfig.kernel}"))
        if len(kern_match) == 0:
            self.logger.error("No kernel build directory for %s in %s", iconfig.kernel, build_base)
            raise Exception("Can not find cheribuild kernel")
        assert len(kern_match) == 1
        kern = kern_match[0] / "kernel.full"
        if not kern.exists():
            self.logger.error("No kernel.full for %s at %s", iconfig.kernel, kern)
            raise Exception("Can not find cheribuild kernel")
        return kern

    def output_file(self):
        return super().output_file().with_suffix(".csv")

    async def after_extract_results(self):
        await super().after_extract_results()
        kern_path = self.full_kernel_path()

        # Avoid regenerating the dataset if the kernel is older than the dataset
        if self.output_file().exists():
            stat = self.output_file().stat()
            out_mtime = stat.st_mtime
            stat = kern_path.stat()
            kern_mtime = stat.st_mtime
            if kern_mtime < out_mtime:
                return

        self.benchmark.dwarf_helper.register_object("kernel.full", kern_path)
        dw = self.benchmark.dwarf_helper.get_object("kernel.full")
        # This will take a while, so keep async things moving forward
        await aio.to_thread(dw.parse_dwarf)
        dwarf_data = dw.get_dwarf_data()
        df = dwarf_data.get_struct_info()
        df.to_csv(self.output_file())

    def load(self):
        df = self._load_csv(self.output_file())
        self._append_df(df)


class KernelStructSizeDataset(KernelStructDWARFInfo):
    """
    Extract kernel struct sizes from the kernel DWARF info.
    """
    dataset_config_name = DatasetName.KERNEL_STRUCT_STATS
    fields = [
        Field.data_field("total_pad", dtype=int, isderived=True),
        Field.data_field("ptr_count", dtype=int, isderived=True),
        Field.data_field("member_count", dtype=int, isderived=True),
        Field.data_field("nested_ptr_count", dtype=int, isderived=True),
        Field.data_field("nested_member_count", dtype=int, isderived=True),
        Field.data_field("nested_packed_size", dtype=int, isderived=True),
        Field.data_field("nested_total_pad", dtype=int, isderived=True),
    ]

    def _check_has_nested(self, group):
        is_flex_array = group["member_type_is_array"] & (group["member_type_array_items"] == 0)
        return group["member_type_is_struct"] & ~is_flex_array

    def post_merge(self):
        super().post_merge()
        # We do not care about per-member information so we collapse by grouping and produce extra columns
        # Do not do this in the aggregation step as it is meant for median/quartile calculations so keep it
        # separate for consistency
        group_key = ["dataset_id", "name", "src_file", "src_line"]
        grouped = self.merged_df.reset_index("member_name").groupby(group_key)

        new_df = grouped[["size"]].first()
        new_df["total_pad"] = grouped["member_pad"].sum()
        new_df["total_pad"] += grouped["member_bit_pad"].sum() / 8

        new_df["ptr_count"] = grouped["member_type_is_ptr"].sum()
        # The member_name column is NaN if the member is a dummy member added for empty structures,
        # count() will ignore NaN.
        new_df["member_count"] = grouped["member_name"].count()

        # Compute nested counts
        new_df["nested_ptr_count"] = np.nan
        new_df["nested_member_count"] = np.nan
        new_df["nested_total_pad"] = np.nan
        new_df["nested_packed_size"] = np.nan
        new_df["_visited"] = False
        has_nested = grouped.apply(lambda g: self._check_has_nested(g).any())
        new_df.loc[~has_nested, "nested_ptr_count"] = new_df.loc[~has_nested, "ptr_count"]
        new_df.loc[~has_nested, "nested_member_count"] = new_df.loc[~has_nested, "member_count"]
        new_df.loc[~has_nested, "nested_total_pad"] = new_df.loc[~has_nested, "total_pad"]
        new_df.loc[~has_nested, "_visited"] = True
        nested_key_fields = ["dataset_id", "member_type_base_name", "member_type_src_file", "member_type_src_line"]
        visit_queue = list(new_df[has_nested].index)
        while len(visit_queue):
            struct_key = visit_queue.pop(0)
            if new_df.loc[struct_key, "_visited"].all():
                continue
            group_df = grouped.get_group(struct_key)
            nested_members = group_df[self._check_has_nested(group_df)]
            if len(nested_members) == 0:
                continue
            # Join to find nested struct records, now we can check if we have data for all of them
            nested_structs = nested_members.merge(new_df,
                                                  left_on=nested_key_fields,
                                                  right_on=group_key,
                                                  suffixes=("", "_nested"))
            # Check if we have visited all of the nested structs
            if nested_structs["_visited"].all():
                # Actually compute the count
                nested_structs["array_mul"] = 1
                array_mul = nested_structs["array_mul"].mask(nested_structs["member_type_is_array"],
                                                             nested_structs["member_type_array_items"])
                nested_ptr_count = (nested_structs["nested_ptr_count"] * array_mul).sum()
                nested_m_count = (nested_structs["nested_member_count"] * array_mul).sum()
                nested_pad = (nested_structs["nested_total_pad"] * array_mul).sum()
                new_df.loc[struct_key, "nested_ptr_count"] = (new_df.loc[struct_key, "ptr_count"] + nested_ptr_count)
                new_df.loc[struct_key,
                           "nested_member_count"] = (new_df.loc[struct_key, "member_count"] + nested_m_count)
                new_df.loc[struct_key, "nested_total_pad"] = (new_df.loc[struct_key, "total_pad"] + nested_pad)
                new_df.loc[struct_key, "_visited"] = True
            else:
                # enqueue back after all the missing members have a chance to be resolved
                visit_queue.append(struct_key)
        new_df.drop(columns=["_visited"], inplace=True)
        new_df["nested_packed_size"] = new_df["size"] - new_df["nested_total_pad"]

        # integrity checks
        assert (new_df["member_count"] >= new_df["ptr_count"]).all()
        assert (new_df["nested_packed_size"] >= 0).all()
        assert (new_df["nested_total_pad"] >= 0).all()

        # Drop duplicate names with same size that are defined in multiple files
        new_df = new_df.reset_index().drop_duplicates(["dataset_id", "name", "size"]).set_index(new_df.index.names)

        # Align and compute deltas
        new_df = align_multi_index_levels(new_df, ["name", "src_file", "src_line"], fill_value=np.nan)
        new_df = self._add_delta_columns(new_df)
        self.merged_df = self._compute_delta_by_dataset(new_df)

    def _get_aggregation_strategy(self):
        agg = super()._get_aggregation_strategy()
        metric_cols = self.merged_df.columns.get_level_values("metric")
        metric_level_idx = self.merged_df.columns.names.index("metric")
        match = metric_cols.isin(agg.keys())
        mapped_agg = {c: agg[c[metric_level_idx]] for c in self.merged_df.columns[match]}
        return mapped_agg

    def aggregate(self):
        super().aggregate()
        grouped = self.merged_df.groupby(["dataset_id"])
        self.agg_df = self._compute_aggregations(grouped)
