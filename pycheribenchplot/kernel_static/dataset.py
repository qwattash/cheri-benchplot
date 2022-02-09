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


class KernelStructSizeDataset(CSVDataSetContainer):
    """
    Extract kernel struct sizes from the kernel DWARF info.
    """
    dataset_config_name = DatasetName.KERNEL_STRUCT_STATS
    dataset_source_id = DatasetArtefact.KERNEL_STRUCT_STATS
    fields = [
        Field.index_field("name", dtype=str),
        Field.index_field("src_file", dtype=str),
        Field.index_field("src_line", dtype=int),
        Field.index_field("member_name", dtype=str),
        Field.data_field("size", dtype=int),
        Field.str_field("from_path"),
        Field.str_field("desc", isderived=True),
        Field.data_field("member_size", dtype=int),
        Field("member_offset", dtype=int),
        Field.data_field("member_pad", dtype=int),
        Field.str_field("member_type_name"),
        Field("member_type_kind", dtype=object),
        # This should probably be dropped as duplicates member_size
        Field("member_type_size", dtype=int),
        # Not a data field as this is already accounted for in member_pad
        Field("member_type_pad", dtype=int),
        # This is a DIE offset, unintresting ouside the DWARF interface
        Field("member_type_ref_offset", dtype=int),
        Field.str_field("member_type_src_file"),
        Field("member_type_src_line", dtype=int),
        Field("member_type_fn_params", dtype=object)
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

    def post_merge(self):
        super().post_merge()
        # compute patched anon names in the "desc" column
        df_index = self.merged_df.index.to_frame()
        name = df_index["name"]
        # bool becomes object when moved to the index for some reason...
        anon = df_index["is_anon"].astype(np.bool)
        tmp = name.mask(anon, "anon")
        suffix_file = df_index["src_file"].where(anon, "")
        suffix_line = df_index["src_line"].astype(str).where(anon, "")
        self.merged_df["desc"] = tmp.str.cat([suffix_file, suffix_line], sep=":").where(anon, name)

        # Align and compute deltas
        new_df = align_multi_index_levels(self.merged_df, ["name", "src_file", "src_line", "is_anon"],
                                          fill_value=np.nan)
        new_df = self._add_delta_columns(new_df)
        self.merged_df = self._compute_delta_by_dataset(new_df)

    def _get_aggregation_strategy(self):
        agg = super()._get_aggregation_strategy()
        metric_cols = self.merged_df.columns.get_level_values("metric")
        metric_level_idx = self.merged_df.columns.names.index("metric")
        match = metric_cols.isin(agg.keys())
        mapped_agg = {c: agg[c[metric_level_idx]] for c in self.merged_df.columns[match]}
        # agg = {(c, "sample"): v for c, v in agg.items()}
        return mapped_agg

    def aggregate(self):
        super().aggregate()
        grouped = self.merged_df.groupby(["__dataset_id"])
        self.agg_df = self._compute_aggregations(grouped)


class KernelStructFragDataset:
    """
    Internal fragmentation for kernel structures
    """
    pass
