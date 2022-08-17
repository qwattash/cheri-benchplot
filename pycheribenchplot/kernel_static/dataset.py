import asyncio as aio
import shutil
from enum import Enum

import numpy as np
import pandas as pd

from ..core.config import DatasetArtefact, DatasetName
from ..core.csv import CSVDataSetContainer
from ..core.dataset import (DatasetProcessingError, Field, align_multi_index_levels, check_multi_index_aligned,
                            filter_aggregate, index_where)
from ..core.elf.symbolizer import ELFSymbolReader
from ..core.instance import InstanceKernelABI


class KernelSymbolsDataset(CSVDataSetContainer):
    """
    Auxiliary dataset to load kernel symbols
    """
    dataset_config_name = DatasetName.KERNEL_SYMBOLS
    dataset_source_id = DatasetArtefact.KERNEL_SYMBOLS

    fields = [
        Field.index_field("path", dtype=str),
        Field.index_field("name", dtype=str),
        Field.index_field("addr", dtype=np.uint64),
        Field("dynamic", dtype=bool),
        Field("size", dtype=int),
        Field("type", dtype=str)
    ]

    def output_file(self):
        return super().output_file().with_suffix(".csv")

    def kernel_asset_file(self):
        return self.benchmark.get_instance_asset_path() / "kernel.full"

    def kernel_object_path(self):
        kernel = self.benchmark.cheribsd_rootfs_path / "boot" / f"kernel.{self.benchmark.config.instance.kernel}" / "kernel.full"
        if not kernel.exists():
            self.logger.debug("Kernel %s not found, fallback to FPGA kernel dir", kernel)
            # FPGA kernels fallback location
            kernel = self.benchmark.user_config.sdk_path / f"kernel-{self.benchmark.config.instance.cheri_target}.{self.benchmark.config.instance.kernel}.full"
        if not kernel.exists():
            self.logger.warning("Kernel %s not found in kernel.<CONF> directories, fallback to default kernel", kernel)
            kernel = self.benchmark.cheribsd_rootfs_path / "boot" / "kernel" / "kernel.full"
        return kernel

    def before_run(self):
        super().before_run()
        # Cleaup previous run assets
        if self.kernel_asset_file().exists():
            self.kernel_asset_file().unlink()

    async def after_extract_results(self, script, instance):
        """
        After benchmarks completed on the instance we generate
        the kernel symbols extract.
        XXX-AM: This could run in parallel with the instance running,
        if we *really* cared.
        """
        if not self.kernel_object_path().exists():
            self.logger.debug("Copy instance kernel asset %s", self.kernel_object_path())
            shutil.copy(self.kernel_object_path(), self.kernel_asset_file())

        sym_loader = ELFSymbolReader.create(self.benchmark.session, self.kernel_object_path())
        df = sym_loader.load_to_df()
        df.to_csv(self.output_file(), index=False)

    def load(self):
        df = self._load_csv(self.output_file())
        self._append_df(df)
        # Fill the symbols into the current address-space mappings manager
        addrspace = self.benchmark.sym_resolver.register_address_space("kernel.full", shared=True)
        addrspace.add_symbols(self.df)
        self.benchmark.sym_resolver.register_address_space_alias("kernel.full", "kernel")
        # Load the dwarf information for the kernel as well
        arch_pointer_size = self.benchmark.config.instance.kernel_pointer_size
        self.benchmark.dwarf_helper.register_object("kernel.full", self.kernel_object_path(), arch_pointer_size)


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
        return self.benchmark.get_output_path() / f"struct-stats-{self.benchmark.uuid}.csv"

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

        dw = self.benchmark.dwarf_helper.get_object("kernel.full")
        # This will take a while, so keep async things moving forward
        await aio.to_thread(dw.parse_dwarf)
        dwarf_data = dw.get_dwarf_data()
        df = dwarf_data.get_struct_info()
        df.to_csv(self.output_file())

    def load(self):
        df = self._load_csv(self.output_file())
        self._append_df(df)


class KernelStructMemberDataset(KernelStructDWARFInfo):
    """
    Kernel struct member-level information.
    """
    dataset_config_name = DatasetName.KERNEL_STRUCT_MEMBER_STATS

    def post_merge(self):
        super().post_merge()
        self.merged_df = align_multi_index_levels(self.merged_df, ["name", "src_file", "src_line", "member_name"],
                                                  fill_value=np.nan)

    def gen_pahole_table(self, selector: pd.Series = None):
        """
        Generate a pahole table for the structures in the merged dataframe.
        The selector can be used to filter out unwanted structures.
        This will return a table with the following properties:
        - index: the iteration level is dropped as it is unused, the member_name level is swapped for
        the new member_index level.
        - rows: extra synthetic members representing the padding are created. These members
        are named from the previous member name and offset and have the member_size column set to
        the padding width.
        - columns: only the member_offset and member_pad are retained

        Note that the index will be aligned at the member_name level. This will include the padding synthetic
        members for all datasets but places without padding will have a NaN for the synthetic member size.
        Offset ordering is expected to be sensible so that the table rows will stay aligned.
        The returned dataset is not sorted.
        """
        if selector is not None and selector.dtype != bool:
            raise TypeError("selector must be a bool series")
        df = self.merged_df.droplevel("iteration")
        if selector is not None:
            df = df.loc[selector]

        # Swap the member_name for member_offset. We can drop NaN member_offsets (missing) as they will
        # be reintroduced later as synthetic members if needed.
        df = df.dropna(subset=["member_offset"])
        pahole_df = df.reset_index("member_name").set_index("member_offset", append=True)
        # Now generate the synthetic padding members, only for those members that have padding.
        synth_members = df.loc[df["member_pad"] != 0, ["member_offset", "member_pad", "member_size"]].copy()
        synth_members.loc[:, "member_offset"] += synth_members["member_size"]
        synth_members["member_size"] = synth_members["member_pad"]
        synth_members = synth_members.reset_index("member_name").set_index("member_offset", append=True)
        synth_members["member_name"] = synth_members["member_name"] + ".pad"
        # Merge the synthetic members. This should result in the frame becoming aligned again.
        # If not, something went wrong.
        out_cols = ["member_name", "member_size"]
        pahole_df = pd.concat([pahole_df[out_cols], synth_members[out_cols]], axis=0)
        pahole_df = pahole_df.sort_index()
        # If members have zero size, there may be duplicate index entries with the member_offset.
        # Add a last-level index to ensure deduplication
        member_index = pahole_df.groupby(["dataset_id", "name", "src_file", "src_line"]).cumcount()
        pahole_df["member_index"] = member_index
        # Now replace the member_offset index level with the member_index column
        pahole_df = pahole_df.reset_index("member_offset").set_index("member_index", append=True)
        # Align the missing indexes for longer structures
        pahole_df = align_multi_index_levels(pahole_df, ["name", "src_file", "src_line", "member_index"])
        return pahole_df


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
        new_df = align_multi_index_levels(new_df, ["name", "src_file", "src_line"], fill_value=np.nan)

        self.merged_df = new_df

    def aggregate(self):
        # Just add identity columns
        self.agg_df = self._add_aggregate_columns(self.merged_df)

    def post_aggregate(self):
        super().post_aggregate()
        # Align and compute deltas
        new_df = self._add_delta_columns(self.agg_df)
        self.agg_df = self._compute_delta_by_dataset(new_df)
