import asyncio as aio
from shutil import copyfileobj

import numpy as np
import pandas as pd

from ..core.csv import CSVDataSetContainer
from ..core.dataset import DatasetArtefact, DatasetName, Field
from ..core.instance import InstanceKernelABI


class SubobjectBoundsStatsDataset(CSVDataSetContainer):
    """
    Extract subobject bounds statistics from cheribuild targets.
    These can be used to inspect statically the distribution of
    subobject bounds.
    """
    fields = [
        Field("alignment_bits", dtype=float),
        Field("size", dtype=float),
        Field.str_field("kind"),
        Field.str_field("source_loc"),
        Field.str_field("compiler_pass"),
        Field.str_field("details"),
    ]


class KernelSubobjectBoundsDataset(SubobjectBoundsStatsDataset):
    """
    Extract subobject bounds from kernel builds.
    For the current benchmark instance, we grab the instance configuration and determine
    which kernel config we should look for
    """
    dataset_config_name = DatasetName.KERNEL_CSETBOUNDS_STATS
    dataset_source_id = DatasetArtefact.KERNEL_CSETBOUNDS_STATS

    def _output_files_cheribuild(self):
        mconfig = self.benchmark.manager_config
        iconfig = self.benchmark.instance_config
        build_base = mconfig.build_path / f"cheribsd-{iconfig.cheri_target}-build"
        path_match = list(build_base.glob(f"**/sys/{iconfig.kernel}"))
        if len(path_match) == 0:
            self.logger.error("No kernel build directory for %s in %s", iconfig.kernel, build_base)
            return None
        assert len(path_match) == 1
        return list(path_match[0].glob("**/kernel-subobject-bounds-stats.csv"))
        # return path_match[0] / "kernel-subobject-bounds-stats.csv"

    def output_file(self):
        base = super().output_file()
        return base.with_suffix(".csv")

    async def after_extract_results(self):
        await super().after_extract_results()
        cheribuild_out = self._output_files_cheribuild()
        iconfig = self.benchmark.instance_config
        if iconfig.kernelabi == InstanceKernelABI.PURECAP:
            if not cheribuild_out:
                self.logger.error(
                    "Missing csetbounds stats for %s, you may need to" +
                    " build the kernels with --collect-csetbounds-stats", iconfig.kernel)
                return
            with open(self.output_file(), "w+") as outfd:
                header = True
                for stats_file in cheribuild_out:
                    df = pd.read_csv(stats_file)
                    df.to_csv(outfd, header=header)
                    header = False
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
        Field.index_field("size", dtype=int),
        Field("is_anon", dtype=bool),
        Field.str_field("from_path"),
        Field.str_field("src_file"),
        Field("src_line", dtype=int),
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
        df = await aio.to_thread(dw.extract_struct_info)
        df.to_csv(self.output_file())

    def load(self):
        df = self._load_csv(self.output_file())
        self._append_df(df)


class KernelStructFragDataset:
    """
    Internal fragmentation for kernel structures
    """
    pass
