import asyncio as aio
from shutil import copyfile

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
        Field("alignment_bits", dtype=int),
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

    def _output_file_cheribuild(self):
        mconfig = self.benchmark.manager_config
        iconfig = self.benchmark.instance_config
        build_base = mconfig.build_path / f"cheribsd-{iconfig.cheri_target}-build"
        path_match = list(build_base.glob(f"**/sys/{iconfig.kernel}"))
        if len(path_match) == 0:
            return None
        assert len(path_match) == 1
        return path_match[0] / "kernel-subobject-bounds-stats.csv"

    def output_file(self):
        base = super().output_file()
        return base.with_suffix(".csv")

    async def after_extract_results(self):
        await super().after_extract_results()
        cheribuild_out = self._output_file_cheribuild()
        iconfig = self.benchmark.instance_config
        if iconfig.kernelabi == InstanceKernelABI.PURECAP:
            if not cheribuild_out.exists():
                self.logger.error("Missing %s, you may need to build the kernels with --collect-csetbounds-stats",
                                  cheribuild_out)
                return
            await aio.to_thread(copyfile, cheribuild_out, self.output_file())
        else:
            # Just make an empty file for load() to find it
            pd.DataFrame({}, columns=self.input_all_columns()).to_csv(self.output_file())

    def load(self):
        conv = {"size": lambda v: np.nan if v == '<unknown>' else int(v)}
        df = self._load_csv(self.output_file(), converters=conv)
        self._append_df(df)
