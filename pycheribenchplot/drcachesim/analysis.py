from pathlib import Path
import typing
from ..core.analysis import BenchmarkAnalysis
from ..core.dataset import DatasetName
import asyncio as aio
import subprocess, os, shutil

from dataclasses import dataclass, field
from ..core.config import TemplateConfig, ConfigPath

@dataclass
class DrCacheSimConfig(TemplateConfig):
    drrun_path: ConfigPath = Path("dynamorio/bin64/drrun")
    remove_saved_results: bool = False
    LL_cache_sizes: typing.List[str] = field(default_factory=list)
    L1D_cache_sizes: typing.List[str] = field(default_factory=list)
    L1I_cache_sizes: typing.List[str] = field(default_factory=list)
    run_cache_levels: typing.List[str] = field(default_factory=list)
    rerun_sim: bool = False

class DrCacheSimRun(BenchmarkAnalysis):
    require = {DatasetName.QEMU_DYNAMORIO}
    name: str = "drcachesim"
    description: str = "Run drcachesim"
    analysis_options_class = DrCacheSimConfig
    def __init__(self, benchmark, config):
        super().__init__(benchmark, config)
        self.processes_dict = {}

    async def _run_drcachesim(self, level_arg, size, indir, out_path):
        if os.path.isfile(out_path) and not self.config.rerun_sim:
           return
        p = await aio.create_subprocess_exec(self.config.drrun_path, '-t', 'drcachesim', '-indir', indir, '-' + level_arg, size, stderr=aio.subprocess.PIPE)
        self.processes_dict[p] = out_path

    async def process_datasets(self):
        dset = self.get_dataset(DatasetName.QEMU_DYNAMORIO)
        trace_file = dset.output_file()
        indir = trace_file.parent
        base = dset.cachesim_output_dir()
        if self.config.remove_saved_results:
            shutil.rmtree(str(base), ignore_errors=True)
        if not base.exists():
            base.mkdir(parents=True)

        self.logger.info(f"Running drcachesim")
        for level in self.config.run_cache_levels:
            if level == 'LL':
                sizes = self.config.LL_cache_sizes
                out_path = base / "LL_size"
                out_path.mkdir(exist_ok=True)
                for s in sizes:
                    await self._run_drcachesim("LL_size", s, indir, out_path / f"{s}.txt")
            elif level == 'L1D':
                sizes = self.config.L1D_cache_sizes
                out_path = base / "L1D_size"
                out_path.mkdir(exist_ok=True)
                for s in sizes:
                    await self._run_drcachesim("L1D_size", s, indir, out_path / f"{s}.txt")
            elif level == 'L1I':
                sizes = self.config.L1I_cache_sizes
                out_path = base / "L1I_size"
                out_path.mkdir(exist_ok=True)
                for s in sizes:
                    await self._run_drcachesim("L1I_size", s, indir, out_path / f"{s}.txt")
            else:
                self.logger.error(f"Unknown cache level {level}")


        for kvp in self.processes_dict.items():
            p = kvp[0]
            path = kvp[1]
            err = (await p.communicate())[1];
            with open(path, "w") as f: 
                f.write(err.decode())
        self.logger.info(f"Finished drcachesim")
                

 