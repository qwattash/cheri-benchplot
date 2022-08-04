from pathlib import Path
import typing
from ..core.analysis import BenchmarkAnalysis
from ..core.dataset import DatasetName
import subprocess, os, shutil

from dataclasses import dataclass, field
from ..core.config import TemplateConfig, ConfigPath

@dataclass
class DrCacheSimConfig(TemplateConfig):
    drrun_path: ConfigPath = Path("dynamorio/bin64/drrun")
    remove_saved_results: bool = False
    LL_cache_sizes: typing.List[str] = field(default_factory=list)
    rerun_sim: bool = True
class DrCacheSimRun(BenchmarkAnalysis):
    require = {DatasetName.QEMU_DYNAMORIO}
    name: str = "drcachesim_run"
    description: str = "Run drcachesim"
    analysis_options_class = DrCacheSimConfig
    def process_datasets(self):
        run_args = ['-t', 'drcachesim', '-indir']
        processes_dict = {} 
        args = self.config
        dset = self.get_dataset(DatasetName.QEMU_DYNAMORIO)
        trace_file = dset.output_file()
        self.logger.info(f"Running drcachesim on {trace_file}")
        indir = trace_file.parent

        base = os.path.basename(indir)
        if base == "":
            base  = os.path.basename(os.path.split(indir)[0])
        outdir = base + '-results'
        if args.remove_saved_results:
            shutil.rmtree(outdir, ignore_errors=True)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for s in args.LL_cache_sizes:
            if os.path.isfile(indir + "/LL_size_" + s + ".txt") and not args.rerun_sim:
                continue
            p = subprocess.Popen([args.bin] + run_args + [indir, '-LL_size', s], stderr=subprocess.PIPE)
            processes_dict[p] = s

        for kvp in processes_dict.items():
            p = kvp[0]
            size = kvp[1]
            err = p.communicate()[1];
            with open(outdir + "/LL_size_" + size + ".txt", "w") as f: 
                f.write(err.decode())
                

 