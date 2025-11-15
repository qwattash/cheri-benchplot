import typing
from pathlib import Path

from .config import DatasetArtefact
from .csv import CSVDataSetContainer
from .dataset import Field


class ProcstatDataset(CSVDataSetContainer):
    dataset_source_id = DatasetArtefact.PROCSTAT
    fields = [
        Field.index_field("PID", dtype=int),
        Field.data_field("START", dtype=int, importfn=lambda x: int(x, 16)),
        Field.data_field("END", dtype=int, importfn=lambda x: int(x, 16)),
        Field.str_field("PRT"),
        Field("RES"),
        Field("PRES"),
        Field("REF"),
        Field("SHD"),
        Field.str_field("FLAG"),
        Field.str_field("TP"),
        Field.str_field("PATH")
    ]

    def _load_csv(self, path: Path, **kwargs):
        kwargs["sep"] = "\s+"
        return super()._load_csv(path, **kwargs)

    def load(self):
        path = self.output_file()
        csv_df = self._load_csv(path)
        self._append_df(csv_df)
        # Register the mapped binaries to the benchmark symbolizer
        for pid, base, guest_path in self.mapped_binaries(self.benchmark.uuid):
            local_path = self.benchmark.cheribsd_rootfs_path / guest_path.relative_to("/")
            self.benchmark.register_mapped_binary(base, local_path, pid)

    def mapped_binaries(self, dataset_id) -> typing.Iterator[typing.Tuple[int, str]]:
        """
        Iterate over (base_addr, path) of all the binaries mapped for the
        given dataset id.
        """
        xsection = self.df.xs(dataset_id, level="dataset_id")
        grouped = xsection.groupby("PID")
        for pid, chunk in grouped:
            path_addr = chunk.groupby("PATH")["START"].min()
            for path, addr in path_addr.items():
                if path == "":
                    continue
                yield (pid, addr, Path(path))

    def output_file(self):
        return super().output_file().with_suffix(".csv")

    def _gen_run_procstat(self, script, proc_handle, header=True):
        """
        This should be used in subclasses to implement gen_pre_benchmark().
        Running procstat requires knowledge of the way to stop the benchmark at the correct time,
        unless we can use a generic way to stop at main() or exit()
        """
        args = ["-v", proc_handle]
        if not header:
            args = ["-h"] + args
        script.gen_cmd("procstat", args, outfile=self.output_file())
        self.logger.debug("Collected procstat info")
