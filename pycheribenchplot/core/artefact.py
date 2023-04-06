from dataclasses import dataclass
from pathlib import Path
from typing import ForwardRef

import pandas as pd
import pandera as pa


@dataclass
class TargetRef:
    """
    Internal helper for targets
    """

    #: The unique key for the target
    name: str
    #: The attribute name for the target in the Task subclass.
    attr: str


class Target:
    """
    Helper to represent output artifacts of a task.
    This is the base class to also support non-file targets if necessary.
    """
    def is_file(self):
        """
        When true, the target should be a subclass of :class:`FileTarget`
        """
        return False


class DataFrameTarget(Target):
    """
    Target wrapping an output dataframe from a task.
    """
    def __init__(self, schema: pa.DataFrameSchema):
        self.schema = schema
        self._df = None

    def assign(self, df: pd.DataFrame):
        self._df = self.schema.validate(df)

    def get(self) -> pd.DataFrame:
        return self._df.copy()


class FileTarget(Target):
    """
    Base class for a target output file.

    The factory methods should be used to generate paths for the file targets.
    """
    @classmethod
    def from_task(cls, task: "SessionTask | BenchmarkTask", prefix: str = None, ext: str = None, **kwargs) -> "Self":
        """
        Create a task path using the task identifier and the parent session data root path.

        :param task: The task that generates this file
        :param prefix: Additional name used to generate the filename, in case the task generates
        multiple files
        :param ext: Optional file extension
        :param `**kwargs`: Forwarded arguments to the :class:`FileTarget` constructor.
        :returns: The file target
        """
        name = re.sub(r"\.", "-", task.task_id)
        if prefix:
            name = f"{prefix}-{name}"
        path = Path(name)

        if ext:
            if not ext.startswith("."):
                ext = "." + ext
            path = path.with_suffix(ext)

        if isinstance(task, SessionTask):
            return cls.from_session(task.session, path, **kwargs)
        elif isinstance(task, BenchmarkTask):
            return cls.from_benchmark(task.benchmark, path, **kwargs)
        else:
            raise TypeError(f"Invalid task type {task.__class__}")

    @classmethod
    def from_session(cls, session: "Session", path: Path) -> "Self":
        """
        Build a file target descriptor from a session.

        The descriptor will point to a single file in the session data root path.
        :param session: The target session
        :param path: The target file name
        :returns: The file target
        """
        raise NotImplementedError("Must override")

    @classmethod
    def from_benchmark(cls, benchmark: "Benchmark", path: Path, use_iterations: bool = False) -> "Self":
        """
        Build a file target descriptor for a benchmark.

        The descriptor will point to one or more files in the benchmark data root.
        If the iterations parameter is set, the descriptor will map to multiple files,
        one for each benchmark iteration.
        :param benchmark: The target benchmark
        :param path: The target file name
        :param use_iterations: Generate multiple files, one per iteration.
        :returns: The file target
        """
        raise NotImplementedError("Must override")

    def __init__(self, paths: list[Path], remote_paths: list[Path] | None = None, use_iterations: bool = False):
        self._paths = paths
        self._remote_paths = remote_paths
        self.use_iterations = use_iterations

    @property
    def path(self) -> Path:
        """
        Shorthand to return the path for targets that do not depend on iterations.
        If the path depends on the iteration index, and there are multiple paths available, this raises a TypeError.
        """
        if len(self._paths) > 1:
            raise ValueError("Can not use shorthand path property when multiple paths are present")
        return self._paths[0]

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    @property
    def remote_path(self) -> Path:
        """
        Shorthand to return the path for targets that do not depend on iterations.
        If the path depends on the iteration index, this raises a TypeError.
        """
        if len(self._paths) > 1:
            raise ValueError("Can not use shorthand remote_path property when multiple paths are present")
        return self.remote_paths[0]

    @property
    def remote_paths(self) -> list[Path]:
        """
        Same as path but all paths are coverted rebased to the guest data output directory
        """
        assert self.needs_extraction(), "Can not use remote paths if file does not need extraction"
        return list(self._remote_paths)

    def is_file(self):
        return True

    def needs_extraction(self):
        raise NotImplementedError("Must override")


class DataFileTarget(FileTarget):
    """
    A target output file that is generated on the guest and needs to be extracted.
    """
    @classmethod
    def from_session(cls, session: "Session", path: Path) -> FileTarget:
        raise TypeError("DataFileTarget does not support session-wide scope, only benchmark scope")

    @classmethod
    def from_benchmark(cls, benchmark: "Benchmark", path: Path, use_iterations: bool = False) -> FileTarget:
        benchmark_data_root = benchmark.get_benchmark_data_path()
        guest_data_root = benchmark.config.remote_output_dir
        if use_iterations:
            base_paths = map(benchmark.get_benchmark_iter_data_path, range(benchmark.config.iterations))
            local = [base / path for base in base_paths]
            remote = [guest_data_root / p.relative_to(benchmark_data_root) for p in local]
        else:
            local = [benchmark_data_root / path]
            remote = [guest_data_root / path]
        return cls(local, remote, use_iterations=use_iterations)

    def needs_extraction(self):
        return True


class LocalFileTarget(FileTarget):
    """
    A target output file that is generated on the host and does not need to be extracted
    """
    @classmethod
    def from_session(cls, session: "Session", path: Path) -> FileTarget:
        return cls([session.get_data_root_path() / path])

    @classmethod
    def from_benchmark(cls, benchmark: "Benchmark", path: Path, use_iterations: bool = False) -> FileTarget:
        benchmark_data_root = benchmark.get_benchmark_data_path()
        if use_iterations:
            base_paths = map(benchmark.get_benchmark_iter_data_path, range(benchmark.config.iterations))
            local = [base / path for base in base_paths]
        else:
            local = [benchmark_data_root / path]
        return cls(local)

    def needs_extraction(self):
        return False
