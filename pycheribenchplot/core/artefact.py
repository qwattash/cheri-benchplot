import re
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Callable, ForwardRef, Iterator, Type

import polars as pl
import sqlalchemy as sqla
from jinja2 import (Environment, PackageLoader, TemplateNotFound, select_autoescape)
from sqlalchemy.orm import Session as SqlSession
from typing_extensions import Self

from .borg import Borg
from .task import DatasetTask, SessionTask, Task, output


class Target(Borg):
    """
    Base class that represents an output artefact of a task.

    Targets are borg objects that share the internal state, as there is only
    one output for each unique output identifier of a task.
    The output identifier is derived from the task_id and other parameters.

    Every target is optionally backed by a file object. The file path is
    constructed from the session and related exec or analysis task.
    The file may be used for caching a target or to persist the target data.

    :param task: The task instance that generates this target.
    :param output_id: Unique name of the output within the task. The output_id
    is also used as the file path prefix.
    """
    def __init__(self,
                 task: Task,
                 output_id: str,
                 template: str | None = None,
                 loader: Callable[[Self], Task] | None = None,
                 **kwargs):
        """
        A target is associated to a task and output_id, which can be used to
        produce the unique target identifier.

        :param task: Task that produces this artefact
        :param output_id: Unique identifier of the artefact within the task.
        :param template: Template string used to produce the path name.
        By default, this is "{prefix}-{base}.{ext}".
        :param **kwargs: Optional parameterization keys. By default, the following
        parameterization keys are defined. The "ext" key for the file extension,
        defaults to "txt"; the "prefix" key, defaults to the output_id.
        """
        self._task = task
        self._output_id = output_id
        self._loader = loader
        if template:
            self._path_template = template
        else:
            self._path_template = "{prefix}/{base}.{ext}"
        self._path_parameters = kwargs
        # Normalize path parameters to lists
        for key, value in self._path_parameters.items():
            if isinstance(value, str):
                self._path_parameters[key] = [value]
        # Note that the root path parameter is special
        self._path_parameters.setdefault("prefix", [""])
        self._path_parameters.setdefault("ext", ["txt"])
        self._path_parameters.setdefault("base", [self.get_base_path()])

        # Borg state initialization occurs here
        super().__init__()

    @property
    def task(self) -> Task:
        return self._task

    @property
    def borg_state_id(self):
        """
        Note that since the unique output ID depends on the task_id, intentional
        collision of task_id values for session-scoped tasks will result in
        session-scoped outputs
        """
        output_suffix = f"-{self._output_id}" if self._output_id else ""
        return f"{self._task.task_id}-output{output_suffix}"

    def get_base_path(self) -> Path:
        """
        Build the base path for this target.
        """
        return self._output_id + "-" + re.sub(r"\.", "-", self._task.task_id)

    def get_root_path(self) -> Path:
        """
        Build the root directory path for the target.
        This defaults to the session or benchmark paths, depending on the
        task type.
        """
        if self._task.is_session_task():
            if self._task.is_exec_task():
                return self._task.session.get_data_root_path()
            else:
                return self._task.session.get_analysis_root_path()
        else:
            # per-benchmark task
            if self._task.is_exec_task():
                return self._task.benchmark.get_benchmark_data_path()
            else:
                return self._task.benchmark.get_analysis_path()

    def single_path(self) -> Path:
        """
        Check that there is only one target path and return it
        """
        paths = self.paths()
        if len(paths) != 1:
            self._task.logger.error("Associated target have invalid number of paths (%s), "
                                    "expected 1.", len(paths))
            assert False, "single_path() invariant failed"
        return paths[0]

    def __iter__(self) -> Iterator[Path]:
        """
        Iterate through all the parameterized paths.
        """
        for path in self.iter_paths():
            yield path

    def iter_paths(self, **kwargs) -> Iterator[Path]:
        """
        Iterate over parameterization key and path pairs.

        :param **kwargs: Filter keys for the path parameterization.
        :return: A generator of paths associated to the target.
        """
        param_gen = self._filter_path_parameters(**kwargs)
        params = product(*param_gen.values())
        for param_set in params:
            root = self.get_root_path()
            param_args = dict(zip(self._path_parameters.keys(), param_set))
            path = Path(self._path_template.format(**param_args))
            if path.is_absolute():
                # Never allow absolute paths, always rebase these onto root
                path = path.relative_to("/")
            yield root / path

    def paths(self, **kwargs) -> list[Path] | Path:
        return list(self.iter_paths(**kwargs))

    def get_loader(self) -> Task:
        """
        The loader for a target is a task that can be scheduled to load the output data.
        By default, this produces a DataFrameLoader with no validator.

        :return: A task that can be depended upon to load the target data.
        """
        if self._loader:
            loader_factory = self._loader
        else:
            loader_factory = make_dataframe_loader()

        return loader_factory(self)

    def _filter_path_parameters(self, **kwargs) -> dict:
        """
        Helper to generate the path parameters dict filtered by
        the key/value pairs in kwargs. If the corresponding key in kwargs
        is a raw value, use that instead of the path_parameters[key] value.
        If kwargs[key] is callable, use that as a filter for path_parameters[key]
        """
        param_gen = {}
        for key, values in self._path_parameters.items():
            if key not in kwargs:
                param_gen[key] = values
            elif callable(kwargs[key]):
                param_gen[key] = list(filter(kwargs[key], values))
            else:
                restrict = kwargs[key]
                if not isinstance(restrict, list):
                    restrict = [restrict]
                param_gen[key] = restrict
        return param_gen


class RemoteTarget(Target):
    """
    Target with an associated remote file.
    """
    def iter_remote_paths(self, **kwargs) -> Iterator[Path]:
        """
        Iterate over parameterization key and remote path pairs.
        The target may not have any remote paths, in this case this
        returns and empty iterator.

        This is only supported for exec tasks.
        If this is a session task, the path is considered relative to the session data root.
        If this is a dataset task, the path is considered relative to the benchmark data root.

        :param **kwargs: Filter keys for the path parameterization.
        :return: A generator of remote paths associated to the target.
        """
        param_gen = self._filter_path_parameters(**kwargs)
        params = product(*param_gen.values())
        for param_set in params:
            param_args = dict(zip(self._path_parameters.keys(), param_set))
            path = Path(self._path_template.format(**param_args))
            if path.is_absolute():
                path = path.relative_to("/")
            yield path

    def remote_paths(self, **kwargs) -> list[Path] | Path:
        return list(self.iter_remote_paths(**kwargs))


class BenchmarkIterationTarget(Target):
    """
    Target that specifies output files for different iterations.
    Note that this assumes that the path prefix is not already parameterized.

    This introduces the "iteration" parameterization key for the target,
    which can be used in the paths() method.
    """
    def __init__(self, task: Task, output_id: str, template: str | None = None, **kwargs):
        assert task.is_dataset_task(), "Task must be a benchmark task"
        if not template:
            template = "{prefix}/{iteration}/{base}.{ext}"
        kwargs.setdefault("iteration", list(range(task.benchmark.config.iterations)))
        # Borg state initialization occurs here
        super().__init__(task, output_id, template, **kwargs)


class RemoteBenchmarkIterationTarget(BenchmarkIterationTarget, RemoteTarget):
    """
    Same as the :class:`BenchmarkIterationTarget` but has associated remote files.
    """
    def shell_path_builder(self, **kwargs) -> "Callable[str, str]":
        """
        Return a function that the script template can call with the current iteration
        bash variable to produce the output path.
        """
        param_gen = self._filter_path_parameters(**kwargs)

        def _gen_path(iteration_variable):
            param_gen["iteration"] = [iteration_variable]
            params = [*product(*param_gen.values())]
            # For this to make sense we should only have a single combination here
            if len(params) > 1:
                self.logger.warning(
                    "Suspect script template output path: the bash template "
                    "should only vary along the 'iteration' axis. "
                    "Found combinations %s, arbitrarily taking the first.", params)
            param_args = dict(zip(self._path_parameters.keys(), params[0]))
            path_template = Path(self._path_template.format(**param_args))
            if path_template.is_absolute():
                path_template = path_template.relative_to("/")
            return path_template

        return _gen_path


class ValueTarget(Target):
    """
    Target with an associated in-memory object.
    """
    def __init__(self, task: Task, output_id: str, validator: Callable[[any], any] | None = None, **kwargs):
        self._value = None
        self._validator = validator
        # Borg state initialization here
        super().__init__(task, output_id, **kwargs)

    def assign(self, value: any):
        if self._validator:
            value = self._validator(value)
        self._value = value

    def get(self) -> any:
        if self._validator:
            return self._validator(self._value)
        return self._value


class DataFrameTarget(ValueTarget):
    """
    Target wrapping an output dataframe from a task.
    """
    def __init__(self,
                 task: Task,
                 model: "BaseDataModel | DerivedSchemaBuilder | None",
                 output_id: str | None = None,
                 **kwargs):
        if model:
            schema = model.to_schema(task.session)
            validator = lambda df: schema.validate(df)
        else:
            validator = lambda df: df.copy()
        # Original model for checking
        self._model = model

        if output_id is None:
            output_id = model.get_name() if model is not None else "frame"
        # Borg state initialization occurs here
        super().__init__(task, output_id, validator=validator, **kwargs)

        # Sanity check to verify that the model is consistent
        if model and self._model != model:
            task.logger.error("DataFrameTarget ID collision detected for %s, output_id=%s", self.borg_state_id,
                              output_id)
            raise RuntimeError("DataFrameTarget Target ID collision")


class HTMLTemplateTarget(Target):
    """
    A target that identifies an HTML template that renders analysis data.

    Note that the templates are located in pycheribenchplot/templates.
    """
    env = Environment(loader=PackageLoader("pycheribenchplot", "templates"), autoescape=select_autoescape())

    def __init__(self, task: Task, template: str, output_id: str | None = None):
        self._template = template
        if output_id is None:
            output_id = template.split(".")[0]
        super().__init__(task, output_id=output_id, ext="html")

    def render(self, **kwargs):
        try:
            tmpl = HTMLTemplateTarget.env.get_template(self._template)
        except TemplateNotFound:
            self._task.logger.error("Can not find file template %s, target setup is wrong", self._template)
            raise RuntimeError("Target error")
        with open(self.single_path(), "w") as fd:
            fd.write(tmpl.render(**kwargs))


class SQLTarget(Target):
    """
    Target that is associated with an sqlite database.

    This uses sqlalchemy to manage an engine connection.
    Queries can be executed by obtaining an SQL session as a
    context manager.
    """
    def __init__(self, task: Task, output_id: str, loader: Callable[[Self], Task] | None = None):
        # Borg state initialization here
        super().__init__(task, output_id, ext="sqlite", loader=loader)

    @cached_property
    def sql_engine(self):
        db_path = self.single_path()
        db_path.parent.mkdir(exist_ok=True)
        assert db_path.is_absolute(), "The path must always be absolute here"
        return sqla.create_engine(f"sqlite+pysqlite:///{db_path}")

    def sql_session(self) -> SqlSession:
        return SqlSession(self.sql_engine)


class DataFrameLoadTaskMixin:
    """
    Polars version of the dataframe load task.
    """
    def _load_one_csv(self, path: Path, **kwargs) -> pl.DataFrame:
        """
        Load a CSV file into a dataframe.

        :param path: The file path.
        :param **kwargs: Additional arguments to polars read_csv().
        :return: The resulting dataframe.
        """
        return pl.read_csv(path, **kwargs)

    def _load_one_json(self, path: Path, **kwargs) -> pl.DataFrame:
        """
        Load a CSV file into a dataframe

        :param path: The file path.
        :param **kwargs: Additional arguments to polars read_json().
        :return: The resulting dataframe.
        """
        return pl.read_json(path, **kwargs)

    def _load_one(self, path: Path) -> pl.DataFrame:
        """
        Load data from a given path, the format is inferred from the file extension.

        :param path: The target path.
        :return: A dataframe containing the data.
        """
        if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
            df = self._load_one_csv(path)
        elif path.suffix == ".json":
            df = self._load_one_json(path)
        else:
            self.logger.error(
                "Can not determine how to load %s, add extesion or override "
                f"{self.__class__.__name__}._load_one()", path)
            raise RuntimeError("Can not infer file type from extension")
        return df

    def _parameter_index_columns(self) -> list[str]:
        """
        Produce a list of benchmark parameterization keys.

        This is only relevant if this task is loading data for a benchmark, if the
        task is loading session-wide data, do not generate the parameterization keys.

        :return: A list of key names for the index levels
        """
        if self.is_dataset_task() and self.benchmark.config.parameters:
            return list(self.benchmark.config.parameters.keys())
        return []

    def _prepare_standard_columns(self, target: Target, df: pl.DataFrame, iteration: int | None) -> pl.DataFrame:
        """
        If the target gives us a model, automatically create the standard set of index
        levels if they are part of the model.

        :param target: The file target we are loading files from.
        :param df: Input dataframe to transform.
        :return: A new dataframe with the additional index levels.
        """
        if self.is_dataset_task():
            # Generate benchmark identifier columns
            new_cols = {}
            if "dataset_id" not in df.columns:
                new_cols["dataset_id"] = pl.lit(self.benchmark.uuid)
            if "dataset_gid" not in df.columns:
                new_cols["dataset_gid"] = pl.lit(self.benchmark.g_uuid)
            if "iteration" not in df.columns:
                if isinstance(target, BenchmarkIterationTarget):
                    new_cols["iteration"] = pl.lit(iteration)
                else:
                    new_cols["iteration"] = pl.lit(-1)

            # Generate parameterization columns
            for name, value in self.benchmark.config.parameters.items():
                if name in df.columns:
                    self.logger.error("Parameterization key '%s' is already in the dataframe", name)
                    raise RuntimeError("Invalid configuration")
                new_cols[name] = pl.lit(value)
            df = df.with_columns(**new_cols)
        return df

    def _load_from(self, target: Target) -> pl.DataFrame:
        """
        Load all the files defined by a target and merge the resulting dataframes

        :param target: The file target we are loading files from.
        :param schema: The target schema that describes the output dataframe
        :return: The dataframe containing the union of the data.
        """
        df_set = []
        for i, path in enumerate(target.paths()):
            self.logger.debug("Loading data[i=%d] from %s", i, path)
            if not path.exists():
                self.logger.error("Can not load %s, does not exist", path)
                raise FileNotFoundError(f"{path} does not exist")
            df = self._load_one(path)
            df = self._prepare_standard_columns(target, df, i)
            df_set.append(df)

        if not df_set:
            # Bail because empty data does not make much sense.
            # We could support empty data but there is no use case for it now.
            self.logger.error("No data has been loaded for %s", self)
            raise ValueError("Loader did not find any data")
        acc = df_set[0]
        for df in df_set[1:]:
            acc.vstack(df, in_place=True)
        return acc.rechunk()


class DataFrameLoadTask(DatasetTask, DataFrameLoadTaskMixin):
    """
    Internal task to load target data.

    This is the default loader task that loads data from file targets.
    """
    task_namespace = "internal"
    task_name = "pl-target-load"

    def __init__(self, target: Target):
        self.target = target
        if target.task.is_session_task():
            raise TypeError("Use DataFrameSessionLoadTask for session-wide targets")

        # Borg state initialization occurs here
        super().__init__(target.task.benchmark)

    @property
    def task_id(self):
        return super().task_id + "-for-" + self.target.borg_state_id

    def run(self):
        self.df.assign(self._load_from(self.target))

    @output
    def df(self):
        return ValueTarget(self, output_id=f"loaded-df-for-{super().task_id}")


class DataFrameSessionLoadTask(SessionTask, DataFrameLoadTaskMixin):
    """
    Internal task to load session-wide target data.
    """
    task_namespace = "internal"
    task_name = "target-session-load"

    def __init__(self, target: Target):
        self.target = target
        if target.task.is_dataset_task():
            raise TypeError("Use DataFrameLoadTask for per-dataset targets")

        # Borg state initialization occurs here
        super().__init__(target.task.session)

    @property
    def task_id(self):
        return super().task_id + "-for-" + self.target.borg_state_id

    def run(self):
        self.df.assign(self._load_from(self.target))

    @output
    def df(self):
        return ValueTarget(self, output_id=f"loaded-df-for-{super().task_id}")


# Just alias the old names
PLDataFrameSessionLoadTask = DataFrameSessionLoadTask
PLDataFrameLoadTask = DataFrameLoadTask
