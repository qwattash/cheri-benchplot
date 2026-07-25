from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl

from ..core.artefact import DataFrameLoadTask, RemoteBenchmarkIterationTarget
from ..core.config import ConfigPath, config_field
from ..core.task import output
from ..generic.timing import TimingExecTask, TimingConfig


class RedisTestType(Enum):
    """Enumeration of available Redis benchmark test types."""

    SET = "set"
    GET = "get"
    LPUSH = "lpush"
    LPOP = "lpop"
    HSET = "hset"
    HGET = "hget"
    MSET = "mset"
    MGET = "mget"


@dataclass
class RedisConfig(TimingConfig):
    """
    The redis-benchmark benchmark parameters.
    """

    redis_benchmark_path: ConfigPath | None = config_field(
        None, desc="Path to redis-benchmark executable in the remote host"
    )
    host: str = config_field(
        "localhost",
        desc="Hostname of the Redis server",
    )
    port: int = config_field(
        6379,
        desc="Port of the Redis server",
    )
    threads: int = config_field(
        1,
        desc="Number of parallel clients",
    )
    connections: int = config_field(
        50,
        desc="Number of parallel connections",
    )
    requests: int = config_field(
        100000,
        desc="Number of total requests to perform",
    )
    pipeline: int = config_field(
        1,
        desc="Pipeline requests",
    )
    test: RedisTestType | None = config_field(
        None,
        desc="Test to run (null = use redis-benchmark default)",
    )


class LoadRedisStats(DataFrameLoadTask):
    """
    Loader for redis-benchmark stats data that produces a standard polars dataframe.
    """

    task_namespace = "redis"
    task_name = "ingest-stats"

    @property
    def data_columns(self) -> list[str]:
        return ["requests_per_second", "latency_ms", "errors", "connections"]

    def _load_one(self, path: Path) -> pl.DataFrame:
        """
        Load data for a benchmark run from the given target file.
        """
        df = pl.read_csv(path)
        if df["errors"].any():
            self.logger.warning("Detected errors during benchmark execution")
        return df


class RedisExecTask(TimingExecTask):
    """
    Generate the redis-benchmark scripts.
    """

    task_namespace = "redis"
    task_name = "exec"
    task_config_class = RedisConfig
    public = True

    @output
    def stats(self):
        return RemoteBenchmarkIterationTarget(
            self, "stats", ext="csv", loader=LoadRedisStats
        )

    def run(self):
        super().run()
        self.script.set_template("redis.sh.jinja")
        self.script.extend_context(
            {
                "redis_config": self.config,
                "redis_gen_output_path": self.stats.shell_path_builder(),
            }
        )
        self.script.register_global("RedisTestType", RedisTestType)
