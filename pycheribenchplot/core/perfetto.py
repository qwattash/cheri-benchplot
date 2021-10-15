import logging
from abc import abstractmethod
from pathlib import Path

import pandas as pd
from perfetto.trace_processor import TraceProcessor

from .dataset import DataSetContainer


class _TraceProcessorCache:
    """
    Cache trace processor instances for later access so that we only load traces once.
    """
    instance = None

    @classmethod
    def get_instance(cls, manager_config: "BenchmarkManagerConfig"):
        if cls.instance is None:
            cls.instance = _TraceProcessorCache(manager_config)
        return cls.instance

    def __init__(self, manager_config: "BenchmarkManagerConfig"):
        self._manager_config = manager_config
        self._instances = {}
        self.logger = logging.getLogger("perfetto-trace-processor-cache")

    def _trace_processor_path(self):
        return self._manager_config.perfetto_path / "trace_processor_shell"

    def get_trace_processor(self, trace_path: Path) -> TraceProcessor:
        if trace_path in self._instances:
            return self._instances[trace_path]
        self.logger.debug("New trace processor for %s", trace_path)
        processor = TraceProcessor(bin_path=self._trace_processor_path(), file_path=trace_path)
        self._instances[trace_path] = processor
        return processor

    def __del__(self):
        for tp in self._instances.values():
            tp.close()


class PerfettoDataSetContainer(DataSetContainer):
    # Map columns in the SQL expression
    key_to_column_map = {}

    def __init__(self, benchmark: "BenchmarkBase", dset_key: str):
        super().__init__(benchmark, dset_key)
        self._tp_cache = _TraceProcessorCache.get_instance(benchmark.manager_config)

    def _integrity_check(self, tp: TraceProcessor):
        result = tp.query("SELECT * FROM stats WHERE name = 'traced_buf_trace_writer_packet_loss'")
        rows = list(result)
        assert len(rows) == 1, "Query for stats.traced_buf_trace_writer_packet_loss failed"
        if rows[0].value != 0:
            self.logger.error("!!!Perfetto packet loss detected!!!")

    def _get_sql_table(self):
        """Get the table to select from"""
        return "slice"

    def _get_sql_where(self):
        """Build the SQL where clause to filter events"""
        return ""

    @abstractmethod
    def _get_sql_expr(self):
        """
        Return the SQL expression to get the data for the dataset
        """
        ...
        # fields = self.raw_fields()
        # table = self._get_sql_table()
        # where = self._get_sql_where()
        # return f"select {fields} from {table} {where}"

    def _map_row_to_df(self, df_dict, row):
        for col, value in row.__dict__.items():
            df_dict[col].append(value)

    @abstractmethod
    def _build_df(self, result: pd.DataFrame):
        ...

    def _extract_events(self, tp: TraceProcessor):
        self._integrity_check(tp)
        query_str = self._get_sql_expr()
        result = tp.query(query_str)
        # XXX-AM: This is unreasonably slow, build the dataframe manually for now
        # df = result.as_pandas_dataframe()
        query_df = pd.DataFrame.from_records(map(lambda row: row.__dict__, result))
        df = self._build_df(query_df)
        # Append dataframe to dataset
        self.df = pd.concat([self.df, df])

    def load(self, path: Path):
        processor = self._tp_cache.get_trace_processor(path)
        self._extract_events(processor)
