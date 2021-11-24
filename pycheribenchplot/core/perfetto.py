import logging
from abc import abstractmethod
from pathlib import Path

import pandas as pd
from perfetto.trace_processor import TraceProcessor

from .util import new_logger, timing
from .dataset import DataSetContainer


class _TraceProcessorCache:
    """
    Cache trace processor instances for later access so that we only load traces once.
    """
    instance = None

    @classmethod
    def get_instance(cls, manager: "BenchmarkManager"):
        if cls.instance is None:
            cls.instance = _TraceProcessorCache(manager)
        return cls.instance

    def __init__(self, manager: "BenchmarkManager"):
        self._manager = manager
        self._instances = {}
        self.logger = new_logger("perfetto-trace-processor-cache")
        self._manager.cleanup_callbacks.append(lambda: self._shutdown())

    def _trace_processor_path(self):
        return self._manager.config.perfetto_path / "trace_processor_shell"

    def get_trace_processor(self, trace_path: Path) -> TraceProcessor:
        if trace_path in self._instances:
            return self._instances[trace_path]
        self.logger.debug("New trace processor for %s", trace_path)
        processor = TraceProcessor(bin_path=self._trace_processor_path(), file_path=trace_path)
        self._instances[trace_path] = processor
        return processor

    def _shutdown(self):
        for tp in self._instances.values():
            tp.close()


class PerfettoDataSetContainer(DataSetContainer):
    # Map columns in the SQL expression
    key_to_column_map = {}

    def __init__(self, benchmark, dset_key, config):
        super().__init__(benchmark, dset_key, config)
        self._tp_cache = _TraceProcessorCache.get_instance(benchmark.manager)

    def _integrity_check(self, tp: TraceProcessor):
        result = tp.query("SELECT * FROM stats WHERE name = 'traced_buf_trace_writer_packet_loss'")
        rows = list(result)
        assert len(rows) == 1, "Query for stats.traced_buf_trace_writer_packet_loss failed"
        if rows[0].value != 0:
            self.logger.error("!!!Perfetto packet loss detected!!!")

    def _extract_events(self, tp: TraceProcessor):
        self._integrity_check(tp)

    def _query_to_df(self, tp: TraceProcessor, query_str):
        # XXX-AM: This is unreasonably slow, build the dataframe manually for now
        # df = result.as_pandas_dataframe()
        with timing(query_str, logging.DEBUG, self.logger):
            result = tp.query(query_str)
            query_df = pd.DataFrame.from_records(map(lambda row: row.__dict__, result))
        return query_df

    def load(self, path: Path):
        processor = self._tp_cache.get_trace_processor(path)
        self._extract_events(processor)
