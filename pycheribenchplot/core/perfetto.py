import logging
import typing
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from perfetto.trace_processor import TraceProcessor
from pypika import Query, Table

from .dataset import DataSetContainer
from .util import new_logger, timing


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
        self.logger.debug("Shutdown perfetto instances %s", self._instances)
        for tp in self._instances.values():
            tp.close()


class PerfettoDataSetContainer(DataSetContainer):
    # Map columns in the SQL expression
    key_to_column_map = {}

    BENCHMARK_ITERATION_MARKER = 0xbeef

    t_track = Table("track")
    t_slice = Table("slice")
    t_args = Table("args")
    t_stats = Table("stats")

    def __init__(self, benchmark, dset_key, config):
        super().__init__(benchmark, dset_key, config)
        self._tp_cache = _TraceProcessorCache.get_instance(benchmark.manager)

    def _integrity_check(self, tp: TraceProcessor):
        query = Query.from_(self.t_stats).select(
            self.t_stats.star).where(self.t_stats.name == "traced_buf_trace_writer_packet_loss")
        result = tp.query(query.get_sql(quote_char=None))
        rows = list(result)
        assert len(rows) == 1, "Query for stats.traced_buf_trace_writer_packet_loss failed"
        if rows[0].value != 0:
            self.logger.error("!!!Perfetto packet loss detected!!!")

    def _query_slice_args(self) -> Query:
        """Shorthand to get a joined query on slice + args tables"""
        return Query.from_(self.t_slice).join(self.t_args).on(self.t_args.arg_set_id == self.t_slice.arg_set_id)

    def _query_slice_ts(self, ts_start, ts_end) -> Query:
        query = self._query_slice_args()
        query = query.where(self.t_slice.ts >= ts_start)
        if np.isfinite(ts_end):
            query = query.where(self.t_slice.ts < ts_end)
        return query

    def _query_to_df(self, tp: TraceProcessor, query_str: str):
        # XXX-AM: This is unreasonably slow, build the dataframe manually for now
        # df = result.as_pandas_dataframe()
        assert query_str is not None, "No query string?"
        assert len(query_str), "Empty query string?"
        with timing(query_str, logging.DEBUG, self.logger):
            result = tp.query(query_str)
            query_df = pd.DataFrame.from_records(map(lambda row: row.__dict__, result))
        return query_df

    def _get_trace_processor(self, path: Path) -> TraceProcessor:
        tp = self._tp_cache.get_trace_processor(path)
        self._integrity_check(tp)
        return tp

    def _extract_iteration_markers(self, tp: TraceProcessor):
        """
        Return a list of time stamps representing the marker for the start
        of a new benchmark iteration.
        These are generated using the QEMU_EVENT_MARKER() cheribsd macro, which
        internally uses the qemu LOG_EVENT_MARKER event type.
        """
        marker_constant = self.BENCHMARK_ITERATION_MARKER
        query = self._query_slice_args()
        query = query.select(self.t_slice.ts)
        query = query.where((self.t_slice.category == "marker") & (self.t_slice.name == "guest")
                            & (self.t_args.flat_key == "qemu.marker")
                            & (self.t_args.int_value == self.BENCHMARK_ITERATION_MARKER))
        result_df = self._query_to_df(tp, query.get_sql(quote_char=None))
        if len(result_df) == 0:
            self.logger.warning("No benchmark iteration markers in trace, assume infinite interval")
            return [[0, np.inf]]
        intervals = result_df[["ts"]]  # ensure we are copying a DataFrame
        intervals["ts_end"] = result_df["ts"].shift(-1, fill_value=np.inf)
        interval_array = intervals.to_numpy()
        self.logger.debug("Detected benchmark iterations in trace: %s", interval_array)
        return interval_array
