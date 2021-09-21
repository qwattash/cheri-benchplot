from abc import abstractmethod
from pathlib import Path

import pandas as pd
from perfetto.trace_processor import TraceProcessor

from .dataset import DataSetContainer


class PerfettoDataSetContainer(DataSetContainer):
    # Map columns in the SQL expression
    key_to_column_map = {}

    def __init__(self, benchmark: "BenchmarkBase", dset_key: str):
        super().__init__(benchmark, dset_key)

    def _trace_processor_path(self):
        return self.benchmark.manager_config.perfetto_path / "trace_processor_shell"

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
        query_str = self._get_sql_expr()
        result = tp.query(query_str)
        # XXX-AM: This is unreasonably slow, build the dataframe manually for now
        # df = result.as_pandas_dataframe()
        query_df = pd.DataFrame.from_records(map(lambda row: row.__dict__, result))
        df = self._build_df(query_df)
        # Append dataframe to dataset
        self.df = pd.concat([self.df, df])

    def load(self, path: Path):
        processor = TraceProcessor(bin_path=self._trace_processor_path(), file_path=path)
        self._extract_events(processor)
