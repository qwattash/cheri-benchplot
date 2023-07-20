import pandas as pd
from pandera import Field
from pandera.typing import Index, Series

from ..core.model import DataModel


class C18NDomainTransitionTraceModel(DataModel):
    trace_source: Index[str]
    sequence: Index[int]
    pid: Index[int]
    binary: Index[str]
    op: Series[str] = Field(isin=["enter", "leave"])
    compartment: Series[str]
    symbol: Series[str]
    address: Series[pd.Int64Dtype] = Field(nullable=True)
