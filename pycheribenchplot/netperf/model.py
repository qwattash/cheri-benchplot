from pandera import Field
from pandera.typing import Index, Series

from pycheribenchplot.core.analysis import StatsField
from pycheribenchplot.core.model import DataModel, ParamGroupDataModel


class NetperfInputModel(DataModel):
    """
    Input data from UDP and TCP RR benchmarks
    """
    request_size_bytes: Index[int] = Field(alias="Request Size Bytes")
    response_size_bytes: Index[int] = Field(alias="Response Size Bytes")
    # Data
    time: Series[float] = Field(alias="Elapsed Time (sec)")
    throughput: Series[float] = Field(alias="Throughput")


class NetperfStatsModel(ParamGroupDataModel):
    """
    Statistics for the netperf RR benchmarks.
    """
    request_size_bytes: Index[int] = Field(alias="Request Size Bytes")
    response_size_bytes: Index[int] = Field(alias="Response Size Bytes")
    # Data
    time: Series[float] = StatsField(r"Elapsed Time \(sec\)", nullable=True)
    throughput: Series[float] = StatsField("Throughput", nullable=True)
