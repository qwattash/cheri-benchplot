from enum import Enum

import numpy as np
import pandas as pd
from pandera import Field, check
from pandera.dtypes import immutable
from pandera.engines import numpy_engine, pandas_engine
from pandera.typing import DateTime, Index, Series

from pycheribenchplot.core.model import GlobalModel


class Marker(Enum):
    YES = "X"
    TEMPORAL = "T"
    UNKNOWN = "?"
    EMPTY = "E"

    @classmethod
    def from_input(cls, value):
        if (np.issubdtype(type(value), np.number) and np.isnan(value)) or not value:
            return cls.EMPTY
        # Currently we interpret 'N' as EMPTY
        if value == "N":
            return cls.EMPTY
        try:
            return cls(value)
        except ValueError:
            # Currently we interpret free-text as YES
            return cls.YES

    @classmethod
    def verified(cls):
        return [cls.YES, cls.TEMPORAL, cls.EMPTY]

    @classmethod
    def all(cls):
        return list(cls)

    def __bool__(self):
        return self == self.YES

    def __str__(self):
        return self.value


@pandas_engine.Engine.register_dtype
@immutable
class MarkerType(numpy_engine.Object):
    """
    Pandera data type for Marker objects
    """
    def check(self, pandera_dtype, data_container):
        return data_container.map(lambda v: isinstance(v, Marker))

    def coerce(self, series: pd.Series) -> pd.Series:
        return series.map(lambda v: v if isinstance(v, Marker) else Marker.from_input(v))


class CheriBSDAdvisories(GlobalModel):
    date: Index[DateTime]
    advisory: Index[str]
    patch_commits: Series[str]
    not_memory_safety_bug: Series[MarkerType] = Field(isin=Marker.verified())
    cheri_ptr_provenance: Series[MarkerType] = Field(isin=Marker.all())
    cheri_ptr_integrity: Series[MarkerType] = Field(isin=Marker.all())
    cheri_bounds_checking: Series[MarkerType] = Field(isin=Marker.all())
    cheri_ptr_perms: Series[MarkerType] = Field(isin=Marker.verified())
    cheri_temporal: Series[MarkerType] = Field(isin=Marker.all())
    cheri_subobject: Series[MarkerType] = Field(isin=Marker.verified())
    cheri_does_not_help: Series[MarkerType] = Field(isin=Marker.all())
    can_unpatch: Series[bool]
    local_remote: Series[str] = Field(isin=["local", "remote"])
    overflow: Series[bool]
    dos: Series[bool]
    exec: Series[bool]
    disclosure: Series[bool]
    corrupt_memory: Series[bool]
    arch_specific: Series[bool]
    uaf_uma_kmalloc: Series[bool]
    race_or_tocttou: Series[bool]
    priv_escalation: Series[bool]
    null_ptr_dereference: Series[bool]
    affects_network_stack: Series[bool]
    notes: Series[str]


class History(GlobalModel):
    """
    The history dataframe maintains interesting events that we may want
    to show in plots
    """
    date: Index[DateTime] = Field(check_name=True)
    occurrence: Series[str]


class CheriBSDUnmitigated(GlobalModel):
    """
    Table of advisories that are not mitigated by CHERI.
    """
    date: Index[DateTime]
    unmitigated_advisory: Index[str]
    reason_padding_initialization: Series[MarkerType] = Field(isin=Marker.verified())
    reason_stack_use_after_free: Series[MarkerType] = Field(isin=Marker.verified())
    reason_direct_vm_subsystem_access: Series[MarkerType] = Field(isin=Marker.verified())
    reason_other: Series[MarkerType] = Field(isin=Marker.verified())
    can_padding_leak_a_capability_y_n: Series[MarkerType] = Field(isin=Marker.verified())
