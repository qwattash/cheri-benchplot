from enum import Enum

from pandera import Field, check
from pandera.typing import DateTime, Index, Series

from pycheribenchplot.core.model import GlobalModel


class Marker(Enum):
    YES = "X"
    TEMPORAL = "T"
    UNKNOWN = "?"

    @classmethod
    def verified(cls):
        return [cls.YES.value, cls.TEMPORAL.value]

    @classmethod
    def all(cls):
        return [m.value for m in cls]


class CheriBSDAdvisories(GlobalModel):
    date: Index[DateTime]
    advisory: Index[str]
    patch_commits: Series[str]
    not_memory_safety_bug: Series[str] = Field(nullable=True, isin=Marker.verified())
    cheri_ptr_provenance: Series[str] = Field(nullable=True, isin=Marker.all())
    cheri_ptr_integrity: Series[str] = Field(nullable=True, isin=Marker.all())
    cheri_bounds_checking: Series[str] = Field(nullable=True, isin=Marker.all())
    cheri_ptr_perms: Series[str] = Field(nullable=True, isin=Marker.verified())
    cheri_temporal: Series[str] = Field(nullable=True, isin=Marker.all())
    cheri_subobject: Series[str] = Field(nullable=True, isin=Marker.verified())
    cheri_does_not_help: Series[str] = Field(nullable=True, isin=Marker.all())
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
    affects_network_stack: Series[bool]
    notes: Series[str]
