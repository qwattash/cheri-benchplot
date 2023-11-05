from collections import namedtuple

import pandas as pd
from pandera import Field, SchemaModel, check, dataframe_check
from pandera.typing import Index, Series

from ..core.dwarf import DWARFStructLayoutModel
from ..core.model import DataModel, GlobalModel

# V2
from enum import IntFlag
from typing import List, Optional

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy_utils import ChoiceType


class ImpreciseSubobjectInfoModel(DataModel):
    """
    Concise information about structures with imprecise sub-object bounds.

    This is used to generate more detailed information about the members layout.
    Note that the index mirrors :class:`DWARFStructLayoutModel`.
    """
    file: Index[str]
    line: Index[int]
    base_name: Index[str]
    member_name: Index[str]
    member_offset: Index[int]

    member_size: Series[int]
    member_aligned_base: Series[int]
    member_aligned_top: Series[int]


#: Helper to keep records for :class:`ImpreciseSubobjectInfoModel` consistent
ImpreciseSubobjectInfoModelRecord = namedtuple("ImpreciseSubobjectInfoModelRecord", [
    "file", "line", "base_name", "member_offset", "member_name", "member_size", "member_aligned_base",
    "member_aligned_top"
])


class ImpreciseSubobjectLayoutModel(DWARFStructLayoutModel):
    """
    Augment the structure layout with capability aliasing information
    """
    #: Identifies a struct member as being the owner of an alias group.
    alias_group_id: Series[pd.Int64Dtype] = Field(nullable=True)
    #: Identifies the aligned base of a member that owns an alias group.
    alias_aligned_base: Series[pd.Int64Dtype] = Field(nullable=True)
    #: Idnetifies the aligned top of a member that owns an alias group.
    alias_aligned_top: Series[pd.Int64Dtype] = Field(nullable=True)
    #: Describe groups of fields that are aliased with the same imprecise capability.
    #: Alias groups are incrementally numbered and are unique for each type_id,
    #: i.e. for each root structure type.
    alias_groups: Series[object] = Field(nullable=True)


class SqlBase(DeclarativeBase):
    pass


class StructTypeFlags(IntFlag):
    Unset = 0
    IsAnonymous = 1
    IsStruct = 2
    IsUnion = 4
    IsClass = 8


class StructType(SqlBase):
    """
    Table containing unique structures in the input data set.
    """
    __tablename__ = "struct_type"

    id: Mapped[int] = mapped_column(primary_key=True)
    file: Mapped[str] = mapped_column(nullable=False)
    line: Mapped[int] = mapped_column(nullable=False)
    name: Mapped[str]
    c_name: Mapped[str]
    size: Mapped[int] = mapped_column(nullable=False)
    flags: Mapped[ChoiceType] = mapped_column(
        ChoiceType(StructTypeFlags, impl=Integer()),
        nullable=False, default=0)

    members: Mapped[List["StructMember"]] = relationship(
        back_populates="owner_struct",
        foreign_keys="StructMember.owner",
        cascade="all, delete-orphan")
    referenced: Mapped[List["StructMember"]] = relationship(
        back_populates="nested_struct",
        foreign_keys="StructMember.nested")

    def __repr__(self):
        return f"StructType[{self.id}] @ {self.file}:{self.line} {self.c_name}"


class StructMemberFlags(IntFlag):
    Unset = 0,
    IsPtr = 1,
    IsFnPtr = 2,
    IsArray = 1 << 2,
    IsDecl = 1 << 3,
    IsStruct = 1 << 4,
    IsUnion = 1 << 5,
    IsClass = 1 << 6


class StructMember(SqlBase):
    """
    Table containing structure members for each structure.
    """
    __tablename__ = "struct_member"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[int] = mapped_column(ForeignKey("struct_type.id"))
    nested: Mapped[Optional[int]] = mapped_column(ForeignKey("struct_type.id"))
    name: Mapped[str] = mapped_column(nullable=False)
    type_name: Mapped[str] = mapped_column(nullable=False)
    line: Mapped[int] = mapped_column(nullable=False)
    size: Mapped[int] = mapped_column(nullable=False)
    bit_size: Mapped[Optional[int]]
    offset: Mapped[int] = mapped_column(nullable=False)
    bit_offset: Mapped[Optional[int]]
    flags: Mapped[ChoiceType] = mapped_column(
        ChoiceType(StructMemberFlags, impl=Integer()),
        nullable=False, default=0)
    array_items: Mapped[Optional[int]]

    owner_struct: Mapped["StructType"] = relationship(
        foreign_keys=owner, back_populates="members")
    nested_struct: Mapped["StructType"] = relationship(
        foreign_keys=nested, back_populates="referenced")

    def __repr__(self):
        return (f"StructMember[{self.id}] ({self.offset}/{self.bit_offset}) {self.type_name}"
                f"{self.owner.name}.{self.name}")
