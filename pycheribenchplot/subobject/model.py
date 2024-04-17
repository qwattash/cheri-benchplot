from collections import namedtuple
# V2
from enum import IntFlag
from typing import List, Optional

import pandas as pd
from pandera import Field, SchemaModel, check, dataframe_check
from pandera.typing import Index, Series
from sqlalchemy import ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy_utils import ChoiceType

from ..core.dwarf import DWARFStructLayoutModel
from ..core.model import DataModel, GlobalModel


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
    __table_args__ = (UniqueConstraint("name", "file", "line"), )

    id: Mapped[int] = mapped_column(primary_key=True)
    file: Mapped[str] = mapped_column(nullable=False)
    line: Mapped[int] = mapped_column(nullable=False)
    name: Mapped[str]
    size: Mapped[int] = mapped_column(nullable=False)
    flags: Mapped[ChoiceType] = mapped_column(ChoiceType(StructTypeFlags, impl=Integer()), nullable=False, default=0)
    has_imprecise: Mapped[bool] = mapped_column(default=False)

    members: Mapped[List["StructMember"]] = relationship(back_populates="owner_entry",
                                                         foreign_keys="StructMember.owner",
                                                         cascade="all, delete-orphan")
    referenced: Mapped[List["StructMember"]] = relationship(back_populates="nested_entry",
                                                            foreign_keys="StructMember.nested")

    def __repr__(self):
        return f"StructType[{self.id}] @ {self.file}:{self.line} {self.name}"


class StructMemberFlags(IntFlag):
    Unset = 0,
    IsPtr = 1,
    IsFnPtr = 2,
    IsArray = 1 << 2,
    IsAnon = 1 << 3,
    IsStruct = 1 << 4,
    IsUnion = 1 << 5,
    IsClass = 1 << 6


class StructMember(SqlBase):
    """
    Table containing structure members for each structure.
    """
    __tablename__ = "struct_member"
    __table_args__ = (UniqueConstraint("owner", "name", "offset"), )

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
    flags: Mapped[ChoiceType] = mapped_column(ChoiceType(StructMemberFlags, impl=Integer()), nullable=False, default=0)
    array_items: Mapped[Optional[int]]

    owner_entry: Mapped["StructType"] = relationship(foreign_keys=owner, back_populates="members")
    nested_entry: Mapped["StructType"] = relationship(foreign_keys=nested, back_populates="referenced")

    def __repr__(self):
        return (f"StructMember[{self.id}] ({self.offset}/{self.bit_offset}) {self.type_name}"
                f"{self.owner.name}.{self.name}")


class SubobjectAlias(SqlBase):
    """
    Table to link each imprecise member_bounds entry to the set of
    member_bounds entries that are aliased by the imprecise capability
    """
    __tablename__ = "subobject_alias"

    id: Mapped[int] = mapped_column(primary_key=True)
    subobj: Mapped[int] = mapped_column(ForeignKey("member_bounds.id"), primary_key=True)
    alias: Mapped[int] = mapped_column(ForeignKey("member_bounds.id"), primary_key=True)

    subobj_entry: Mapped["MemberBounds"] = relationship(foreign_keys=subobj)
    alias_entry: Mapped["MemberBounds"] = relationship(foreign_keys=alias)

    def __repr__(self):
        return (f"SubobjectAlias[{self.subobj_entry.name} ->"
                f" {self.alias_entry.name}]")


class MemberBounds(SqlBase):
    """
    Table containing the flattened layout for each structure.
    """
    __tablename__ = "member_bounds"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    owner: Mapped[int] = mapped_column(ForeignKey("struct_type.id"))
    member: Mapped[int] = mapped_column(ForeignKey("struct_member.id"))
    mindex: Mapped[int] = mapped_column(nullable=False)
    offset: Mapped[int] = mapped_column(nullable=False)
    base: Mapped[int]
    top: Mapped[int]
    is_imprecise: Mapped[bool] = mapped_column(default=False)
    precision: Mapped[int]

    owner_entry: Mapped["StructType"] = relationship(foreign_keys=owner)
    member_entry: Mapped["StructMember"] = relationship(foreign_keys=member)
    aliasing_with: Mapped[List["MemberBounds"]] = relationship(secondary="subobject_alias",
                                                               primaryjoin=(id == SubobjectAlias.subobj),
                                                               secondaryjoin=(SubobjectAlias.alias == id),
                                                               order_by=offset,
                                                               viewonly=True)
    aliased_by: Mapped[List["MemberBounds"]] = relationship(secondary="subobject_alias",
                                                            primaryjoin=(id == SubobjectAlias.alias),
                                                            secondaryjoin=(SubobjectAlias.subobj == id),
                                                            order_by=offset,
                                                            viewonly=True)

    def __repr__(self):
        return (f"MemberBounds[{self.name}] 0x{self.offset:x} "
                f"[0x{self.base:x}, 0x{self.top:x}]")
