from typing import List, Optional

from sqlalchemy import ForeignKey, UniqueConstraint, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class UInt64(TypeDecorator):
    """
    We encode uint64 as a string, because it can't fit sqlite INT type.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return int(value)
        return None


class SqlBase(DeclarativeBase):
    pass


class Binary(SqlBase):
    """
    File from which the metadata is extracted
    """

    __tablename__ = "binary"

    id: Mapped[int] = mapped_column(primary_key=True)
    file: Mapped[str] = mapped_column(nullable=False)

    layouts: Mapped[List["TypeLayout"]] = relationship(
        back_populates="binary",
        foreign_keys="TypeLayout.binary_id",
        cascade="all, delete-orphan",
    )


class TypeLayout(SqlBase):
    """
    Table containing flattened layouts for each structure declaration.
    """

    __tablename__ = "type_layout"
    __table_args__ = (UniqueConstraint("name", "file", "line", "size"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    file: Mapped[str] = mapped_column(nullable=False)
    line: Mapped[int] = mapped_column(nullable=False)
    name: Mapped[str]
    size: Mapped[int] = mapped_column(nullable=False)
    is_union: Mapped[bool] = mapped_column(nullable=False)
    has_vla: Mapped[bool] = mapped_column(nullable=False)

    total_padding: Mapped[int] = mapped_column(nullable=False)
    tail_padding: Mapped[int] = mapped_column(nullable=False)
    holes: Mapped[int] = mapped_column(nullable=False)
    nested_padding: Mapped[int] = mapped_column(nullable=False)
    nested_holes: Mapped[int] = mapped_column(nullable=False)
    has_extra_padding: Mapped[bool] = mapped_column(nullable=False)

    binary_id: Mapped[int] = mapped_column(ForeignKey("binary.id"))
    binary: Mapped["Binary"] = relationship(
        foreign_keys=binary_id, back_populates="layouts"
    )

    members: Mapped[List["LayoutMember"]] = relationship(
        back_populates="owner_entry",
        foreign_keys="LayoutMember.owner",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"TypeLayout[{self.id}] @ {self.file}:{self.line} {self.name}"


class LayoutMember(SqlBase):
    """
    Table containing structure members for each structure.
    """

    __tablename__ = "layout_member"
    __table_args__ = (UniqueConstraint("owner", "name", "byte_offset", "bit_offset"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[int] = mapped_column(ForeignKey("type_layout.id"))
    name: Mapped[str] = mapped_column(nullable=False)
    type_name: Mapped[str] = mapped_column(nullable=False)
    byte_size: Mapped[int] = mapped_column(nullable=False)
    bit_size: Mapped[int] = mapped_column(nullable=False)
    byte_offset: Mapped[int] = mapped_column(nullable=False)
    bit_offset: Mapped[int] = mapped_column(nullable=False)
    alignment: Mapped[int] = mapped_column(nullable=False)
    array_items: Mapped[Optional[int]]
    base: Mapped[Optional[int]] = mapped_column(UInt64(), nullable=True)
    top: Mapped[Optional[int]] = mapped_column(UInt64(), nullable=True)
    required_precision: Mapped[Optional[int]]
    max_vla_size: Mapped[Optional[int]] = mapped_column(UInt64(), nullable=True)
    is_pointer: Mapped[bool] = mapped_column(nullable=False, default=False)
    is_function: Mapped[bool] = mapped_column(nullable=False, default=False)
    is_anon: Mapped[bool] = mapped_column(nullable=False, default=False)
    is_union: Mapped[bool] = mapped_column(nullable=False, default=False)
    is_imprecise: Mapped[bool] = mapped_column(nullable=False, default=False)

    owner_entry: Mapped["TypeLayout"] = relationship(
        foreign_keys=owner, back_populates="members"
    )

    def __repr__(self):
        return (
            f"LayoutMember[{self.id}] ({self.byte_offset}/{self.bit_offset}) "
            f"{self.type_name} {self.owner.name}.{self.name}"
        )
