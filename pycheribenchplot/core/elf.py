import itertools as it
import typing
from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from enum import Flag, IntFlag, auto
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from elftools.elf.elffile import ELFFile
from sortedcontainers import SortedDict

from .util import new_logger


@dataclass
class SymInfo:
    name: str
    filepath: Path
    size: int
    addr: int


class SymbolizerMapping:
    """
    Per-mapping set of symbols.
    """
    def __init__(self, logger, mapbase: int, path: Path):
        self.logger = logger
        self.symbols = SortedDict()
        self.functions = SortedDict()
        with open(path, "rb") as fd:
            ef = ELFFile(fd)
            if not self._is_dynamic(ef):
                base = 0
            else:
                base = mapbase
            self.logger.debug("Load symbols for %s base=0x%x dyn=%s", path, base, self._is_dynamic(ef))
            symtab = ef.get_section_by_name(".symtab")
            for sym in symtab.iter_symbols():
                addr = base + sym["st_value"]
                syminfo = SymInfo(name=sym.name, filepath=path, addr=addr, size=sym["st_size"])
                st_info = sym["st_info"]
                if st_info.type == "STT_FUNC":
                    self.functions[addr] = syminfo
                else:
                    self.symbols[addr] = syminfo
            self.map_end = base + self._get_end_addr(ef)
        assert self.map_end > mapbase

    def _is_dynamic(self, ef):
        if ef.header.e_type == "ET_DYN":
            return True
        return False

    def _get_end_addr(self, ef):
        maxaddr = 0
        for p in ef.iter_segments():
            pend = p["p_vaddr"] + p["p_memsz"]
            if p["p_type"] == "PT_LOAD" and pend > maxaddr:
                maxaddr = pend
        return maxaddr

    def lookup_fn(self, addr: int) -> typing.Optional[SymInfo]:
        idx = self.functions.bisect(addr) - 1
        if idx < 0:
            return None
        syminfo = self.functions.values()[idx]
        # XXX Symbol size check seems unreliable for some reason?
        # if syminfo.addr + syminfo.size < addr:
        #     return None
        return syminfo

    def lookup_addr_for_symbol(self, sym: str) -> typing.Optional[SymInfo]:
        """
        Find address of a symbol given its name.
        Note that this is not optimized.
        """
        for syminfo in it.chain(self.functions.values(), self.symbols.values()):
            if syminfo.name == sym:
                return syminfo
        return None


class SymbolizerSet:
    """
    Per-AS set of mappings that have associated symbols.
    """
    def __init__(self, logger):
        self.logger = logger
        self.mappings = SortedDict()

    def add_sym_source(self, mapbase: int, path: Path):
        mapping = SymbolizerMapping(self.logger, mapbase, path)
        self.mappings[mapbase] = mapping

    def _lookup_mapping(self, addr):
        idx = self.mappings.bisect(addr) - 1
        if idx < 0:
            return None
        return self.mappings.values()[idx]

    def lookup_fn(self, addr: int):
        mapping = self._lookup_mapping(addr)
        if mapping is None or mapping.map_end < addr:
            return None
        return mapping.lookup_fn(addr)


class Symbolizer:
    """
    Resolve addresses to symbols by address-space.
    Symbol sources must be registered to an address space, this is identified
    by a custom string (generally the name of the process owning the address space).
    Shared address spaces are used to match all symbols.
    """
    def __init__(self, benchmark: "BenchmarkBase"):
        self.logger = new_logger(f"{benchmark.uuid}.symbolizer")
        self.addrspace = {}

    def _get_or_create_addrspace(self, key: str):
        if key not in self.addrspace:
            self.logger.debug("New symbolizer address space %s", key)
            self.addrspace[key] = SymbolizerSet(self.logger)
        return self.addrspace[key]

    def guess_load_address(self, path: Path, symbol: str, addr: int) -> int:
        """
        Attempt to guess the load address of the given binary given a symbol name
        and its address in the loaded image.
        """
        mapping = SymbolizerMapping(self.logger, 0, path)
        syminfo = mapping.lookup_addr_for_symbol(symbol)
        if syminfo is None:
            self.logger.warning("Can not guess load address for %s given %s=0x%x", path, symbol, addr)
            raise ValueError("Can not guess load address")
        return addr - syminfo.addr

    def register_sym_source(self, mapbase: int, as_key: str, path: Path, shared=False):
        """
        Register a symbol source with the given address space key. If the
        as_key is None, the symbols are considered to be shared.
        """
        if shared:
            # also add the symbol to the shared AS key=None
            shared_addrspace = self._get_or_create_addrspace(None)
            shared_addrspace.add_sym_source(mapbase, path)
        addrspace = self._get_or_create_addrspace(as_key)
        addrspace.add_sym_source(mapbase, path)

    def _lookup_fn_shared(self, addr):
        addrspace = self.addrspace.get(None, None)
        if addrspace is None:
            return None
        return addrspace.lookup_fn(addr)

    def lookup_fn(self, addr: int, as_key: str):
        """
        Lookup a function in the given address space
        """
        addrspace = self.addrspace.get(as_key, None)
        if addrspace is None:
            # Try the shared addrspace
            return self._lookup_fn_shared(addr)
        syminfo = addrspace.lookup_fn(addr)
        if syminfo is None:
            return self._lookup_fn_shared(addr)
        return syminfo

    def match_fn(self, addr: int, as_key: str):
        """
        Lookup for an exact match between the address and the resolved symbol
        """
        addrspace = self.addrspace.get(as_key, None)
        if addrspace is None:
            syminfo = self._lookup_fn_shared(addr)
        else:
            syminfo = addrspace.lookup_fn(addr)
            if syminfo is None:
                syminfo = self._lookup_fn_shared(addr)
        if syminfo is None or addr != syminfo.addr:
            return None
        return syminfo


def extract_at_str(die, at_name, default=None):
    """Helper to extract a string attribute from a DIE"""
    try:
        at = die.attributes[f"DW_AT_{at_name}"]
        return at.value.decode("ascii")
    except KeyError:
        return default


def extract_at_int(die, at_name, default=-1):
    """Helper to extract a string attribute from a DIE"""
    try:
        at = die.attributes[f"DW_AT_{at_name}"]
        return at.value
    except KeyError:
        return default


def extract_at_name(die, default=None):
    """Helper to extract a name from a DIE"""
    return extract_at_str(die, "name", default)


def extract_at_size(die, default=0):
    """Helper to extract the size attributes from a DIE"""
    return extract_at_int(die, "byte_size", default)


def extract_at_fileline(die, debug_line):
    src_file_off = extract_at_int(die, "decl_file", None)
    src_line = extract_at_int(die, "decl_line", None)
    if not src_file_off or not src_line:
        return (None, None)
    # decl_file is 1-based
    src_file_entry = debug_line["file_entry"][src_file_off - 1]
    src_file = src_file_entry["name"].decode("ascii")
    return (Path(src_file), src_line)


class DWARFDataRegistry:
    """
    Top level container for the data we mine from DWARF.
    This is passed around during the visit passes to collect
    and cross-reference the data.
    This is part of the main interface exposed to clients of this module.
    """
    def __init__(self, ef, dw, logger, arch_pointer_size):
        self.ef = ef
        self.dw = dw
        self.logger = logger
        self.arch_ptr_size = arch_pointer_size
        self.is_little_endian = self.ef.little_endian
        # map structure info by DIE offset
        self.struct_by_offset = {}
        # map structure info by unique key
        self.struct_by_key = {}
        # map typedef targets by DIE offset
        self.typedef_by_offset = {}
        # map struct offsets to callbacks for delayed fixup of fields that
        # would require recursively traversing structures.
        self.struct_ref_fixup = defaultdict(list)

    def get_struct_info(self, dedup=True) -> pd.DataFrame:
        """
        Export structure descriptors as pandas dataframe.
        If dedup is true, duplicate structs are removed.
        """
        valid_si = filter(lambda si: not si.is_anon, self.struct_by_key.values())
        si_frames = map(lambda si: si.to_frame(), valid_si)
        self.logger.debug("Build structure info dataframe")
        df = pd.concat(si_frames)
        df = df.set_index(["name", "src_file", "src_line", "member_name"])
        if dedup and not df.index.is_unique:
            dup = df.index.duplicated(keep="first")
            df = df[~dup]
        return df.sort_index()


@dataclass
class DWARFStructInfo:
    # Name of the structure or None if anonymous
    name: typing.Optional[str]
    # Structure size
    size: int
    # Compilation unit where we found the definition
    from_path: Path
    # Source file
    src_file: Path
    # Source line
    src_line: int
    # List of struct members descriptors
    members: typing.List["DWARFMemberInfo"] = field(default_factory=list)
    # Is the struct anonymous?
    is_anon: bool = False

    def to_frame(self):
        records = [m.to_frame_record() for m in self.members]
        if not records:
            # Add a single dummy row with no member name
            empty_record = {"name": np.nan, "offset": np.nan, "size": 0, "pad": 0}
            empty_record.update(DWARFTypeInfo().to_frame_record())
            records = [empty_record]
        df = pd.DataFrame.from_records(records)
        df = df.add_prefix("member_")
        df["name"] = self.name
        df["size"] = self.size
        df["from_path"] = self.from_path
        df["src_file"] = self.src_file
        df["src_line"] = self.src_line
        return df

    def apply_typedef(self, typedef: "DWARFTypeDef"):
        """Replace anonymous struct info with the typedef'ed info"""
        assert self.is_anon
        self.name = typedef.name
        self.src_file = typedef.src_file
        self.src_line = typedef.src_line
        self.is_anon = False

    @property
    def key(self):
        if self.is_anon:
            return ("anon", self.size, self.src_file, self.src_line)
        else:
            return (self.name, self.size, self.src_file, 0)

    def __str__(self):
        return f"{self.name} {self.src_file}:{self.src_line} @ {self.from_path}"


@dataclass
class DWARFMemberInfo:
    """
    Helper to wrap a struct member information
    """
    # member name, autogenerated <anon>.offset if anonymous
    name: str
    # type information
    type_info: "DWARFType"
    # byte offset in the structure
    offset: int
    # bit offset from byte offset for bitfields
    bit_offset: int = 0
    # byte size of the member
    size: int = None
    # bit size for bitfields
    bit_size: int = 0
    # byte padding to the next member
    pad: int = 0
    # bit padding to the next member for bitfields
    bit_pad: int = 0

    def __post_init__(self):
        assert self.type_info.size is not None
        if self.size is None:
            self.size = self.type_info.size

    def to_frame_record(self) -> dict:
        omit_fields = ["type_info"]
        record = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in omit_fields}
        type_record = {f"{k}": v for k, v in self.type_info.to_frame_record().items()}
        record.update(type_record)
        return record

    def __str__(self):
        value = f"{self.name} ({self.offset}:{self.bit_offset}) {self.type_info}"
        return value


@dataclass
class DWARFTypeDef:
    """Helper to hold information from a typedef"""
    name: str
    src_file: Path
    src_line: int

    def __str__(self):
        return f"typedef {self.name}"


@dataclass
class DWARFTypeInfo:
    """
    New class for holding type information
    """
    base_name: str = "<invalid>"
    name: str = ""
    size: int = 0
    # typedef name
    alias_name: str = None
    # Where it is defined, for typedefs and struct/union/enum
    src_file: Path = None
    src_line: int = None
    # Trailing internal padding, if applicable
    pad: int = 0
    # DIE offset for reference
    offset: int = None
    # Function parameters
    params: list = None
    # Array size if type is_array
    array_items: int = None
    # Flags
    is_typedef: bool = False
    is_ptr: bool = False
    is_struct: bool = False
    is_union: bool = False
    is_enum: bool = False
    is_const: bool = False
    is_volatile: bool = False
    is_array: bool = False

    def set_flag(self, name, value=True):
        if hasattr(self, name) and name.startswith("is_"):
            setattr(self, name, value)
        else:
            raise AttributeError(f"No DWARFTypeInfo flag {name}")

    def clear_flags(self):
        for f in fields(self):
            if f.name.startswith("is_"):
                setattr(self, f.name, False)

    def to_frame_record(self):
        record = {f"type_{k}": v for k, v in self.__dict__.items()}
        return record


class DWARFTypeResolver:
    """
    This is intended to be used internally to produce a DWARFTypeInfo describing
    a type.
    """
    def __init__(self, visitor, ctx):
        self.visitor = visitor
        self.ctx = ctx

    @property
    def debug_line(self):
        return self.visitor.current_debug_line

    def resolve_die(self, die) -> DWARFTypeInfo:
        chain = []
        self._follow_die_chain(die, chain)
        assert len(chain) > 0, "DIE resolved to nothing"
        # Now traverse the chain in reverse and fill the type info
        ti = DWARFTypeInfo()
        for die in reversed(chain):
            self._do_resolve_die(ti, die)
        if ti.is_struct:
            self._try_resolve_struct_pad(ti)
        return ti

    def _follow_die_chain(self, die, chain):
        try:
            t_die = die.get_DIE_from_attribute("DW_AT_type")
        except KeyError:
            # signal void
            chain.append(None)
            return

        chain.append(t_die)
        if (t_die.tag == "DW_TAG_base_type" or t_die.tag == "DW_TAG_subroutine_type"
                or t_die.tag == "DW_TAG_structure_type" or t_die.tag == "DW_TAG_union_type"
                or t_die.tag == "DW_TAG_enumeration_type"):
            return
        elif (t_die.tag == "DW_TAG_pointer_type" or t_die.tag == "DW_TAG_const_type"
              or t_die.tag == "DW_TAG_volatile_type" or t_die.tag == "DW_TAG_array_type"
              or t_die.tag == "DW_TAG_typedef"):
            self._follow_die_chain(t_die, chain)
        else:
            raise ValueError("Type resolver did not handle %s", t_die.tag)

    def _try_resolve_struct_pad(self, ti):
        """
        Attempt to resolve the trailing padding for a structure.
        If this is not possible, queue the TypeInfo for late resolution.
        """
        try:
            sinfo = self.ctx.struct_by_offset[ti.offset]
            if sinfo.members:
                ti.pad = sinfo.members[-1].pad
        except KeyError:
            # We can not recurse and visit again, so we keep a record of the
            # entries that need padding adjustment
            def callback(sinfo):
                if not sinfo.members:
                    return
                ti.pad = sinfo.members[-1].pad

            self.ctx.struct_ref_fixup[ti.offset].append(callback)

    def _do_resolve_die(self, ti, die):
        if die is None:
            ti.base_name = "void"
            ti.name = "void"
            return
        if die.tag == "DW_TAG_base_type":
            self._resolve_base_type(ti, die)
        elif die.tag == "DW_TAG_pointer_type":
            self._resolve_ptr_type(ti, die)
        elif die.tag == "DW_TAG_const_type":
            self._resolve_qualifier("const", ti, die)
        elif die.tag == "DW_TAG_volatile_type":
            self._resolve_qualifier("volatile", ti, die)
        elif die.tag == "DW_TAG_array_type":
            self._resolve_array_type(ti, die)
        elif die.tag == "DW_TAG_subroutine_type":
            self._resolve_fn_type(ti, die)
        elif die.tag == "DW_TAG_typedef":
            self._resolve_typedef(ti, die)
        elif (die.tag == "DW_TAG_structure_type" or die.tag == "DW_TAG_union_type"
              or die.tag == "DW_TAG_enumeration_type"):
            self._resolve_complex_type(ti, die)
        else:
            raise ValueError("Type resolver did not handle %s", die.tag)

    def _resolve_base_type(self, ti, die):
        name = extract_at_name(die)
        size = extract_at_size(die, None)
        assert size is not None, "No size for base type"
        ti.base_name = name
        ti.name = name
        ti.size = size

    def _resolve_ptr_type(self, ti, die):
        size = extract_at_size(die, self.ctx.arch_ptr_size)
        ti.name = ti.name + " *"
        ti.clear_flags()
        ti.set_flag("is_ptr")
        ti.size = size

    def _resolve_qualifier(self, qual, ti, die):
        ti.name = ti.name + f" {qual}"
        if qual == "volatile":
            ti.set_flag("is_volatile")
        elif qual == "const":
            ti.set_flag("is_const")

    def _resolve_array_type(self, ti, die):
        s_die = None
        for child in die.iter_children():
            if child.tag == "DW_TAG_subrange_type":
                s_die = child
                break
        assert s_die is not None
        lb = extract_at_int(s_die, "lower_bound", None)
        if lb is not None:
            raise ValueError("Unhandled DW_AT_lower_bound")

        count = extract_at_int(s_die, "count", None)
        ub = extract_at_int(s_die, "upper_bound", None)
        if count is not None:
            nitems = count
        elif ub is not None:
            nitems = ub + 1
        else:
            nitems = 0
        size_str = nitems or ""

        ti.name = ti.name + f" [{size_str}]"
        ti.set_flag("is_array")
        ti.array_items = nitems
        ti.size = ti.size * nitems

    def _resolve_fn_type(self, ti, die):
        ret_type = self.resolve_die(die)
        params = [ret_type]
        for child in die.iter_children():
            if child.tag == "DW_TAG_formal_parameter":
                param_type = self.resolve_die(child)
                params.append(param_type)
        param_str = ",".join(map(lambda p: p.name, params[1:]))

        ti.name = f"{ret_type.name}({param_str})" + ti.name
        ti.size = 0

    def _resolve_typedef(self, ti, die):
        name = extract_at_name(die)
        if not ti.is_typedef:
            # update alias name
            ti.alias_name = ti.name + (ti.alias_name or "")
        ti.set_flag("is_typedef")
        ti.name = name

    def _resolve_complex_type(self, ti, die):
        name = extract_at_name(die, "<anon>")
        size = extract_at_size(die)

        if die.tag == "DW_TAG_structure_type":
            ti.set_flag("is_struct")
            name_prefix = "struct"
        elif die.tag == "DW_TAG_union_type":
            ti.set_flag("is_union")
            name_prefix = "union"
        elif die.tag == "DW_TAG_enumeration_type":
            ti.set_flag("is_enum")
            name_prefix = "enum"
        else:
            assert False, "Not reached"
        ti.size = size
        ti.base_name = name
        ti.name = f"{name_prefix} {name}"
        ti.src_file, ti.src_line = extract_at_fileline(die, self.debug_line)
        ti.offset = die.offset


class DWARFVisitor:
    def __init__(self, ctx):
        self.ctx = ctx
        self.type_resolver = None
        self.current_unit = None
        self.current_unit_path = None
        self.current_debug_line = None

    def visit_unit(self, unit, die):
        self.current_unit = unit
        self.current_unit_path = Path(die.get_full_path())
        self.current_debug_line = self.ctx.dw.line_program_for_CU(unit)
        self.ctx.logger.debug("DWARF %s visit %s", self, self.current_unit_path)
        self.type_resolver = DWARFTypeResolver(self, self.ctx)

    def visit_structure_type(self, die):
        pass

    def visit_union_type(self, die):
        pass

    def visit_typedef(self, die):
        pass

    def __str__(self):
        return self.__class__.__name__


class StructInfoVisitor(DWARFVisitor):
    """
    Handle generation of the structure information from the DWARF tree.
    This will fill the `DataRegistry.struct_by_offset`, `DataRegistry.struct_by_key`
    and `DataRegistry.typedef_by_offset` fields.
    """
    def _resolve_type(self, member_name, die):
        try:
            return self.type_resolver.resolve_die(die)
        except Exception:
            self.ctx.logger.error("Error while handling member %s", member_name)
            raise

    def _resolve_nested_anon(self, prefix, offset, die):
        """Resolve members from a nested anonymous struct/union"""
        members = self._resolve_members(die)
        for m in members:
            m.offset += offset
            m.name = f"{prefix}.{m.name}"
        return members

    def _resolve_members(self, die):
        members = []
        for child in die.iter_children():
            if child.tag != "DW_TAG_member":
                continue
            offset = extract_at_int(child, "data_member_location", 0)
            name = extract_at_name(child, f"<anon>.{offset}")
            # if this is an anonymous struct/union, we traverse it and add its members
            # directly as members of the current struct/union
            member_die = child.get_DIE_from_attribute("DW_AT_type")
            member_type_name = extract_at_name(member_die)
            if member_die.tag == "DW_TAG_structure_type" and member_type_name is None:
                members.extend(self._resolve_nested_anon(name, offset, member_die))
            else:
                type_info = self._resolve_type(name, child)
                member = DWARFMemberInfo(name, type_info, offset)
                self._resolve_bitfield(child, member)
                members.append(member)
        self._resolve_members_padding(die, members)
        return members

    def _resolve_bitfield(self, die, member):
        """
        Check if a member die contains a bitfield and resolve the bitfield size/offset
        """
        bit_offset = extract_at_int(die, "bit_offset", 0)
        bit_size = extract_at_int(die, "bit_size", None)
        if bit_size is None:
            return
        if self.ctx.is_little_endian:
            bit_offset = member.type_info.size * 8 - (bit_offset + bit_size)
        assert bit_offset >= 0
        member.bit_offset = bit_offset
        member.size = int(np.floor(bit_size / 8))
        member.bit_size = bit_size % 8

    def _resolve_members_padding(self, die, members):
        # compute member paddings, assume there is never leading padding in a struct
        # sort to make sure ordering is reliable
        members = sorted(members, key=lambda m: m.offset * 8 + m.bit_offset)
        prev_off = extract_at_size(die) * 8
        for m in reversed(members):
            curr_off = m.offset * 8 + m.bit_offset
            if m.bit_size != 0:
                # XXX maybe test for None to distinguish non-bitfields for sure
                total_pad = prev_off - (curr_off + m.bit_size)
            else:
                total_pad = prev_off - (curr_off + m.size * 8)
            assert total_pad >= 0
            m.pad, m.bit_pad = int(total_pad / 8), total_pad % 8
            prev_off = curr_off
        return members

    def visit_structure_type(self, die):
        at_decl = die.attributes.get("DW_AT_declaration", None)
        if at_decl:
            # Skip non-defining entries
            return

        src_file, src_line = extract_at_fileline(die, self.current_debug_line)
        if not src_file or not src_line:
            raise ValueError("Missing file and line attributes")

        size = extract_at_size(die, None)
        if size is None:
            raise ValueError("Unexpected DWARF struct size attributes")

        name = extract_at_name(die)
        try:
            members = self._resolve_members(die)
        except Exception:
            self.ctx.logger.error("Error while handling %s from %s at %s %s", name, self.current_unit_path, src_file,
                                  src_line)
            raise

        if len(members) == 0:
            self.ctx.logger.debug("Struct with no members %s %s:%d", name, src_file, src_line)

        is_anon = False
        if name is None:
            # Try to see whether the struct has been typedef'ed
            try:
                typedef = self.ctx.typedef_by_offset[die.offset]
                name = typedef.name
                src_file = typedef.src_file
                src_line = typedef.src_line
            except KeyError:
                is_anon = True

        si = DWARFStructInfo(name=name,
                             size=size,
                             from_path=self.current_unit_path,
                             src_file=src_file,
                             src_line=src_line,
                             members=members,
                             is_anon=is_anon)
        self.ctx.struct_by_key[si.key] = si
        self.ctx.struct_by_offset[die.offset] = si
        if die.offset in self.ctx.struct_ref_fixup:
            for cbk in self.ctx.struct_ref_fixup[die.offset]:
                cbk(si)
            del self.ctx.struct_ref_fixup[die.offset]

    def visit_typedef(self, die):
        """
        If we find a typedef, it might be for an anonymous struct or union we are going
        to encounter later on. To resolve the correct name we store anonymous struct and union
        typedef targets for later lookup. We do not care about other typedefs as they will
        be in the type chain if the type is encountered.
        """
        target_die = die.get_DIE_from_attribute("DW_AT_type")
        if target_die.tag == "DW_TAG_structure_type" or target_die.tag == "DW_TAG_union_type":
            target_name = extract_at_name(target_die, None)
            if target_name is not None:
                # target struct is not anon, skip
                return
            typedef_name = extract_at_name(die)
            assert typedef_name is not None, "typedef without name"
            src_file, src_line = extract_at_fileline(die, self.current_debug_line)
            typedef = DWARFTypeDef(typedef_name, src_file, src_line)
            self.ctx.typedef_by_offset[target_die.offset] = typedef

            # Try to patch any anon structure we already found
            target = self.ctx.struct_by_offset.get(target_die.offset)
            if target and target.is_anon:
                target.apply_typedef(typedef)


class DWARFInfoSource:
    """
    Helper to access DWARF debug information for an ELF file.
    This currently relies on the capabilities of pyelftools and
    does not support DWARF-5.
    This is part of the main interface exposed to clients of this module.
    """
    def __init__(self, logger, path: Path, arch_pointer_size: int):
        self.logger = logger
        self.path = path
        self._fd = open(path, "rb")
        self._ef = ELFFile(self._fd)
        self._dw = self._ef.get_dwarf_info()
        self._arch_pointer_size = arch_pointer_size
        self._ctx = DWARFDataRegistry(self._ef, self._dw, logger, arch_pointer_size)

    def _handle_die(self, die, visitor):
        if die.tag == "DW_TAG_structure_type":
            visitor.visit_structure_type(die)
        elif die.tag == "DW_TAG_union_type":
            visitor.visit_union_type(die)
        elif die.tag == "DW_TAG_typedef":
            visitor.visit_typedef(die)

    def _handle_unit(self, unit, visitor):
        unit_die = unit.get_top_DIE()
        visitor.visit_unit(unit, unit_die)
        for die in unit_die.iter_children():
            self._handle_die(die, visitor)

    def parse_dwarf(self):
        """Main entry point to extract all the DWARF information"""
        visitors = [StructInfoVisitor(self._ctx)]
        for unit in self._dw.iter_CUs():
            # There might be an opportunity to further parallelize this if it is too slow
            for v in visitors:
                self._handle_unit(unit, v)

    def get_dwarf_data(self) -> DWARFDataRegistry:
        """Once parse_dwarf has completed, this can be used to retrieve the data registry"""
        return self._ctx


class DWARFHelper:
    """
    Manages DWARF information for all object files imported by a
    benchmark instance.
    Each source object file is registered using a unique name
    (akin to what is done with the symbolizer). At some point
    we may need to integrate this with the symbolizer in some way.
    """
    def __init__(self, benchmark: "BenchmarkBase"):
        self.logger = new_logger(f"{benchmark.uuid}.DWARFHelper")
        self.objects = {}

    def register_object(self, obj_key: str, path: Path, arch_pointer_size=8):
        assert obj_key not in self.objects, "Duplicate DWARF object source"
        self.objects[obj_key] = DWARFInfoSource(self.logger, path, arch_pointer_size=arch_pointer_size)

    def get_object(self, obj_key: str) -> typing.Optional[DWARFInfoSource]:
        return self.objects.get(obj_key, None)
