import itertools as it
import typing
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass
class DWARFStructInfo:
    name: str
    size: int
    from_path: Path
    src_file: str
    src_line: int
    is_anon: bool = False

    @property
    def key(self):
        if self.is_anon:
            return ("anon", self.size, self.src_file, self.src_line)
        else:
            return (self.name, self.size, self.src_file, 0)


class DWARFVisitor:
    def __init__(self, dw):
        self.dw = dw
        self.current_unit = None
        self.current_debug_line = None

    def visit_structure_type(self, die):
        pass

    def visit_union_type(self, die):
        pass

    def visit_typedef(self, die):
        pass


class StructInfoVisitor(DWARFVisitor):
    def __init__(self, dw):
        super().__init__(dw)
        self.struct_by_key = {}
        self.struct_by_offset = {}

    def visit_structure_type(self, die):
        at_decl = die.attributes.get("DW_AT_declaration", None)
        if at_decl:
            # Skip non-defining entries
            return

        at_name = die.attributes.get("DW_AT_name", None)
        at_B_size = die.attributes.get("DW_AT_byte_size", None)
        at_b_size = die.attributes.get("DW_AT_bit_size", None)
        at_file = die.attributes.get("DW_AT_decl_file", None)
        at_line = die.attributes.get("DW_AT_decl_line", None)

        name = at_name.value.decode("ascii") if at_name else None
        if at_B_size:
            size = at_B_size.value
        elif at_b_size:
            size = at_b_size.value / 8
        else:
            self.logger.warning("Unexpected DWARF struct size attributes")

        if not at_file or not at_line:
            self.logger.warning("Missing file and line attributes")
            return
        # at_file is 1-based
        entry = self.current_debug_line["file_entry"][at_file.value - 1]
        src_file = entry.name.decode("ascii")
        src_line = at_line.value
        si = DWARFStructInfo(name, size, self.current_unit_path, src_file, src_line)
        if name is None:
            si.is_anon = True
        # Duplicate keys are not allowed, if the key is equal, the struct
        # we are referring to must be the same.
        self.struct_by_key[si.key] = si
        self.struct_by_offset[die.offset] = si


class DWARFInfoSource:
    """
    Helper to access DWARF debug information for an ELF file.
    This currently relies on the capabilities of pyelftools and
    does not support DWARF-5.
    """
    def __init__(self, logger, path: Path):
        self.logger = logger
        self.path = path
        self._fd = open(path, "rb")
        self._ef = ELFFile(self._fd)
        self._dw = self._ef.get_dwarf_info()

    def _struct_typedef(self, die):
        parent = die.get_parent()
        get_DIE_from_attribute()

    def _handle_die(self, die, visitor):
        if die.tag == "DW_TAG_structure_type":
            visitor.visit_structure_type(die)
        elif die.tag == "DW_TAG_union_type":
            visitor.visit_union_type(die)
        elif die.tag == "DW_TAG_typedef":
            visitor.visit_typedef(die)

    def _handle_unit(self, unit, visitor):
        unit_die = unit.get_top_DIE()
        visitor.current_unit = unit
        visitor.current_unit_path = Path(unit_die.get_full_path())
        visitor.current_debug_line = self._dw.line_program_for_CU(unit)
        for die in unit_die.iter_children():
            self._handle_die(die, visitor)

    def _visit(self, visitor):
        for unit in self._dw.iter_CUs():
            self._handle_unit(unit, visitor)

    def extract_struct_info(self) -> pd.DataFrame:
        """
        Iterate all compilation units and extract information on structure
        size, alignment etc.
        """
        visitor = StructInfoVisitor(self._dw)
        self._visit(visitor)
        df = pd.DataFrame.from_records(map(asdict, visitor.struct_by_key.values()))
        return df.set_index(["name", "src_file", "src_line"])


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

    def register_object(self, obj_key: str, path: Path):
        assert obj_key not in self.objects, "Duplicate DWARF object source"
        self.objects[obj_key] = DWARFInfoSource(self.logger, path)

    def get_object(self, obj_key: str) -> typing.Optional[DWARFInfoSource]:
        return self.objects.get(obj_key, None)
