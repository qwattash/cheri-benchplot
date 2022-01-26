import itertools as it
import typing
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
    from_path: Path = None
    is_anon: bool = False
    src_file: typing.Optional[str] = None
    src_line: typing.Optional[int] = None

    def __eq__(self, other):
        return self.name == other.name and self.size == other.size and self.src_file == other.src_file and self.src_line == other.src_line

    def __lt__(self, other):
        return self.size < other.size


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

    def _dw_struct_info(self, die, debug_line) -> typing.Optional[DWARFStructInfo]:
        at_name = die.attributes.get("DW_AT_name", None)
        at_B_size = die.attributes.get("DW_AT_byte_size", None)
        at_b_size = die.attributes.get("DW_AT_bit_size", None)
        at_decl = die.attributes.get("DW_AT_declaration", None)
        at_file = die.attributes.get("DW_AT_decl_file", None)
        at_line = die.attributes.get("DW_AT_decl_line", None)

        name = at_name.value.decode("ascii") if at_name else None
        src_file = None
        src_line = None
        if at_B_size:
            size = at_B_size.value
        elif at_b_size:
            size = at_b_size.value / 8
        elif at_decl:
            # Declaration, skip
            return None
        else:
            self.logger.warning("Unexpected DWARF struct size attributes")
            return None
        si = DWARFStructInfo(name, size)

        if at_file and at_line and at_file.value > 0:
            # at_file is 1-based
            entry = debug_line["file_entry"][at_file.value - 1]
            si.src_file = entry.name.decode("ascii")
            si.src_line = at_line.value
        if name is None:
            si.is_anon = True
        return si

    def _extract_struct_info_cu(self, cu, struct_info):
        debug_line = self._dw.line_program_for_CU(cu)
        die_cu = cu.get_top_DIE()
        cu_path = die_cu.get_full_path()
        for die in die_cu.iter_children():
            if die.tag == "DW_TAG_structure_type":
                info = self._dw_struct_info(die, debug_line)
                if info is None:
                    continue
                info.from_path = Path(cu_path)
                if info.is_anon:
                    # For anonymous structures, we need to generate an unique name
                    # based on where they are defined
                    if info.src_file:
                        src_name = info.src_file
                    else:
                        src_name = cu_path
                    info.name = f"<anon.{src_name}>"
                if (info.name, info.size) not in struct_info:
                    struct_info[(info.name, info.size)] = info
            elif die.tag == "DW_TAG_union_type":
                pass

    def extract_struct_info(self, as_dict=False) -> pd.DataFrame:
        """
        Iterate all compilation units and extract information on structure
        size, alignment etc.
        """
        struct_info = {}
        for cu in self._dw.iter_CUs():
            self._extract_struct_info_cu(cu, struct_info)
        if as_dict:
            return struct_info
        df = pd.DataFrame.from_records(map(asdict, struct_info.values()))
        return df.set_index(["name", "size"])


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
