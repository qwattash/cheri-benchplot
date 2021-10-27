import typing
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sortedcontainers import SortedDict
from elftools.elf.elffile import ELFFile
from elftools.elf.enums import ENUM_ST_SHNDX, ENUM_E_TYPE


class ELFInfo:
    def __init__(self, path: Path):
        self.path = path
        self.ef = ELFFile(open(path, "rb"))

    def is_dynamic(self):
        if self.ef.header.e_type == ENUM_E_TYPE["ET_DYN"]:
            return True
        return False

    def image_end_addr(self):
        maxaddr = 0
        for p in self.ef.iter_segments():
            pend =  p["p_vaddr"] + p["p_memsz"]
            if p["p_type"] == "PT_LOAD" and pend > maxaddr:
                maxaddr = pend
        return maxaddr


@dataclass
class SymInfo:
    name: str
    filepath: Path
    size: int
    addr: int

@dataclass
class MapInfo:
    end: int
    elf: ELFInfo
    symbols: SortedDict
    functions: SortedDict

class SymResolver:
    """
    Resolve symbols from a set of ELF files with optional mapping addresses
    """
    def __init__(self, benchmark: "BenchmarkBase"):
        self.bench = benchmark
        self.files = {}

        # Map base address to the corresponding ELF object
        # For each ELF object we have a functions map for looking up.
        self.mapping = SortedDict()

    def import_symbols(self, path: Path, mapbase: int, mapend: int=None):
        info = ELFInfo(path)
        if info.is_dynamic():
            mapbase = 0
        if mapend is None:
            # Try to infer size of image
            mapend = mapbase + info.image_end_addr()
        self.files[mapbase] = info
        print("XXX", path, f"0x{mapbase:x}", f"0x{mapend:x}")

        functions = SortedDict()
        symbols = SortedDict()
        symtab = info.ef.get_section_by_name(".symtab")
        for sym in symtab.iter_symbols():
            addr = sym["st_value"] + mapbase
            syminfo = SymInfo(name=sym.name, filepath=path, addr=addr, size=sym["st_size"])
            st_info = sym["st_info"]
            symbols[addr] = syminfo
            if st_info.type == "STT_FUNC":
                functions[addr] = syminfo
        self.mapping[mapbase] = MapInfo(end=mapend, elf=info, functions=functions, symbols=symbols)

    def get_sym_addr(self, sym_name: str):
        for base, mapping in self.mapping.items():
            for addr, sym in mapping.symbols.items():
                if sym.name == sym_name:
                    return addr
        return None

    def _lookup_mapping(self, addr: int) -> typing.Optional[MapInfo]:
        """
        Find mapping containing the given symbol
        """
        mapidx = self.mapping.bisect(addr) - 1
        if mapidx < 0:
            return None
        mapinfo = self.mapping.values()[mapidx]
        if mapinfo.end < addr:
            return None
        return mapinfo

    def _lookup_symbol(self, sym_map: SortedDict[int, SymInfo], addr: int) -> typing.Optional[SymInfo]:
        """
        Find the symbol preceding the given address, if it belongs to a valid mapping
        """
        symidx = sym_map.bisect(addr) - 1
        if symidx < 0:
            return None
        info = sym_map.values()[symidx]
        return info

    def lookup_fn(self, addr: int) -> typing.Optional[SymInfo]:
        mapinfo = self._lookup_mapping(addr)
        if mapinfo is None:
            return None
        syminfo = self._lookup_symbol(mapinfo.functions, addr)
        if syminfo is None:
            return None
        return syminfo

    def lookup_fn_bounded(self, addr: int) -> typing.Optional[SymInfo]:
        mapinfo = self._lookup_mapping(addr)
        if mapinfo is None:
            return None
        syminfo = self._lookup_symbol(mapinfo.functions, addr)
        if syminfo is None or addr > syminfo.addr + syminfo.size:
            return None
        return syminfo

    def lookup_fn_exact(self, addr: int) -> typing.Optional[SymInfo]:
        """
        Find the symbol matching exactly the given address.
        If none is found, return None.
        """
        mapinfo = self._lookup_mapping(addr)
        if mapinfo is None:
            return None
        return mapinfo.functions.get(addr, None)
