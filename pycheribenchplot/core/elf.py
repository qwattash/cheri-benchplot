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


@dataclass
class SymInfo:
    name: str
    filepath: Path
    size: int
    addr: int


class SymResolver:
    """
    Resolve symbols from a set of ELF files with optional mapping addresses
    """
    def __init__(self, benchmark: "BenchmarkBase"):
        self.bench = benchmark
        self.files = {}
        self.mapping = {}
        self.symbols = SortedDict()
        self.functions = SortedDict()

    def import_symbols(self, path: Path, mapbase: int):
        info = ELFInfo(path)
        if info.is_dynamic():
            mapbase = 0
        self.files[mapbase] = info
        self.mapping[path] = mapbase
        symtab = info.ef.get_section_by_name(".symtab")
        for sym in symtab.iter_symbols():
            addr = sym["st_value"] + mapbase
            syminfo = SymInfo(name=sym.name, filepath=path, addr=addr, size=sym["st_size"])
            st_info = sym["st_info"]
            self.symbols[addr] = syminfo
            if st_info.type == "STT_FUNC":
                self.functions[addr] = syminfo

    def get_sym_addr(self, sym_name: str):
        for addr, sym in self.symbols.items():
            if sym.name == sym_name:
                base = self.mapping[sym.filepath]
                return base + addr
        return None

    def _lookup(self, sym_map: SortedDict[int, SymInfo], addr: int) -> typing.Optional[tuple[int, SymInfo]]:
        """
        Find the symbol preceding the given address
        """
        index = sym_map.bisect(addr) - 1
        if index < 0:
            return None
        info = sym_map.values()[index]
        return (index, info)

    def lookup_fn(self, addr: int) -> SymInfo:
        result = self._lookup(self.functions, addr)
        if result is None:
            return None
        _, syminfo = result
        return syminfo

    def lookup_fn_bounded(self, addr: int) -> typing.Optional[SymInfo]:
        result = self._lookup(self.functions, addr)
        if result is None:
            return None
        index, syminfo = result
        if addr > syminfo.addr + syminfo.size:
            return None
        return syminfo

    def lookup_fn_exact(self, addr: int) -> typing.Optional[SymInfo]:
        """
        Find the symbol matching exactly the given address.
        If none is found, return None.
        """
        return self.functions.get(addr, None)
