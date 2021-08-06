import subprocess as subp
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


class SymResolver:
    """
    Resolve symbols from a set of ELF files with optional mapping addresses
    """
    def __init__(self, benchmark: "BenchmarkBase"):
        self.bench = benchmark
        self.files = {}
        self.mapping = {}
        self.symbols = SortedDict()

    def import_symbols(self, path: Path, mapbase: int, include_local=False):
        info = ELFInfo(path)
        if info.is_dynamic():
            mapbase = 0
        self.files[mapbase] = info
        self.mapping[path] = mapbase
        symtab = info.ef.get_section_by_name(".symtab")
        for sym in symtab.iter_symbols():
            if not include_local and sym.name.startswith(".LBB"):
                continue
            self.symbols[sym["st_value"] + mapbase] = SymInfo(name=sym.name, filepath=path)

    def lookup(self, addr: int):
        index = self.symbols.bisect(addr) - 1
        if index < 0:
            return None
        info = self.symbols.values()[index]
        # Sanity check:
        if self.bench.manager_config.verbose:
            return info
            addr2line = self.bench.manager_config.sdk_path.expanduser() / "sdk" / "bin" / "addr2line"
            base = self.mapping[info.filepath]
            print(base, addr, addr - base)
            result = subp.run([addr2line, "--obj", info.filepath, f"0x{int(addr - base):x}"], capture_output=True)
            print(result, "=?=", info.name)
        return info
