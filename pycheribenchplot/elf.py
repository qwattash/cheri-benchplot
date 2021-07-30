
import pandas as pd
from pathlib import Path
from sortedcontainers import SortedDict
from elftools.elf.elffile import ELFFile
from elftools.elf.enums import ENUM_ST_SHNDX

class ELFInfo:
    def __init__(self, path: Path):
        self.path = path
        self.ef = ELFFile(open(path, "rb"))

class SymResolver:
    """
    Resolve symbols from a set of ELF files with optional mapping addresses
    """

    def __init__(self):
        self.files = {}
        self.symbols = SortedDict()

    def register(self, einfo: ELFInfo, base_addr):
        self.files[base_addr] = einfo
        symtab = einfo.ef.get_section_by_name(".symtab")
        for sym in symtab.iter_symbols():
            self.symbols[sym["st_value"] + base_addr] = sym.name

    def resolve(self, addr):
        index = self.symbols.bisect(addr) - 1
        if index < 0:
            return None
        return self.symbols.values()[index]

