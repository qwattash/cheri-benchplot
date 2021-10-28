import typing
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sortedcontainers import SortedDict
from elftools.elf.elffile import ELFFile

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
    def __init__(self, sset, mapbase: int, path: Path):
        self.logger = sset.logger
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

    def lookup_fn(self, addr: int):
        idx = self.functions.bisect(addr) - 1
        if idx < 0:
            return None
        syminfo = self.functions.values()[idx]
        # XXX Symbol size check seems unreliable for some reason?
        # if syminfo.addr + syminfo.size < addr:
        #     return None
        return syminfo


class SymbolizerSet:
    """
    Per-AS set of mappings that have associated symbols.
    """
    def __init__(self, symbolizer):
        self.logger = symbolizer.logger
        self.mappings = SortedDict()

    def add_sym_source(self, mapbase: int, path: Path):
        mapping = SymbolizerMapping(self, mapbase, path)
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
    def __init__(self):
        self.logger = new_logger("symbolizer")
        self.addrspace = {}

    def _get_or_create_addrspace(self, key: str):
        if key not in self.addrspace:
            self.logger.debug("New symbolizer address space %s", key)
            self.addrspace[key] = SymbolizerSet(self)
        return self.addrspace[key]

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
