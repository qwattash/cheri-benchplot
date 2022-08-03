import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from elftools.elf.elffile import ELFFile
from sortedcontainers import SortedDict

from ..util import new_logger


@dataclass
class SymInfo:
    name: str
    filepath: Path
    size: int
    addr: int
    unknown: bool = False

    @classmethod
    def unknown(cls, addr):
        return SymInfo(name=f"0x{addr:x}", filepath=None, size=0, addr=addr, unknown=True)

    def __bool__(self):
        return not self.unknown


class SymbolizerMapping(ABC):
    """
    Represent a single file mapped in an address space.
    """
    def __init__(self, logger, path: Path):
        self.logger = logger
        self.path = path
        self.ef = ELFFile(open(path, "rb"))

    def is_dynamic(self) -> bool:
        """
        Check whether the binary file is dynamically linked.
        """
        if self.ef.header.e_type == "ET_DYN":
            return True
        return False

    def end_addr(self) -> bool:
        """
        Get the last mapped address according to the elf binary sections.
        """
        maxaddr = 0
        for p in self.ef.iter_segments():
            pend = p["p_vaddr"] + p["p_memsz"]
            if p["p_type"] == "PT_LOAD" and pend > maxaddr:
                maxaddr = pend
        return maxaddr

    @abstractmethod
    def lookup_function(self, addr: int) -> SymInfo:
        ...


class ELFToolsMapping(SymbolizerMapping):
    """
    Handle fetching symbol information for a binary using elftools.
    """
    def __init__(self, logger, path):
        super().__init__(logger, path)
        self.symbols = SortedDict()
        self.functions = SortedDict()

        self.logger.debug("Load symbols for %s dyn=%s", path, self.is_dynamic())
        symtab = self.ef.get_section_by_name(".symtab")
        for sym in symtab.iter_symbols():
            addr = sym["st_value"]
            syminfo = SymInfo(name=sym.name, filepath=path, addr=addr, size=sym["st_size"])
            st_info = sym["st_info"]
            if st_info.type == "STT_FUNC":
                self.functions[addr] = syminfo
            else:
                self.symbols[addr] = syminfo

    def lookup_function(self, addr):
        idx = self.functions.bisect(addr) - 1
        if idx < 0:
            return SymInfo.unknown(addr)
        syminfo = self.functions.values()[idx]
        # XXX Symbol size check seems unreliable for some reason?
        # if syminfo.addr + syminfo.size < addr:
        #     return None
        return syminfo


class LLVMToolsMapping(SymbolizerMapping):
    """
    Handle fetching symbol information for a binary using addr2line
    """
    def __init__(self, sdk_path, logger, path):
        super().__init__(logger, path)
        self.addr2line_bin = sdk_path / "sdk" / "bin" / "llvm-addr2line"
        objdump_bin = sdk_path / "sdk" / "bin" / "llvm-objdump"

        # Build the symbol table
        # XXX invoke objdump

        # Lazily spawned addr2line
        self.addr2line = None

    def _start_addr2line(self):
        """
        Startup the internal addr2line process
        """
        assert self.addr2line is None
        self.addr2line = subprocess.Popen([self.addr2line_bin, "-obj", str(self.path), "-f"],
                                          stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,
                                          text=True,
                                          encoding="utf-8")

    def lookup_function(self, addr):
        if self.addr2line is None:
            self._start_addr2line()
        self.addr2line.stdin.write(f"0x{addr:x}\n")
        self.addr2line.stdin.flush()
        symbol = self.addr2line.stdout.readline().strip()
        # Currently we discard line info
        line_info = self.addr2line.stdout.readline().strip()
        if line_info == "??:0":
            return SymInfo.unknown(addr)
        return SymInfo()


class Symbolizer(ABC):
    """
    A symbolizer takes care of symbols within an address space.
    When loading data, file mappings should be registered to the symbolizer
    responsible for a specific address space.
    The symbolizer will then be used to query symbols by address.
    """
    def __init__(self, logger):
        self.logger = logger
        # Sorted mapping of <base address> => SymbolizerMapping
        self.mappings = SortedDict()

    @abstractmethod
    def _make_mapping(self, path: Path) -> SymbolizerMapping:
        ...

    def add_mapping(self, base: int, path: Path):
        """
        Register a new mapping to the given file.

        :param base: Relocated base address of the mapping
        :param path: Path to the ELF executable or shared library being mapped
        """
        if base in self.mappings:
            raise ValueError("Duplicate mapping registered")
        self.mappings[base] = self._make_mapping(path)

    def lookup_function(self, addr: int) -> SymInfo:
        """
        Lookup a function at the given address.

        :param addr: The address to lookup
        """
        idx = self.mappings.bisect(addr) - 1
        if idx < 0:
            return SymInfo.unknown(addr)
        base, mapping = self.mappings.items()[idx]
        if not mapping.is_dynamic():
            base = 0
        sym = mapping.lookup_function(addr - base)
        sym.addr = base + sym.addr
        if not sym:
            # Also update the name for unknown symbols
            sym.name = f"0x{sym.addr:x}"
        return sym


class ELFToolsSymbolizer(Symbolizer):
    """
    Symbolizer that uses python elftools to load symbols from the binaries.
    """
    def _make_mapping(self, path):
        return ELFToolsMapping(self.logger, path)


class LLVMToolsSymbolizer(Symbolizer):
    """
    Symbolizer that calls addr2line to resolve function addresses
    """
    def __init__(self, logger, sdk_path: Path):
        super().__init__(logger)
        self.sdk_path = sdk_path

    def _make_mapping(self, path):
        return LLVMToolsMapping(self.sdk_path, self.logger, path)


class AddressSpaceManager:
    """
    Manager symbolizers for various address spaces.
    Each address space is identified by a unique name.
    An address space may be marked as shared, this is usually the case for the
    kernel address space.
    """
    def __init__(self, benchmark: "Benchmark"):
        self.benchmark = benchmark
        self.logger = new_logger(f"{benchmark.uuid}.symbolizer")
        self.shared_addrspaces = []
        self.addrspaces = {}

    def register_address_space(self, as_key: str, shared: bool = False):
        """
        Create a new address space.

        :param as_key: Address space identifier
        :param shared: The address space is shared, and will be used as fallback
        if symbols are not found in other address spaces.
        """
        if as_key in self.addrspaces:
            self.logger.debug(f"Attempted to register duplicate address space {as_key}")
            # For now just reuse the one we already have
            return
        if self.benchmark.session.analysis_config.use_builtin_symbolizer:
            symbolizer = ELFToolsSymbolizer(self.logger)
        else:
            symbolizer = LLVMToolsSymbolizer(self.logger, self.benchmark.user_config.sdk_path)
        self.addrspaces[as_key] = symbolizer
        if shared:
            self.shared_addrspaces.append(symbolizer)

    def register_address_space_alias(self, as_key: str, alias: str):
        """
        Register an address space that is the same as an existing one
        """
        self.addrspaces[alias] = self.addrspaces[as_key]

    def add_mapped_binary(self, as_key: str, base: int, path: Path):
        """
        Register a new mapped binary for the given address space

        :param as_key: The address space identifier
        :param base: The base load address
        :param path: The path of the ELF binary mapped
        """
        try:
            symbolizer = self.addrspaces[as_key]
        except KeyError:
            self.logger.error("Attempted to register mapped binary for missing address space %s", as_key)
        symbolizer.add_mapping(base, path)

    def lookup_function(self, as_key: str, addr: int, exact: bool = False) -> SymInfo:
        """
        Search for a function in the given address space. If exact is set, only return
        a valid symbol information if the given address matches exactly the function symbol address.

        :param as_key: The address space identifier
        :param addr: The address to lookup
        :param exact: Only look for exact matches.
        :return: Symbol information
        """
        if as_key not in self.addrspaces:
            self.logger.warn("Lookup function 0x%x in non-existing address space %s", addr, as_key)
            return SymInfo.unknown(addr)
        target_symbolizer = self.addrspaces[as_key]
        sym = target_symbolizer.lookup_function(addr)
        if sym:
            return sym
        # Fallback shared address space lookup
        for symbolizer in self.shared_addrspaces:
            if symbolizer == target_symbolizer:
                # We already checked this one
                continue
            sym = symbolizer.lookup_function(addr)
            if sym:
                return sym
        return SymInfo.unknown(addr)

    def lookup_function_to_df(self, df: pd.DataFrame, addr_column: str, as_key_column: str, exact: bool = False):
        """
        Resolve function file/symbol for each entry of the given dataframe
        addr_column. A new dataframe is returned with the same index and the
        columns ["file", "symbol", "valid_symbol"].

        :param df: The input dataframe
        :param addr_column: Column in the input dataframe containing the addresses to lookup
        :param as_key_column: Column in the input dataframe containing the address space identifiers
        :param exact: Perform an exact lookup (see :method:`lookup_function()`)
        """
        # XXX-AM: the symbol size does not appear to be reliable?
        # otherwise we should check also the resolved syminfo size as in:
        # sym_end = resolved.map(lambda syminfo: syminfo.addr + syminfo.size if syminfo else np.nan)
        # size_mismatch = (~sym_end.isna()) & (self.df["start"] > sym_end)
        # self.df.loc[size_mismatch, "valid_symbol"] = "size-mismatch"
        resolved = df.apply(lambda row: self.lookup_function(row[as_key_column], row[addr_column], exact=exact), axis=1)
        # Now fill the resolved parameters from the symbol information objects
        out_df = pd.DataFrame(None, index=resolved.index)
        out_df["valid_symbol"] = resolved.mask(resolved, "no-match")
        out_df["valid_symbol"].where(resolved, "ok", inplace=True)
        out_df["symbol"] = resolved.map(lambda si: si.name, na_action="ignore")
        out_df["symbol"].mask(resolved, df[addr_column].transform(lambda addr: f"0x{addr:x}"), inplace=True)
        # Note: For the file name, we omit the directory part as otherwise the same executable
        # in different directories will be picked up as a completely different file. This is
        # not useful when comparing different compilations that have different paths e.g. the kernel
        # TODO: We also have to handle rtld manually to map its name.
        out_df["file"] = resolved.map(lambda si: si.filepath.name, na_action="ignore")
        out_df["file"].mask(resolved, "unknown", inplace=True)
        return out_df
