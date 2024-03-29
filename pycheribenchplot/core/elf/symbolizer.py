import io
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
    is_unknown: bool = False

    @classmethod
    def unknown(cls, addr):
        return SymInfo(name=f"0x{addr:x}", filepath=None, size=0, addr=addr, is_unknown=True)

    def __hash__(self):
        return hash((self.is_unknown, self.name))

    def __eq__(self, other):
        return self.is_unknown == other.is_unknown and self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __bool__(self):
        return not self.is_unknown

    def to_folded_stack_str(self) -> str:
        """
        Covert to the string representation expected by flame graphs.
        Note that we use addr == -1 to signal an empty stack entry.
        """
        if self.is_unknown and self.addr == -1:
            return ""
        elif self.is_unknown:
            return f"??`{self.name}"
        else:
            location = Path(self.filepath).stem
            if location.startswith("kernel"):
                location = "kernel"
            return f"{location}`{self.name}"

    def __str__(self):
        if self.is_unknown and self.addr == -1:
            return "|empty|"
        return f"|{self.name}|"


class ELFSymbolReader(ABC):
    """
    Base class for symbol extractors
    """
    @classmethod
    def create(cls, session: "Session", *args):
        """
        Build a symbol reader based on the strategy configured in the session
        """
        if session.config.use_builtin_symbolizer:
            return ELFToolsSymbolReader(session, *args)
        else:
            return LLVMSymbolReader(session, *args)

    def __init__(self, session: "Session", path: Path):
        self.session = session
        self.logger = session.logger
        self.path = path

    @abstractmethod
    def load_to_df(self) -> pd.DataFrame:
        """
        Load symbols to a dataframe
        """
        ...


class ELFToolsSymbolReader(ELFSymbolReader):
    """
    A symtab reader to dataframe.
    """
    def __init__(self, session, path):
        super().__init__(session, path)
        self.ef = ELFFile(open(path, "rb"))

    def is_dynamic(self):
        """
        Check whether the binary file is dynamically linked.
        """
        if self.ef.header.e_type == "ET_DYN":
            return True
        return False

    def load_to_df(self):
        """
        Load symbols from the target ELF file into a pandas dataframe
        """
        self.logger.debug("Load symbols for %s dyn=%s", self.path, self.is_dynamic())
        symtab = self.ef.get_section_by_name(".symtab")
        data_map = map(lambda s: (s.name, s["st_value"], s["st_size"], s["st_info"].type), symtab.iter_symbols())
        df = pd.DataFrame.from_records(data_map, columns=["name", "addr", "size", "type"])
        df = df.astype({"addr": np.uint64, "size": np.uint64, "type": str, "name": str})
        df["path"] = self.path
        df["dynamic"] = self.is_dynamic()
        return df


class LLVMSymbolReader(ELFSymbolReader):
    """
    Read symtabl using llvm tools instead of elftools
    """
    def __init__(self, session, path):
        super().__init__(session, path)
        self.nm = session.user_config.sdk_path / "sdk" / "bin" / "nm"
        et = ELFFile(open(path, "rb"))
        self.dynamic = et.header.e_type == "ET_DYN"

    def is_dynamic(self):
        return self.dynamic

    def load_to_df(self):
        self.logger.debug("Load symbols for %s", self.path)
        result = subprocess.run([self.nm, "--print-size", "--format", "posix", self.path],
                                capture_output=True,
                                text=True)
        if result.returncode != 0:
            self.logger.error("Failed to run nm %s: %s", self.path, result.stderr)
            raise RuntimeError("Failed to run nm")
        df = pd.read_csv(io.StringIO(result.stdout), delim_whitespace=True, names=["name", "type", "addr", "size"])
        df["path"] = self.path
        df["dynamic"] = self.dynamic
        # Normalize addr and size to integers
        df["addr"] = df["addr"].map(lambda v: int(v, 16))
        df["size"] = df["size"].map(lambda v: int(v, 16))
        df["type"] = df["type"].mask(df["type"].str.upper() == "T", "STT_FUNC")
        df["type"] = df["type"].mask(df["type"].str.upper() == "D", "STT_OBJECT")
        return df


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

        self.df = pd.DataFrame()
        self.df_funcs = SortedDict()

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

    def add_symbols(self, symbols_df: pd.DataFrame):
        """
        expect index {"path", "name", "addr"}
        expect columns {"dynamic", "size", "type"}
        from ELFSymbolReader
        """
        self.df = pd.concat([self.df, symbols_df.reset_index()])
        # update the bisect mapper to access rows
        fn_df = self.df.loc[self.df["type"] == "STT_FUNC"]
        self.df_funcs = SortedDict(zip(fn_df["addr"], fn_df.index))

    def lookup_function(self, addr):
        idx = self.df_funcs.bisect(addr) - 1
        if idx < 0:
            return SymInfo.unknown(addr)
        df_index = self.df_funcs.values()[idx]
        row = self.df.iloc[df_index]
        return SymInfo(row["name"], row["path"], row["addr"], row["size"])

    def lookup_function_df(self, addr: pd.DataFrame) -> pd.DataFrame:
        fn_df = self.df.loc[self.df["type"] == "STT_FUNC"]
        indices = fn_df.index.get_level_values("addr").searchsorted(addr, side="right")
        mapped = fn_df.iloc[indices - 1]
        mapped["addr"] = mapped.loc[:, "addr"].mask(indices == 0, np.nan)

        # Find last valid address
        # mapped["addr"] = mapped.loc[:, "addr"].mask(indices == len(fn_df), np.nan)
        def make_symbols(row):
            return SymInfo.unknown(addr)

        # elif idx < len(fn_df):
        #     return SymInfo(fn_df.iloc[idx - 1]["name"],
        #                    fn_df.iloc[idx - 1]["path"],
        #                    fn_df.iloc[idx - 1]["size"],
        #                    addr)
        # else:
        #     sym = SymInfo.unknown(addr)
        mapped["sym"] = mapped.apply(make_symbol, axis=1)
        mapped.index = addr.index
        return mapped

    def _test_lookup_function(self, addr: int) -> SymInfo:
        fn_df = self.df.loc[self.df["type"] == "STT_FUNCTION"]
        idx = fn_df.index.get_level_values("addr").searchsorted(addr, side="right")
        if idx < 1:
            return SymInfo.unknown(addr)
        elif idx < len(fn_df):
            return SymInfo(fn_df.iloc[idx - 1]["name"], fn_df.iloc[idx - 1]["path"], fn_df.iloc[idx - 1]["size"], addr)
        else:
            sym = SymInfo.unknown(addr)
        return sym

    def _old_lookup_function(self, addr: int) -> SymInfo:
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

    def register_address_space(self, as_key: str, shared: bool = False) -> Symbolizer:
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
        return symbolizer

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

    def lookup_function_df(self, as_key: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        temporary test implementation
        """
        sym_df = df.copy()
        sym_df["sym"] = None
        target_as = None
        if as_key in self.addrspaces:
            target_as = self.addrspaces[as_key]
            sym_df["sym"] = target_as.lookup_function_df(sym_df)
            if not sym_df["sym"].isna().any():
                # Resolved everything
                return sym_df
        # Try to resolve missing using fallbacks
        for curr_as in self.shared_addrspaces:
            if curr_as == target_as:
                continue
            sym_df_missing = sym_df["sym"].isna()
            sym_df.loc[sym_df_missing, "sym"] = curr_as.lookup_function_df(sym_df.loc[sym_df_missing])
            if not sym_df["sym"].isna().any():
                break
        return sym_df

    def lookup_function(self, as_key: str, addr: int, exact: bool = False) -> SymInfo:
        """
        Search for a function in the given address space. If exact is set, only return
        a valid symbol information if the given address matches exactly the function symbol address.

        :param as_key: The address space identifier
        :param addr: The address to lookup
        :param exact: Only look for exact matches.
        :return: Symbol information
        """
        target_symbolizer = None
        if as_key in self.addrspaces:
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
