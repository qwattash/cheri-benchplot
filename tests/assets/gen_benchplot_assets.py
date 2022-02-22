import logging
from pathlib import Path

from pycheribenchplot.core.elf import DWARFInfoSource

logger = logging.getLogger("asset-gen")


def gen_dwarf_assets():
    dw = DWARFInfoSource(logger, Path.cwd() / "test_dwarf_nested_structs", 8)
    dw.parse_dwarf()
    data = dw.get_dwarf_data()
    df = data.get_struct_info()
    df.to_csv(Path.cwd() / "test_dwarf_nested_structs.csv")


def main():
    gen_dwarf_assets()


if __name__ == "__main__":
    main()
