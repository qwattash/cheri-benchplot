#!/bin/python

import argparse as ap
import json
import shutil
import subprocess
import re
from collections import defaultdict
from pathlib import Path

import polars as pl

INSN_LINE = re.compile(r"^\[0:[0-9]+\]\s+([0-9abcdefx]+):\s+([0-9abcdefx]+)\s+([a-z0-9]+)")
SECTION_LINE = re.compile(r"^// SECTION\[([ 0-9a-zA-Z_-]+)\]")
ENDSECTION_LINE = re.compile(r"^// ENDSECTION")

addr2line = shutil.which("llvm-addr2line")

def collect_sections(trace_file):
    sections = {}
    active = []
    insn = 0
    for line in trace_file:
        if INSN_LINE.match(line):
            insn += 1
        elif m := SECTION_LINE.match(line):
            name = m.group(1)
            active.append((name, insn))
        elif ENDSECTION_LINE.match(line):
            name, start = active.pop()
            sections[name] = insn - start

    sections["sections total"] = sum([n for n in sections.values()])
    sections["total"] = insn

    return sections

def collect_functions(trace_file, cumulative=False):
    fn_icount = defaultdict(lambda: 0)
    fn_icount["_top_"] = 0
    calls = {}
    stack = []
    global_icount = 0
    icount = 0
    has_call = False
    has_ret = False

    for line in trace_file:
        if m := INSN_LINE.match(line):
            pc = m.group(1)
            if has_call:
                stack.append((pc, global_icount))
                has_call = False
                fn_icount[pc] = 0
            elif has_ret:
                pc, start = stack.pop()
                calls[pc] = global_icount - start
                has_ret = False
                fn_icount[pc] += icount
                icount = 0
            global_icount += 1
            icount += 1
            opcode = m.group(2)
            mnemonic = m.group(3)
            if mnemonic == "blr" or mnemonic == "bl" or mnemonic == "svc":
                has_call = True
                curr_frame = "_top_" if len(stack) == 0 else stack[-1][0]
                fn_icount[curr_frame] += icount
                icount = 0
            elif mnemonic == "ret" or mnemonic == "eret" or opcode == "d69f03e0":
                has_ret = True
    calls["_top_"] = global_icount
    fn_icount["_top_"] += icount
    cum_icount_df = pl.from_records([list(calls.keys()), list(calls.values())], schema=["fn", "cum_icount"])
    fn_icount_df = pl.from_records([list(fn_icount.keys()), list(fn_icount.values())], schema=["fn", "icount"])
    return cum_icount_df.join(fn_icount_df, on="fn")


def resolve_symbol(obj_set: list[Path], addr: str) -> str:
    try:
        addr = int(addr, 16)
    except ValueError:
        return addr

    for base, obj_path in obj_set:
        if base is not None:
            addr = addr - base
        result = subprocess.run(["llvm-addr2line", "-f", "--obj", obj_path.expanduser(), f"{addr:x}"],
                                capture_output=True, check=True)
        out = result.stdout.decode("UTF-8").splitlines()
        fn = out[0]
        print("Lookup symbol", obj_path.expanduser(), f"{addr:x}", "->", fn)
        if fn == "??":
            return f"0x{addr:x}"
        return fn


def main():
    parser = ap.ArgumentParser("QEMU instruction trace tool")
    parser.add_argument("trace_file", type=Path, help="Trace file to inspect")
    parser.add_argument("--obj", type=Path, help="Path to an object file for function annotation")
    parser.add_argument("--cumulative", default=False, action="store_true",
                        help="Cumulative instruction count for function calls")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "output.csv", help="Output file name")

    args = parser.parse_args()

    print("Using llvm-addr2line:", addr2line)

    obj_set = []
    if args.obj:
        obj_set.append((0, args.obj))

    # with open(args.trace_file, "r") as fd:
    #     sections = collect_sections(fd)
    #     print(json.dumps(sections, indent=4))

    with open(args.trace_file, "r") as fd:
        hist_df = collect_functions(fd, args.cumulative)

        sym_df = hist_df.with_columns(
            pl.col("fn").map_elements(lambda addr: resolve_symbol(obj_set, addr), return_dtype=pl.String)
        )
        sym_df.write_csv(args.output)
        print(sym_df)

if __name__ == "__main__":
    main()
