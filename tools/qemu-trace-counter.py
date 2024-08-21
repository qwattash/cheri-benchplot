#!/bin/python

import argparse as ap
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import polars as pl
import polars.selectors as cs

INSN_LINE = re.compile(r"^\[0:[0-9]+\]\s+([0-9abcdefx]+):\s+([0-9abcdefx]+)\s+([a-z0-9]+)\s+(.*)")
CAP_OPERAND = re.compile(r"^c[zr0-9]+,")

addr2line = shutil.which("llvm-addr2line")


class ICount:
    def __init__(self):
        self.icount = {
            "all": 0,
            "cheri": 0,
            # all single-operand load/store
            "ld": 0,
            "st": 0,
            # all pair load/store
            "ld_pair": 0,
            "st_pair": 0,
            # pair load/store with capability registers
            "cheri_ld_pair": 0,
            "cheri_st_pair": 0,
            # pair load/store with integer registers
            "int_ld_pair": 0,
            "int_st_pair": 0,
            # adrp instructions, should correlate to # of GOT accesses
            "adrp": 0,
        }

    def __add__(self, other):
        r = ICount()
        r += other
        return r

    def __sub__(self, other):
        r = ICount()
        for cat, count in self.icount.items():
            r.icount[cat] = count - other.icount[cat]
        return r

    def __iadd__(self, other):
        for cat, count in self.icount.items():
            self.icount[cat] += other.icount[cat]
        return self

    def increment(self, opcode, mnemonic, operands):
        self.icount["all"] += 1

        if mnemonic.startswith("ldr") or mnemonic == "ldp":
            if mnemonic == "ldp":
                self.icount["ld_pair"] += 1
                if CAP_OPERAND.match(operands):
                    self.icount["cheri_ld_pair"] += 1
                else:
                    self.icount["int_ld_pair"] += 1
            else:
                self.icount["ld"] += 1
        elif mnemonic.startswith("str") or mnemonic == "stp":
            if mnemonic == "stp":
                self.icount["st_pair"] += 1
                if CAP_OPERAND.match(operands):
                    self.icount["cheri_st_pair"] += 1
                else:
                    self.icount["int_st_pair"] += 1
            else:
                self.icount["st"] += 1
        elif mnemonic.startswith("scbnds"):
            self.icount["cheri"] += 1
        elif mnemonic == "adrp":
            self.icount["adrp"] += 1

    def clone(self):
        r = ICount()
        r.icount = dict(self.icount)
        return r


def collect_functions(trace_file, cumulative=False):
    fn_icount = defaultdict(ICount)
    fn_icount["_top_"] = ICount()
    calls = {}
    stack = []
    global_icount = ICount()
    icount = ICount()
    has_call = False
    has_ret = False

    for line in trace_file:
        if m := INSN_LINE.match(line):
            pc = m.group(1)
            if has_call:
                stack.append((pc, global_icount.clone()))
                has_call = False
                fn_icount[pc] = ICount()
            elif has_ret:
                pc, start = stack.pop()
                calls[pc] = global_icount - start
                has_ret = False
                fn_icount[pc] += icount
                icount = ICount()

            opcode = m.group(2)
            mnemonic = m.group(3)
            operands = m.group(4)
            global_icount.increment(opcode, mnemonic, operands)
            icount.increment(opcode, mnemonic, operands)

            if mnemonic == "blr" or mnemonic == "bl" or mnemonic == "svc":
                has_call = True
                curr_frame = "_top_" if len(stack) == 0 else stack[-1][0]
                fn_icount[curr_frame] += icount
                icount = ICount()
            elif mnemonic == "ret" or mnemonic == "eret" or opcode == "d69f03e0":
                has_ret = True

    calls["_top_"] = global_icount
    fn_icount["_top_"] += icount
    cumul_icount_df = (pl.from_records([list(calls.keys()), [i.icount for i in calls.values()]],
                                       schema=["fn", "cumul_icount"]).unnest("cumul_icount").select(
                                           pl.col("fn"),
                                           cs.all().exclude("fn").name.prefix("cumul_")))
    fn_icount_df = (pl.from_records([list(fn_icount.keys()), [i.icount for i in fn_icount.values()]],
                                    schema=["fn", "icount"]).unnest("icount"))
    return cumul_icount_df.join(fn_icount_df, on="fn")


def resolve_symbol(obj_set: list[Path], addr: str) -> str:
    try:
        addr = int(addr, 16)
    except ValueError:
        return addr

    for base, obj_path in obj_set:
        if base is not None:
            addr = addr - base
        result = subprocess.run(
            ["llvm-addr2line", "-f", "--obj", obj_path.expanduser(), f"{addr:x}"], capture_output=True, check=True)
        out = result.stdout.decode("UTF-8").splitlines()
        fn = out[0]
        print("Lookup symbol", obj_path.expanduser(), f"{addr:x}", "->", fn)
        if fn == "??":
            return f"0x{addr:x}"
        return fn


def symbolize_trace(trace_file, obj_set, out_file):
    base = obj_set[0][0]
    obj = obj_set[0][1]
    addr2line = subprocess.Popen(["llvm-addr2line", "-f", "--obj", obj.expanduser()],
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE)

    def resolve_location(addr):
        addr = int(addr, 16)
        if base is not None:
            addr = addr - base

        addr2line.stdin.write(f"{addr:x}\n".encode("UTF-8"))
        addr2line.stdin.flush()
        fn = addr2line.stdout.readline().decode("UTF-8").strip()
        location = addr2line.stdout.readline().decode("UTF-8").strip()
        return f"{location}//{fn}"

    for line in trace_file:
        if m := INSN_LINE.match(line):
            pc = m.group(1)
            location = resolve_location(pc)
            out_file.write(f"{location} {line}\n")

    addr2line.stdin.close()
    addr2line.wait()


def main():
    parser = ap.ArgumentParser("QEMU instruction trace tool")
    parser.add_argument("trace_file", type=Path, help="Trace file to inspect")
    parser.add_argument("--obj", type=Path, help="Path to an object file for function annotation")
    parser.add_argument("--cumulative",
                        default=False,
                        action="store_true",
                        help="Cumulative instruction count for function calls")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "output.csv", help="Output file name")

    args = parser.parse_args()

    print("Using llvm-addr2line:", addr2line)

    obj_set = []
    if args.obj:
        obj_set.append((0, args.obj))

    with open(args.trace_file, "r") as fd:
        hist_df = collect_functions(fd, args.cumulative)

        sym_df = hist_df.with_columns(
            pl.col("fn").map_elements(lambda addr: resolve_symbol(obj_set, addr), return_dtype=pl.String))
        sym_df.write_csv(args.output)
        print(sym_df)

    with open(args.trace_file, "r") as fd:
        with open(args.trace_file.with_suffix(".sym"), "w+") as out:
            symbolize_trace(fd, obj_set, out)


if __name__ == "__main__":
    main()
