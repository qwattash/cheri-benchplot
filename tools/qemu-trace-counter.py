#!/bin/python

import argparse as ap
import logging
import re
import subprocess
from pathlib import Path

import polars as pl
import polars.selectors as cs

INSN_LINE = re.compile(
    r"^\[0:[0-9]+\]\s+([0-9abcdefx]+):\s+([0-9abcdefx]+)\s+([a-z0-9.]+)\s+(.*)"
)
BASIC_LINE = re.compile(r"^\[0:[0-9]+\]")
CAP_OPERAND = re.compile(r"^c[zr0-9]+,")

logger = logging.getLogger("qemu-trace-counter")


class Bucket:
    def __init__(self, iclass):
        self.iclass = iclass
        self.count = 0

    @classmethod
    def from_match(cls, m):
        _opcode = m.group(2)
        mnemonic = m.group(3)
        operands = m.group(4)

        if mnemonic.startswith("ldr") or mnemonic == "ldp":
            iclass = "ld"
            if mnemonic == "ldp":
                iclass += "_pair"
            if CAP_OPERAND.match(operands):
                iclass += "_cap"
            else:
                iclass += "_int"
        elif mnemonic.startswith("str") or mnemonic == "stp":
            iclass = "st"
            if mnemonic == "stpp":
                iclass += "_pair"
            if CAP_OPERAND.match(operands):
                iclass += "_cap"
            else:
                iclass += "_int"
        elif mnemonic.startswith("scbnds"):
            iclass = "cheri"
        else:
            iclass = "other"

        return Bucket(iclass)


def filter_by_kernel_pc(pc, line_match):
    """
    Predicate to discard trace lines with non-kernel PCs.
    Return true if the entry is valid.
    """
    if pc <= 0x0001000000000000:
        return False
    return True


def collect_pc_hist(trace_file, line_filters):
    """
    Build a histogram for every PC hit in the trace.
    Produces a dataframe with columns pc, iclass, count
    """
    hist = {}

    for line in trace_file:
        if m := INSN_LINE.match(line):
            pc = int(m.group(1), 16)
            if line_filters:
                valid = all([fn(pc, m) for fn in line_filters])
                if not valid:
                    continue
            logger.debug("ENTRY %x: mnemonic='%s'", pc, m.group(3))
            bucket = hist.get(pc)
            if not bucket:
                bucket = Bucket.from_match(m)
                hist[pc] = bucket
            bucket.count += 1
        elif m := BASIC_LINE.match(line):
            logger.warning("Suspicious line mismatch '%s'", line)

    print("Collected", len(hist), "buckets")
    hist_df = pl.DataFrame(
        zip(
            hist.keys(),
            map(lambda b: b.iclass, hist.values()),
            map(lambda b: b.count, hist.values()),
        ),
        schema=["pc", "iclass", "count"],
    )

    return hist_df


def collect_symbols(obj_set):
    """
    Collect symbols from each object file.
    """
    syms = []

    for base, obj_path in obj_set:
        result = subprocess.Popen(
            ["llvm-nm", "-C", "-D", "-P", obj_path.expanduser()],
            stdout=subprocess.PIPE,
            text=False,
        )

        df = pl.read_csv(
            result.stdout,
            separator=" ",
            has_header=False,
            new_columns=["name", "flags", "addr", "size"],
            schema_overrides={"addr": pl.String, "size": pl.String},
        ).with_columns(
            pl.col("addr").str.to_integer(base=16, dtype=pl.UInt64) + base,
            pl.col("size").str.to_integer(base=16, dtype=pl.UInt64),
        )

        result.wait()
        syms.append(df)

    return pl.concat(syms)


def symbolize_fallback(df, obj_set):
    base = obj_set[0][0]
    obj = obj_set[0][1]

    addr2line = subprocess.Popen(
        f"llvm-addr2line -f --obj {obj.expanduser()} | paste -d ',' - -",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        shell=True,
    )

    fallback = df.filter(pl.col("name").is_null()).with_columns(pl.col("pc") - base)
    addr2line_data = fallback.select(pl.col("pc").map_elements(hex)).write_csv(
        include_header=False
    )
    stdout, _ = addr2line.communicate(input=addr2line_data.encode("utf-8"))

    symbols = (
        pl.read_csv(stdout, has_header=False)
        .with_columns(
            fallback["pc"],
            pl.when(pl.col("column_1").str.starts_with("??"))
            .then(None)
            .otherwise(pl.col("column_1"))
            .alias("fallback"),
            pl.when(pl.col("column_2").str.starts_with("??"))
            .then(None)
            .otherwise(pl.col("column_2"))
            .alias("location"),
        )
        .select(["pc", "fallback"])
    )

    print(
        "Fallback symbol resolution for",
        (~symbols["fallback"].is_null()).sum(),
        "buckets",
    )
    sym_df = (
        df.join(symbols, on="pc", how="left")
        .with_columns(pl.coalesce("name", "fallback").alias("name"))
        .select(cs.exclude("fallback"))
    )
    return sym_df


def symbolize_trace(trace_file, obj_set, out_file):
    base = obj_set[0][0]
    obj = obj_set[0][1]
    addr2line = subprocess.Popen(
        ["llvm-addr2line", "-f", "--obj", obj.expanduser()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

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
    parser.add_argument(
        "--obj", type=Path, help="Path to an object file for function annotation"
    )
    parser.add_argument(
        "--kernel-only",
        default=False,
        action="store_true",
        help="Filter out user space instructions",
    )
    parser.add_argument(
        "--skip-until", type=str, help="Skip entries until the given symbol is called"
    )
    parser.add_argument(
        "--skip-until-pc", type=str, help="Skip entries until the given address is hit"
    )
    parser.add_argument(
        "--skip-after", type=str, help="Record entries until the given symbol is called"
    )
    parser.add_argument(
        "--skip-after-pc",
        type=str,
        help="Record entries until the given address is hit",
    )
    parser.add_argument(
        "--skip-after-count",
        default=1,
        type=int,
        help="Stop after hitting the skip-after symbol the given number of times",
    )
    parser.add_argument(
        "--aggregate",
        default=False,
        action="store_true",
        help="Aggregate counts by symbol",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "output.csv",
        help="Output file name",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    obj_set = []
    if args.obj:
        obj_set.append((0, args.obj))

    line_filters = []
    if args.kernel_only:
        line_filters.append(filter_by_kernel_pc)

    symbols = collect_symbols(obj_set)

    trigger_state = {}

    def _skip_until(pc, target):
        if pc == target:
            trigger_state["skip_until"] = True
        return trigger_state["skip_until"]

    def _skip_after(pc, target):
        if pc == target:
            trigger_state["skip_after_count"] -= 1
            if trigger_state["skip_after_count"] <= 0:
                trigger_state["skip_after"] = False
        return trigger_state["skip_after"]

    if args.skip_until or args.skip_until_pc:
        if args.skip_until:
            target = symbols.filter(name=args.skip_until)
            if len(target) == 0:
                logger.error(
                    "skip-until: trigger symbol '%s' not found", args.skip_until
                )
                exit(1)
            target = target["addr"].first()
            logger.info("Skip entries until %s %x", args.skip_until, target)
        else:
            target = int(args.skip_until_pc, 16)
            logger.info("Skip entries until %x", target)
        trigger_state["skip_until"] = False
        line_filters.append(lambda pc, _: _skip_until(pc, target))

    if args.skip_after or args.skip_after_pc:
        if args.skip_after:
            target = symbols.filter(name=args.skip_after)
            if len(target) == 0:
                logger.error(
                    "skip-after: trigger symbol '%s' not found", args.skip_after
                )
                exit(1)
            target = target["addr"].first()
            logger.info(
                "Skip entries after %s %x %d times",
                args.skip_after,
                target,
                args.skip_after_count,
            )
        else:
            target = int(args.skip_after_pc, 16)
            logger.info("Skip entries after %x %d times", target, args.skip_after_count)
        trigger_state["skip_after"] = True
        trigger_state["skip_after_count"] = args.skip_after_count
        line_filters.append(lambda pc, _: _skip_after(pc, target))

    with open(args.trace_file, "r") as fd:
        hist_df = collect_pc_hist(fd, line_filters)

    # Assign symbols to buckets
    resolved = hist_df.join_where(
        symbols,
        pl.col("pc") < pl.col("addr") + pl.col("size"),
        pl.col("pc") >= pl.col("addr"),
    ).select(["pc", "name", "iclass", "count"])

    sym_df = hist_df.join(resolved, on="pc", how="left").select(
        ["pc", "name", "iclass", "count"]
    )
    print(
        "Initial symbolized frame resolved symbols for",
        (~sym_df["name"].is_null()).sum(),
        "buckets",
    )
    if sym_df["name"].is_null().sum() != 0:
        sym_df = symbolize_fallback(sym_df, obj_set)

    # Detect the function entrypoint buckets
    callsite_buckets = sym_df.join(symbols, left_on="pc", right_on="addr", how="inner")

    # Dump before aggregating anything
    sym_df.write_csv(args.output.with_suffix(".full"))

    # Finalize output
    if args.aggregate:
        sym_df = sym_df.group_by(["name"]).agg(pl.col("count").sum())
        sym_df = (
            sym_df.join(callsite_buckets, on="name")
            .with_columns(pl.col("count_right").alias("call_count"))
            .select(["name", "count", "call_count"])
        )
        sym_df = sym_df.sort(by=["count", "call_count", "name"])
    else:
        sym_df = sym_df.group_by(["name", "iclass"]).agg(pl.col("count").sum())
        sym_df = sym_df.sort(by=["count", "name", "iclass"])
        callsite_buckets.write_csv(args.output.with_suffix(".calls"))
    sym_df.write_csv(args.output)

    print("Total instructions matched", sym_df["count"].sum())

    # with open(args.trace_file, "r") as fd:
    #     with open(args.trace_file.with_suffix(".sym"), "w+") as out:
    #         symbolize_trace(fd, obj_set, out)


if __name__ == "__main__":
    main()
