#!/usr/bin/env python3

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl
import polars.selectors as cs

rules = {
    "dev": r"^sys/dev/.*",
    "vm": r"^sys/vm/.*",
    "kern": r"^sys/kernel/.*",
    "compat": r"^sys/compat/.*",
    "headers": r"^sys/sys/.*",
    "contrib": r"^sys/contrib/.*",
    "fs": r"^(sys/u?fs/.*|sys/cam/.*)",
    "net": r"^sys/net.*",
}

ignore = ["^.git/.*", "^tools/.*", "^sys/sys/cdefs.h"]

annotations = [
    "__subobject_use_container_bounds|__subobject_variable_length",
    "__subobject_use_remaining_size|__subobject_member_used_for_c_inheritance",
    "__no_subobject_bounds",
    "__bounded_addressof",
]


def do_scan(src: Path, annotation: str):
    print(f"Count {annotation} annotations")
    locations = defaultdict(lambda: 0)

    result = subprocess.run(
        ["grep", "-E", "-r", annotation, src], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        path = Path(line.split(":")[0])
        rel = path.relative_to(src)
        locations[str(rel)] += 1

    df = pl.DataFrame({"file": locations.keys(), "count": locations.values()})
    df = df.with_columns(pl.lit(annotation).alias("annotation"))
    return df


def main():
    parser = argparse.ArgumentParser(description="Find sub-object bounds annotations.")
    parser.add_argument("src_directory", type=Path, help="Source directory")
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default=Path("./subobject-annotations.csv"),
        type=Path,
        help="Output CSV data.",
    )

    args = parser.parse_args()

    src = args.src_directory
    if not src.is_dir():
        print(f"Error: The specified input path '{src}' is not a valid directory.")
        sys.exit(1)

    print("Scanning C files")
    all_df = []
    for ann in annotations:
        df = do_scan(src, ann)
        all_df.append(df)
    df = pl.concat(all_df)

    # Now assign components to file names
    batches = [
        pl.when(pl.col("file").str.contains(pattern))
        .then(pl.lit(component))
        .alias("component")
        for component, pattern in rules.items()
    ]
    df = df.with_columns(component=pl.coalesce(batches).fill_null("other"))
    df = df.filter(~pl.col("file").str.contains("|".join(ignore)))
    print("Total opt-out annotations:", df["count"].sum())

    df = df.group_by(["annotation", "component"], maintain_order=True).agg(
        pl.col("count").sum()
    )
    df = df.pivot(
        index="component", on="annotation", values="count", maintain_order=True
    ).fill_null(0)
    total = df.select(pl.lit("total").alias("component"), cs.exclude("component").sum())
    df = df.select(["component", *annotations]).sort(by=["component"])
    df = pl.concat([df, total])
    df.write_csv(args.output)

    print(df)


if __name__ == "__main__":
    main()
