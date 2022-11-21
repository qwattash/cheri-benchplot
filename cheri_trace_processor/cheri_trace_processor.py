import argparse as ap
import struct
from pathlib import Path

import qemu_log_entry_pb2 as qemu_proto
from sortedcontainers import SortedList


class Interval:
    def __init__(self, l, r):
        self.l = l
        self.r = r

    def __lt__(self, other):
        return (self.l, self.r) < (other.l, other.r)

    def __eq__(self, other):
        return (self.l, self.r) == (other.l, other.r)

    def __str__(self):
        return f"({self.l:#08x}, {self.r:#08x})"


class CheriTraceProcessor:
    def __init__(self, pb_file):
        self.pb_file = pb_file

    def __iter__(self):
        """
        Read entries from the protobuf file
        """
        self.pb_file.seek(0)
        while True:
            # Preamble is always uint32
            preamble = self.pb_file.read(4)
            if len(preamble) == 0:
                break
            data_len = struct.unpack("@I", preamble)[0]
            data = self.pb_file.read(data_len)
            assert len(data) == data_len, f"Truncated data: expected {data_len} found {len(data)}"
            log_entry = qemu_proto.QEMULogEntry()
            log_entry.ParseFromString(data)
            yield log_entry

    def find_pc_ranges(self):
        # Ranges are mapped as start => [start, end]
        # Overlapping intervals are merged, so the ranges are guaranteed to be disjoint
        pc_ranges = SortedList()

        print("Scanning trace...")
        elapsed = 0
        for entry in self:
            if elapsed % 10000 == 0:
                print("N =", elapsed)
            elapsed += 1

            i = Interval(entry.pc, entry.pc + 4)
            idx = pc_ranges.bisect_left(i)
            prev_idx = idx - 1
            next_idx = idx + 1
            pc_ranges.add(i)

            # Check if we can merge left
            if prev_idx >= 0:
                prev = pc_ranges[prev_idx]
                if prev.r >= i.l:
                    prev.r = max(prev.r, i.r)
                    # Drop i and prelace it with prev
                    pc_ranges.discard(i)
                    next_idx -= 1
                    i = prev
            # Check if we can merge right
            if next_idx < len(pc_ranges):
                nxt = pc_ranges[next_idx]
                if nxt.l <= i.r:
                    nxt.l = min(nxt.l, i.l)
                    # Drop i, we replaced it with nxt
                    pc_ranges.discard(i)
        return pc_ranges


def main():
    parser = ap.ArgumentParser("cheri_trace_processor")
    parser.add_argument("qemu_protobuf", type=Path)
    parser.add_argument("--list", action="store_true", help="Display all instructions in the trace")
    parser.add_argument("--pc-ranges", action="store_true", help="Display the ranges of PC in the trace")

    args = parser.parse_args()
    if not args.qemu_protobuf.exists():
        print(f"File does not exist {args.qemu_protobuf}")
        exit(1)

    with open(args.qemu_protobuf, "rb") as pb_file:
        tp = CheriTraceProcessor(pb_file)

        if args.list:
            for entry in tp:
                print(f"[{entry.cpu}:{entry.asid}] {entry.pc:#08x} {entry.disas}")
        if args.pc_ranges:
            intervals = tp.find_pc_ranges()
            for i in intervals:
                print(f"PC: {i.l:#08x}-{i.r:#08x}")


if __name__ == "__main__":
    main()
