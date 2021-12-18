import argparse as ap
import struct
from pathlib import Path

import protobuf_backend_entry_pb2 as qemu_proto


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
            log_entry = qemu_proto.QEMULogInstrEntry()
            log_entry.ParseFromString(data)
            yield log_entry


def main():
    parser = ap.ArgumentParser("cheri_trace_processor")
    parser.add_argument("qemu_protobuf", type=Path)

    args = parser.parse_args()
    if not args.qemu_protobuf.exists():
        print(f"File does not exist {args.qemu_protobuf}")
        exit(1)

    with open(args.qemu_protobuf, "rb") as pb_file:
        tp = CheriTraceProcessor(pb_file)
        for entry in tp:
            print(f"[{entry.cpu}:{entry.asid}] {entry.pc:#08x} {entry.disas}")


if __name__ == "__main__":
    main()
