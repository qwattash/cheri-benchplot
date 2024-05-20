import argparse as ap
import itertools as it
from pathlib import Path

from pycheribenchplot.iperf.iperf_exec import (IPerfMode, IPerfProtocol, IPerfScenario, IPerfTransferLimit)

DEFAULT = None

ABBREV_MODES = {
    IPerfMode.CLIENT_SEND: "send",
    IPerfMode.CLIENT_RECV: "recv",
    IPerfMode.BIDIRECTIONAL: "bidir",
}


def check_mss(value):
    if value.endswith("K"):
        mss = int(value[:-1]) * 2**10
    elif value.endswith("M"):
        mss = int(value[:-1]) * 2**20
    else:
        mss = int(value)
    return mss


def main():
    parser = ap.ArgumentParser("IPerf scenario generator")

    parser.add_argument("-p", type=IPerfProtocol, help="Protocol, one of tcp, udp and smtp")
    parser.add_argument("-l",
                        "--limit-mode",
                        type=IPerfTransferLimit,
                        help="Transfer limit mode, one of bytes, time, pkts")
    parser.add_argument("-m",
                        "--mode",
                        type=IPerfMode,
                        help="Transfer mode, one of client-send, client-recv and bidirectional")
    parser.add_argument("-n", "--limit", nargs="+", type=str, help="Transfer limit (depends on -l)")
    parser.add_argument("-s", type=int, help="Number of streams")
    parser.add_argument("-b", "--buffer-size", nargs="+", type=str, help="Send/recv buffer size")
    parser.add_argument("--mss", nargs="+", type=check_mss, help="MSS size")
    parser.add_argument("--window", nargs="+", type=str, help="Socket buffer size")
    parser.add_argument("--affinity", type=str, help="CPU affinity")
    parser.add_argument("-r", type=str, help="Remote host, default localhost")

    parser.add_argument("--pretend", action="store_true", default=False, help="Do not generate files")

    parser.add_argument("scenario_dir", type=Path, help="Destination path")

    args = parser.parse_args()

    scenario_file_pattern = "{protocol}_{mode}_{limit}"
    if args.s:
        scenario_file_pattern += "_s{streams}"
    if args.window:
        scenario_file_pattern += "_w{window_size}"
    if args.mss:
        scenario_file_pattern += "_m{mss}"
    if args.buffer_size:
        scenario_file_pattern += "_{buffer_size}"

    for (limit, window, mss, size) in it.product(args.limit or [DEFAULT], args.window or [DEFAULT], args.mss
                                                 or [DEFAULT], args.buffer_size or [DEFAULT]):
        kwargs = dict(protocol=args.p,
                      transfer_mode=args.limit_mode,
                      mode=args.mode,
                      transfer_limit=limit,
                      streams=args.s,
                      window_size=window,
                      mss=mss,
                      buffer_size=size,
                      remote_host=args.r,
                      cpu_affinity=args.affinity)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        scenario = IPerfScenario(**kwargs)

        file_name = scenario_file_pattern.format(protocol=scenario.protocol.value,
                                                 mode=ABBREV_MODES[scenario.mode],
                                                 limit=scenario.transfer_limit,
                                                 streams=scenario.streams,
                                                 window_size=scenario.window_size,
                                                 mss=scenario.mss,
                                                 buffer_size=scenario.buffer_size)
        dst_file = (args.scenario_dir / file_name).with_suffix(".json")
        print("Emit scenario", file_name, "to", dst_file)
        if args.pretend:
            print(scenario.emit_json())
        else:
            with open(dst_file, "w+") as fd:
                fd.write(scenario.emit_json())


if __name__ == "__main__":
    main()
