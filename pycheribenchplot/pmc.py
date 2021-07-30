
import logging

import pandas as pd
import numpy as np

from .core.cpu import *
from .core.dataset import *


class PMCStatData:
    fields = [
        StrField("progname"),
        StrField("archname"),
    ]

    @classmethod
    def get_pmc_for_cpu(cls, cpu, options, benchmark):
        if cpu == BenchmarkCPU.FLUTE or cpu == BenchmarkCPU.QEMU_RISCV:
            return FluteStatcountersData(options, benchmark)
        elif cpu == BenchmarkCPU.TOOOBA:
            fatal("TODO")
        elif cpu == BenchmarkCPU.MORELLO:
            fatal("TODO")
        elif cpu == BenchmarkCPU.BERI:
            return BeriStatcountersData(options, benchmark)
        assert False, "Not Reached"

    def __init__(self, options, benchmark):
        self.options = options
        self.stats_df = None
        self._gen_extra_index = lambda df: benchmark.pmc_map_index(df)
        extra = self._gen_extra_index(pd.DataFrame(columns=["progname", "archname"]))
        self._extra_index = list(extra.columns)

    def raw_fields(self):
        return PMCStatData.fields

    def all_columns(self):
        return set(self.index_columns() + [f.name for f in self.raw_fields()])

    def index_columns(self):
        return ["__dataset_id"] + self._extra_index + ["sample_index"]

    def data_columns(self):
        return [f.name for f in self.raw_fields() if f.isdata]

    def valid_data_columns(self):
        return self.data_columns()

    def process(self):
        self._gen_composite_metrics()
        self._gen_stats()

    def add_statistic(self, new_df, prefix):
        new_df = new_df.add_prefix(prefix + "_")
        if self.stats_df is None:
            self.stats_df = new_df
        else:
            self.stats_df = pd.concat([self.stats_df, new_df], axis=1)

    def _import_df(self, csv_df):
        # Add sample indexes (could reuse the current index)
        csv_df["sample_index"] = np.arange(len(csv_df))

        # Generate extra index columns if any
        prog_and_arch = csv_df[["progname", "archname"]].copy()
        idx_df = self._gen_extra_index(prog_and_arch)
        if len(self._extra_index):
            csv_df[self._extra_index] = idx_df
        # Properly set the dataset index
        csv_df.set_index(self.index_columns(), inplace=True)

        csv_df = self._enforce_data_types(csv_df)
        valid_columns = set(csv_df.columns).intersection(
            set(self.all_columns()))
        self.df = pd.concat([self.df, csv_df[valid_columns]])


class FluteStatcountersData(PMCStatData):
    """
    Statcounters description and processing for CHERI Flute (RISC-V)
    """
    fields = [
        DataField("cycles"),
        DataField("instructions"),
        DataField("time"),
        DataField("redirect", "PC redirects"), # 0x01
        DataField("branch"), # 0x03
        DataField("jal"), # 0x04
        DataField("jalr"), # 0x05
        DataField("trap", "Stage 2 trap"), # 0x02
        DataField("auipc"), # 0x06
        DataField("load"), # 0x07
        DataField("store"), # 0x08
        DataField("LR"), # 0x09
        DataField("SC"), # 0x0a
        DataField("AMO"), # 0x0b
        DataField("serial_shift"), # 0x0c
        DataField("integer_muldiv", "Integer Mul/Div insn count"), # 0x0d
        DataField("FP", "Floating point insn count"), # 0x0e
        DataField("SC_success", "SC success"), # 0x0f
        DataField("load_wait", "Stage 2 cycles waiting on load"), # 0x10
        DataField("store_wait", "Stage 2 cycles waiting on store"), # 0x11
        DataField("fence", "Fence insn count"), # 0x12
        DataField("F_busy", "Cycles where stage F is busy"), # 0x13
        DataField("D_busy", "Cycles where stage D is busy"), # 0x14
        DataField("1_busy", "Cycles where stage 1 is busy"), # 0x15
        DataField("2_busy", "Cycles where stage 2 is busy"), # 0x16
        DataField("3_busy", "Cycles where stage 3 is busy"), # 0x17
        DataField("imprecise_setbounds", "Count setbounds NOT resulting in the exact bounds requested"), # 0x18
        DataField("cap_unrepresentable", "Count capability tag lost due to unrepresentable bounds"), # 0x19
        DataField("cap_load", "Stage 2 capability-wide load"), # 0x1a
        DataField("cap_store", "Stage 2 capability-wide store"), # 0x1b
        DataField("cap_load_tag_set", "Stage 2 loads tagged capability"), # 0x1c
        DataField("cap_store_tag_set", "Stage 2 stores tagged capability"), # 0x1d
        DataField("icache_load", "iCache load count"), # 0x20
        DataField("icache_load_miss", "iCache load missed"), # 0x21
        DataField("icache_load_miss_wait", "iCache load miss latency (cycles)"), # 0x22
        DataField("dcache_load", "dCache load count"), # 0x30
        DataField("dcache_load_miss", "dCache load missed"), # 0x31
        DataField("dcache_load_miss_wait", "dCache load miss latency (cycles)"), # 0x32
        DataField("icache_store", "iCache store count"), # 0x23
        DataField("icache_store_miss", "iCache store missed -- UNIMPL"),
        DataField("icache_store_miss_wait", "iCache store miss latency (cycles) -- UNIMPL"),
        DataField("dcache_store", "dCache store count"), # 0x33
        DataField("dcache_store_miss", "dCache store missed -- UNIMPL"),
        DataField("dcache_store_miss_wait", "dCache store miss latency (cycles) -- UNIMPL"),
        DataField("dcache_amo", "dCache atomic operation requested"), # 0x36
        DataField("dcache_amo_miss", "dCache atomic operation missed"), # 0x37
        DataField("dcache_amo_miss_wait", "dCache atomic operation miss latency (cycles)"), # 0x38
        DataField("itlb_access", "iTLB access count"), # 0x29
        DataField("itlb_miss", "iTLB miss count"), # 0x2a
        DataField("itlb_miss_wait", "iTLB miss latency (cycles)"), # 0x2b
        DataField("itlb_flush", "iTLB flush"), # 0x2c
        DataField("dtlb_access", "dTLB access count"),  # 0x39
        DataField("dtlb_miss", "dTLB miss count"), # 0x3a
        DataField("dtlb_miss_wait", "dTLB miss latency (cycles)"), # 0x3b
        DataField("dtlb_flush", "iTLB flush"), # 0x3c
        DataField("icache_evict", "iCache eviction count"), # 0x2d
        DataField("dcache_evict", "dCache eviction count"), # 0x3d
        DataField("llcache_load_miss", "last-level cache load missed"), # 0x61
        DataField("llcache_load_miss_wait", "last-level cache load miss latency (cycles)"), # 0x62
        DataField("llcache_evict", "last-level cache eviction count"), # 0x64
        DataField("tagcache_store", "tag cache store count"), # 0x40
        DataField("tagcache_store_miss", "tag cache store missed"), # 0x41
        DataField("tagcache_load", "tag cache load count"), # 0x42
        DataField("tagcache_load_miss", "tag cache load missed"), # 0x43
        DataField("tagcache_evict", "tag cache eviction count"), # 0x44
        DataField("tagcache_set_store", "tag cache set tag write"), # 0x45
        DataField("tagcache_set_load", "tag cache set tag read"), # 0x46
    ]

    def __init__(self, options, benchmark):
        super().__init__(options, benchmark)
        self.df = pd.DataFrame(columns=self.all_columns())
        self.df.set_index(self.index_columns(), inplace=True)
        self.df = self._enforce_data_types(self.df)

    def raw_fields(self):
        return super().raw_fields() + FluteStatcountersData.fields

    def valid_data_columns(self):
        tmp = self.df[self.data_columns()].dropna(axis=1, how="all")
        return list(tmp.columns)

    def load(self, dataset_id, filepath):
        csv_df = pd.read_csv(filepath)
        csv_df["__dataset_id"] = dataset_id
        self._import_df(csv_df)

    def _enforce_data_types(self, df):
        coltypes = {f.name: f.dtype for f in self.raw_fields()
                    if f.name in df.columns}
        return df.astype(coltypes, copy=True)

    def _gen_composite_metrics(self):
        new_columns = {}
        for c in self.valid_data_columns():
            if c.endswith("_miss"):
                base_metric = c[:-len("_miss")]
                if base_metric not in self.df.columns:
                    base_metric = base_metric + "_access"
                    if base_metric not in self.df.columns:
                        logging.warning("Can not determine base column for %s, skipping", c)
                        continue

                hit = self.df[base_metric] - self.df[c]
                if ((hit >= 0) | hit.isna()).all():
                    hit_rate = hit / self.df[base_metric]
                else:
                    if (self.df[base_metric] == 0).all():
                        logging.warning("Invalid hit count for %s, maybe %s is unimplemented",
                                        c, base_metric)
                        hit_rate = np.NaN
                    else:
                        logging.critical("Negative hit count %s".format(base_metric))
                        exit(1)
                new_columns["{}_hit_rate".format(base_metric)] = hit_rate
        self.df = self.df.assign(**new_columns)

    def _gen_stats(self):
        # Median and quartiles by index (except sample_index)
        group_index = self.index_columns()
        group_index.remove("sample_index")
        grouped = self.df[self.valid_data_columns()].groupby(level=group_index)
        self.add_statistic(grouped.median(), "median")
        self.add_statistic(grouped.quantile(q=0.25), "q25")
        self.add_statistic(grouped.quantile(q=0.75), "q75")
        # Compute error columns from median and quartiles
        for col in self.valid_data_columns():
            data_col = "median_{}".format(col)
            q75_col = "q75_{}".format(col)
            q25_col = "q25_{}".format(col)
            err_hi = self.stats_df[q75_col] - self.stats_df[data_col]
            err_lo = self.stats_df[data_col] - self.stats_df[q25_col]
            self.stats_df["errhi_{}".format(col)] = err_hi
            self.stats_df["errlo_{}".format(col)] = err_lo


class BeriStatcountersData:
    """
    Statcounters description and processing for CHERI-BERI
    """

    data_columns = [
        "cycles", "instructions", "dcache_write", "dcache_read",
        "l2cache_write", "l2cache_read", "tagcache_write", "tagcache_read"]
    plot_columns = [
        "cycles", "instructions", "cpi", "dtlb_miss", "itlb_miss",
        "icache_fetch_icount",
        "tagcache_w_miss_rate", "tagcache_r_miss_rate",
        "dcache_w_miss_rate", "dcache_r_miss_rate",
        "icache_r_miss_rate",
        "dcache_set_tag_write", "dcache_set_tag_read",
        "l2cache_w_miss_rate", "l2cache_r_miss_rate",
        "l2cache_set_tag_write", "l2cache_set_tag_read",
        'mipsmem_byte_read', 'mipsmem_byte_write',
        'mipsmem_hword_read', 'mipsmem_hword_write',
        'mipsmem_word_read', 'mipsmem_word_write',
        'mipsmem_dword_read', 'mipsmem_dword_write',
        'mipsmem_cap_read', 'mipsmem_cap_write',
        'mipsmem_cap_read_tag_set', 'mipsmem_cap_write_tag_set',
        'total_mem_read', 'mem_r_op_per_byte',
        'total_mem_write', 'mem_w_op_per_byte',
#        'kern_unaligned_access', 'kern_unaligned_access_per_insn'
    ]
    plot_descriptions = [
        "cycles", "committed instructions", "cycles per instr", "dTLB miss", "iTLB miss",
        "committed instructions / icache fetch",
        "tagcache W miss rate", "tagcache R miss rate",
        "dcache W miss rate", "dcache R miss rate",
        "icache R miss rate",
        "dcache set tag W", "dcache set tag R", "l2cache W miss rate",
        "l2cache R miss rate", "l2cache set tag W", "l2cache set tag R",
        "uArch byte R", "uArch byte W", "uArch hword R", "uArch hword W",
        "uArch word R", "uArch word W", "uArch dword R", "uArch dword W",
        "uArch cap R", "uArch cap W", "uArch tagged cap R", "uArch tagged cap W",
        "uArch total R", "uArch total R / total bytes R",
        "uArch total W", "uArch total W / total bytes W",
#        "Kern unaligned access", "Kern unaligned access per instr"
    ]

    def __init__(self, input_df, baseline_arch):
        self.baseline_arch = baseline_arch
        if not "progname" in input_df.index.names:
            input_df = input_df.set_index(["progname"], append=True)
        if not "archname" in input_df.index.names:
            input_df = input_df.set_index(["archname"], append=True)
        self.df = input_df

        self.make_composite_metrics()

    def make_composite_metrics(self):
        assert "dcache_write" in BeriStatcountersData.data_columns
        assert "dcache_read" in BeriStatcountersData.data_columns
        self.df = self.df.assign(
            dcache_w_hit_rate = hit_rate(self.df, "dcache_write"),
            dcache_r_hit_rate = hit_rate(self.df, "dcache_read"),
            icache_w_hit_rate = hit_rate(self.df, "icache_write"),
            icache_r_hit_rate = hit_rate(self.df, "icache_read"),
            l2cache_w_hit_rate = hit_rate(self.df, "l2cache_write"),
            l2cache_r_hit_rate = hit_rate(self.df, "l2cache_read"),
            tagcache_w_hit_rate = hit_rate(self.df, "tagcache_write"),
            tagcache_r_hit_rate = hit_rate(self.df, "tagcache_read"))
        # account for never having a single hit/miss on the write side
        self.df["icache_w_hit_rate"].fillna(1, inplace=True)

        self.df = self.df.assign(
            dcache_w_miss_rate = 1 - self.df["dcache_w_hit_rate"],
            dcache_r_miss_rate = 1 - self.df["dcache_r_hit_rate"],
            icache_w_miss_rate = 1 - self.df["icache_w_hit_rate"],
            icache_r_miss_rate = 1 - self.df["icache_r_hit_rate"],
            l2cache_w_miss_rate = 1 - self.df["l2cache_w_hit_rate"],
            l2cache_r_miss_rate = 1 - self.df["l2cache_r_hit_rate"],
            tagcache_w_miss_rate = 1 - self.df["tagcache_w_hit_rate"],
            tagcache_r_miss_rate = 1 - self.df["tagcache_r_hit_rate"])

        total_read_op = (self.df["mipsmem_byte_read"] +
                         self.df["mipsmem_hword_read"] +
                         self.df["mipsmem_word_read"] +
                         self.df["mipsmem_dword_read"] +
                         self.df["mipsmem_cap_read"])
        total_bytes_read = (
            self.df["mipsmem_byte_read"] +
            self.df["mipsmem_hword_read"] * 2 +
            self.df["mipsmem_word_read"] * 4 +
            self.df["mipsmem_dword_read"] * 8 +
            self.df["mipsmem_cap_read"] * 16)
        total_write_op = (self.df["mipsmem_byte_write"] +
                         self.df["mipsmem_hword_write"] +
                         self.df["mipsmem_word_write"] +
                         self.df["mipsmem_dword_write"] +
                         self.df["mipsmem_cap_write"])
        total_bytes_write = (
            self.df["mipsmem_byte_write"] +
            self.df["mipsmem_hword_write"] * 2 +
            self.df["mipsmem_word_write"] * 4 +
            self.df["mipsmem_dword_write"] * 8 +
            self.df["mipsmem_cap_write"] * 16)

        self.df = self.df.assign(
            total_mem_read = total_read_op,
            total_mem_write = total_write_op,
            mem_r_op_per_byte = total_read_op / total_bytes_read,
            mem_w_op_per_byte = total_write_op / total_bytes_write)

        self.df = self.df.assign(
            # instruction cache fetches over instruction count
            # give a metric of the committed instructions ratio
            icache_fetch_icount = self.df["instructions"] / (self.df["icache_read_hit"] + self.df["icache_read_miss"]),
            cpi = self.df["cycles"] / self.df["instructions"])

        self.df.drop(["kern_unaligned_access"], axis=1)
        # self.df["kern_unaligned_access"].fillna(0, inplace=True)
        # kunaligned_per_insn = self.df["kern_unaligned_access"] / self.df["inst_kernel"]
        # self.df = self.df.assign(kern_unaligned_access_per_insn = kunaligned_per_insn)

    def add_statistic(self, df, prefix, stats_df=None):
        new_df = df.add_prefix(prefix + "_")
        if stats_df is None:
            return new_df
        else:
            return pd.concat((stats_df, new_df), axis=1)

    def baseline_set(self, df):
        return df.xs(self.baseline_arch, level="archname")

    def data_set(self, df):
        match = (df.get_level_values("archname") != self.baseline_arch)
        return df[match].droplevel("archname")

    def aggregate_statistics(self):
        # median and quartiles by progname and archname
        grouped = self.df.groupby(level=["progname", "archname"])
        median = grouped.median()
        q25 = grouped.quantile(q=0.25)
        q75 = grouped.quantile(q=0.75)
        stats_df = self.add_statistic(median, "median")
        stats_df = self.add_statistic(q25, "q25", stats_df)
        stats_df = self.add_statistic(q75, "q75", stats_df)

        # absolute overhead and relative error
        b_median = self.baseline_set(median)
        b_q25 = self.baseline_set(q25)
        b_q75 = self.baseline_set(q75)
        abs_delta_median = median.groupby("archname").apply(
            lambda grp: grp - b_median)
        abs_delta_median_err_hi = q75.groupby("archname").apply(
            lambda grp: grp - b_q25)
        abs_delta_median_err_lo = q25.groupby("archname").apply(
            lambda grp: grp - b_q75)
        stats_df = self.add_statistic(
            abs_delta_median, "delta_median", stats_df)
        stats_df = self.add_statistic(
            abs_delta_median_err_hi, "delta_median_err_hi", stats_df)
        stats_df = self.add_statistic(
            abs_delta_median_err_lo, "delta_median_err_lo", stats_df)
        # normalized median and quartiles w.r.t. baseline arch
        norm_delta_median = median.groupby("archname").apply(
            lambda grp: 100 * (grp - b_median) / b_median)
        norm_delta_median_err_hi = q75.groupby("archname").apply(
            lambda grp: 100 * (grp - b_median) / b_median)
        norm_delta_median_err_lo = q25.groupby("archname").apply(
            lambda grp: 100 * (grp - b_median) / b_median)
        # account for never having a single hit/miss on the write side
        norm_delta_median["icache_w_miss_rate"].fillna(0, inplace=True)
        norm_delta_median_err_hi["icache_w_miss_rate"].fillna(0, inplace=True)
        norm_delta_median_err_lo["icache_w_miss_rate"].fillna(0, inplace=True)
        stats_df = self.add_statistic(
            norm_delta_median, "norm_delta_median", stats_df)
        stats_df = self.add_statistic(
            norm_delta_median_err_hi, "norm_delta_median_err_hi", stats_df)
        stats_df = self.add_statistic(
            norm_delta_median_err_lo, "norm_delta_median_err_lo", stats_df)
        # mean and std by progname and archname
        # self.add_statistic(grouped.mean(), "mean")
        # self.add_statistic(grouped.std(), "std")
        # self.add_statistic(grouped.var(), "var")
        return stats_df
