import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cxxfilt
import networkx as nx
import pandas as pd
import polars as pl

from ..core.artefact import (BenchmarkIterationTarget, PLDataFrameLoadTask, Target)
from ..core.config import Config, ConfigPath
from ..core.task import DataGenTask, output


@dataclass
class IngestPMCStatStacksConfig(Config):
    """
    Configuration for the task that loads pmcstat stacks files.
    """
    #: Path of the stacks files to ingest
    path: ConfigPath
    #: Stacks file pattern
    #: Use benchmark parameter templates to substitute the name for each
    #: different benchmark variation here. Note that a special {iteration}
    #: template key is used to substitute the iteration number.
    stack_file_pattern: str = "pmcstat.{iteration}.stacks"
    #: Enable C++ symbol demangling
    demangle: bool = False


class IngestPMCStatStacks(DataGenTask):
    """
    Task that takes data from stack sample files generated by

    pmcstat -R <logfile> -G <stacksfile>

    and produces a tree weighted by the relative frequency of the children.
    """
    task_config_class = IngestPMCStatStacksConfig
    task_namespace = "pmc"
    task_name = "ingest-stacks"
    public = True

    @output
    def data(self):
        return BenchmarkIterationTarget(self, "stacks", ext="json")

    @output
    def collapsed_stacks(self):
        return BenchmarkIterationTarget(self, "collapsed-stacks", ext="txt")

    def run(self):
        stackcollapse = self.session.user_config.flamegraph_path / "stackcollapse-pmc.pl"
        if stackcollapse.exists():
            self.logger.debug("Folding stacks using %s", stackcollapse)

        data_paths = list(self.data.iter_paths())
        stc_paths = list(self.collapsed_stacks.iter_paths())
        for i in range(self.benchmark.config.iterations):
            stack_file = self.config.stack_file_pattern.format(iteration=i + 1)
            data_file = self.config.path / stack_file
            self.logger.debug("Ingest %s", data_file)

            if not data_file.exists():
                self.logger.error("Data file missing %s", data_file)
                continue
            stacks = self._parse_stacks(data_file)
            with open(data_paths[i], "w+") as outfd:
                json.dump(stacks, outfd)

            if stackcollapse.exists():
                with open(stc_paths[i], "w+") as outfd:
                    subprocess.run([stackcollapse, data_file], stdout=outfd)

    def _parse_line(self, line: str) -> dict:
        """
        Parse a line in the stacks file.

        This will have the format:
        <spaces><percent> [<N_samples>] <symbol> @ <location>

        We extract these fields into a dictionary and return it.
        The leading number of spaces is parsed as the 'level'.
        """
        expr = (r"^(?P<spaces>\s*)(?P<percent>[0-9.]+)%\s+\[(?P<samples>[0-9]+)\]"
                r"\s+(?P<raw_symbol>[0-9a-zA-Z_.$]+)\s*(?:@\s+(?P<path>[0-9a-zA-Z_/.-]+))?")

        m = re.match(expr, line)
        if not m:
            self.logger.error("Failed to parse: invalid stacks line '%s'", line)
            raise RuntimeError("Invalid stacks file")
        groups = m.groupdict()
        # Fixup data types and level value
        groups["samples"] = int(groups["samples"])
        groups["percent"] = float(groups["percent"])
        groups["level"] = len(groups.pop("spaces"))
        symbol = groups.pop("raw_symbol")
        if self.config.demangle:
            try:
                symbol = cxxfilt.demangle(symbol)
            except cxxfilt.InvalidName:
                pass
        groups["symbol"] = symbol

        return groups

    def _parse_stacks(self, stacks_file: Path) -> nx.DiGraph:
        data = {
            "samples": 0,
            "counter": None,
            "stacks": {}
        }
        stacks = data["stacks"]

        with open(stacks_file, "r") as st_fd:
            # The first line should be
            # @ <counter_name> [<N> samples]
            header = st_fd.readline()
            m = re.match(r"@\s+(?P<counter>[A-Za-z0-9_-]+)\s+\[(?P<samples>[0-9]+) samples\]", header)
            if not m:
                self.logger.error("Failed to parse %s: invalid header line %s", stacks_file, header)
                raise RuntimeError("Invalid stacks file")
            groups = m.groupdict()
            data["samples"] = groups["samples"]
            data["counter"] = groups["counter"]

            chain = []
            def add_stack():
                # Emit the stack and unwind up to the current level
                stack = ";".join(map(lambda t: t[0], reversed(chain)))
                if stack in stacks:
                    # Just add duplicated samples, although this should not happen, it
                    # seems that sometimes we see it
                    stacks[stack] += chain[-1][1]  # samples
                else:
                    stacks[stack] = chain[-1][1]  # samples
                del chain[entry["level"]:]

            for line in st_fd:
                line = line.rstrip()
                if not line:
                    # Skip blank lines
                    continue
                entry = self._parse_line(line)

                # check if going back to a previous level
                if entry["level"] <= len(chain) - 1:
                    add_stack()
                chain.append((entry["symbol"], entry["samples"]))
            add_stack()

        return data


@dataclass
class IngestPMCStatCountersConfig(Config):
    """
    Configuration for the task that loads pmcstat counters output.
    """
    #: Path of the counter data files to ingest
    path: ConfigPath
    #: Counter file pattern
    #: Use benchmark parameter templates to substitute the name for each
    #: different benchmark variation here. Note that a special {iteration}
    #: template key is used to substitute the iteration number.
    counter_file_pattern: str = "pmcstat.{iteration}.cnt"


class IngestPMCStatCounters(DataGenTask):
    """
    Task that takes data from pmcstat counters data files generated by

    pmcstat -O <logfile> -P counter ...

    and produces a dataframe containing the samples.

    Note that we assume that sample counters are incremental
    """
    task_config_class = IngestPMCStatCountersConfig
    task_namespace = "pmc"
    task_name = "ingest-counters"
    public = True

    @output
    def counter_data(self):
        return Target(self, "counter-data", ext="csv", loader=PLDataFrameLoadTask)

    def run(self):
        all_data = []
        for i in range(self.benchmark.config.iterations):
            pmc_file = self.config.counter_file_pattern.format(iteration=i + 1)
            data_file = self.config.path / pmc_file
            self.logger.debug("Ingest %s", data_file)

            if not data_file.exists():
                self.logger.error("Data file missing %s", data_file)
                continue
            pmc_df = self._parse_pmc(data_file)
            pmc_df = pmc_df.sum().with_columns(pl.lit(i).alias("iteration"))
            all_data.append(pmc_df)

        pl.concat(all_data).write_csv(self.counter_data.single_path())

    def _parse_pmc(self, pmc_file) -> pl.DataFrame:
        """
        Parse the PMC counters file
        """
        with open(pmc_file, "r") as pmc_fd:
            # The first line should be
            # # p/COUNTER ..
            header = pmc_fd.readline()
            if not header.startswith("#"):
                self.logger.error("Invalid PMC file header")
                raise RuntimeError("PMC file parsing error")
            matches = re.findall(r"[ps]/(?P<col>[a-zA-Z0-9_-]+)", header)
            cols = [c.lower() for c in matches]
        df = pl.from_pandas(pd.read_csv(pmc_file, sep="\s+", header=0, names=cols, index_col=False))
        return df
