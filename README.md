# Introduction
The CHERI benchplot tool/library is an attempt to build a flexible system to run CHERI experiments on CheriBSD and process the results.
Ideally the structure is modular and reusable enough that it can be adapted to speed-up data analysis or used as a starting point
for custom data analysis and visualization.

## Design
The tool uses two configuration files:
 - user configuration: by default in `~/.config/cheri-benchplot.json`, specifies cheri SDK information
 - session configuration: specifies the benchmark/cheribsd instance combinations to run
 
Currently the runner only supports starting up CHERI-QEMU instances for behavioural analysis and qemu tracing.

There are two modes of operation for the *benchplot* tool:
 - run: Run all combinations of instance/benchmarks in the given session configuration
 - analyse: Run all compatible analysis steps (e.g. table generators, plotting, integrity checks...)
The `demo` directory contains examples for the configuration files.

For each benchmark/instance combination a new `Benchmark` object is created. This represents a benchmark run and is uniquely
identified by an UUID. The benchmark instance will manage the benchmark run and analysis for a benchmark/instance configuration pair.
Each benchmark contains a set of datasets (`DataSetContainer`) that represent data sources.
Each dataset is responsible for generating the commands to produce the data, extract the data from the instance
once the benchmark is done, and load the data for later analysis.

The analysis step is split in two phases:
 1. For each recorded benchmark run, we load the raw data and identify the baseline benchmark run.
    The datasets are merged into a single dataframe in the baseline benchmark run, then we aggregate
    the data to compute statistical metrics (e.g. mean and median) across iterations and deltas between
    the baseline and non-baseline runs.
 2. All the analysis steps that depend on datasets we can provide are run. This step uses the loaded and
    aggregated data from datasets and generates some output representation for it.
    This is done to split the data ingestion and normalization from the actual data analysis and reporting.

## Example

Required branches:
 - cheribsd: statcounters-update
 - qemu: qemu-experimental-tracing
 - netperf: cheri-netperf
 - cheri-perfetto: cheri-perfetto
 - cheri-benchplot: master
 
The example runs a simple UDP_RR netperf benchmark and collects data from all supported data sources.
 1. Build cheribsd: `cheribuild cheribsd-riscv64-purecap --cheribsd/build-alternate-abi-kernels --cheribsd/build-bench-kernels`
 2. Build qemu: `cheribuild qemu --qemu/configure-options '--enable-perfetto-log-instr'`
 3. Build netperf: `cheribuild netperf-riscv64-purecap`
 4. Build cheri-perfetto: this is not currently automated by cheribuild.
 ```
 cd path/to/cheri-perfetto
 mkdir -p out
 tools/install-build-deps
 # Can either use the shipped gn/ninja or the system gn/ninja
 tools/gn gen out/cheri-perfetto
 tools/ninja -C out/cheri-perfetto
 ```
 7. Install benchplot and the perfetto python bindings (might be best to use a virtualenv)
 ```
 pip install -e path/to/cheri-benchplot
 pip install -e path/to/cheri-perfetto/src/trace_processor/python
 ```
 5. Make sure the benchplot user configuration in `.config/cheri-benchplot.json` matches the your setup:
```
{
    "sdk_path": "path/to/cherisdk",
    "perfetto_path": "path/to/cheri-perfetto/out/cheri-perfetto",
    "cheribuild_path": "path/to/cheribuild/cheribuild.py"
}
```
 6. Run the demo session: 
 ```
 python benchplot.py demo/demo-netperf-multi-instance/benchplot_config.json run
 python benchplot.py demo/demo-netperf-multi-instance/benchplot_config.json analyse
 ```

## CHERI Trace Processor
This is currently a skeleton for more advanced parsing of qemu traces from both the protobuf and the perfetto backends.
It contains an example to read a protobuf trace from the protobuf backend. Perfetto traces should be loaded using the
`trace_processor_shell` from the cheri-perfetto repository.

### Example setup to record and observe protobuf traces
Required branches:
 - cheribsd: statcounters-update (should also work on dev, with less data output)
 - qemu: qemu-experimental-tracing
 - cheri-benchplot: master
 
 1. Build qemu `cheribuild qemu --qemu/configure-options '--enable-protobuf-log-instr'`
 2. Run qemu (assuming you have built cheribsd and a disk image) `cheribuild run-riscv64-purecap --run-riscv64-purecap/extra-options '--cheri-trace-backend protobuf'`
 3. In the qemu guest, run a sample program:

     ```sh
     $ sysctl hw.qemu_trace_perthread=1
     $ qtrace exec helloworld
     $ poweroff
     ```
     This will generate the qemu-protobuf.pb file (currently hardcoded, sorry I am lazy...)
 4. Run `python cheri_trace_processor/cheri_trace_processor.py path/to/qemu-protobuf.pb`
