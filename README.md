# Introduction
The CHERI benchplot tool/library is an attempt to build a flexible system to run CHERI experiments on CheriBSD and process the results.
Ideally the structure is modular and reusable enough that it can be adapted to speed-up data analysis or used as a starting point
for custom data analysis and visualization.

## Installation

The framework depends on Python 3.11+ due to the use of some newer features.
Furthermore, the `tools` directory contains some C++ tools that are used to accelerate parsing of
DWARF structures and QEMU protobuf traces. These are optionally needed, depending what analysis is being run.
The DWARF scraper tool requires the CHERI LLVM libraries to be installed in the CHERISDK directory and the `boost` libraries.

The following commands should install the dependencies and the pycheribenchplot package:
```
# This must point to the root of the cherisdk, containing the rootfs and the 'sdk' directory.
$ export CHERISDK=/path/to/cherisdk
# It is recommended to setup a virtual environment
$ virtualenv my-env
$ source my-env/bin/activate
(my-env) $ pip install .
# To check that the installation succeeded:
(my-env) $ benchplot-cli.py -h
```

It is possible to run tests as follows:
```
(my-env) $ pytest tests
```

> [!WARNING]
> Currently, the top-level makefile is only used to produce the documentation pages.
> It should be extended to build the tools automatically.

## Approach
The benchplot framework organisise data in _sessions_. A _session_ is a self-contained directory tree containing the following artefacts:
 - Generated scripts to run the benchmark data collection.
 - Data produced during the benchmark run.
 - Plots and other artefacts that are produced during analysis.
 - Any other supporting assets that may be required.

Each _session_ is created from a specific __pipeline configuration_ (`core.config.PipelineConfig`) file,
which defines the benchmark parameterisation axes and the combinations of benchmark configurations to run.
A separate _user configuration_ (`core.config.BenchplotUserConfig`) file is used to resolve artefacts
that may vary from system to system; this allows to move an existing session to a different machine
for analisys and still get useful results without baking local paths into the internal state.

The main entry point is the `benchplot-cli` tool. This provides a number of subcommands:
```
(my-env) $ benchplot-cli.py
usage: benchplot-cli.py [-h] [-v] [-l LOGFILE] [-w WORKERS] [-c CONFIG] {session,info} ...

CHERI plot and data analysis tool benchplot-cli

positional arguments:
  {session,info}        command
    session             Manage and ispect sessions.
    info                Display information about tasks.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose output
  -l LOGFILE, --logfile LOGFILE
                        logfile
  -w WORKERS, --workers WORKERS
                        Override max number of workers from configuration file
  -c CONFIG, --config CONFIG
                        User environment configuration file. Defaults to ~/.config/cheri-benchplot.json
```

The workflow is organised in the following phases:
 1. Write or reuse and existing _pipeline configuration_.
 2. Create a new session from the _pipeline configuration_
    `benchplot-cli.py session create /path/to/config.json path/to/session`.
 3. Generate benchmark run scripts for the session
    `benchplot-cli.py session generate path/to/session`.
    This step will populate the `<session>/run` directory with run scripts.
 4. Bundle the session
    `benchplot-cli.py session bundle -o bundles/path path/to/session`.
    This will produce a `bundles/path/session.tar.gz`, but you can always manually do this step.
 5. Move the bundle to the benchmark host and untar it.
 6. On the benchmark host, enter the `<session>/run` directory and run the appropriate script for the host.
    This will generate output files in the `<session>/run` directory tree.
 7. Once all the benchmarks have run, collect the session directory tree back to the analysis host.
 8. Write or reuse an existing _analysis configuration_.
 9. Run the analysis pass on your data
    `benchplot-cli.py session analyse -a path/to/analysis-config.json path/to/session`.
    This will populate plots and output artefrats in the `<session>/plots` directory.

Note that the benchmark host has minimal dependencies, we only require that it can run shell scripts and the benchmark executable. Depending on the data collected, we may require some auxiliary commands to be available; for example, collecting hardware performance counters requires the FreeBSD `pmcstat` utility.

## Internals

> [!WARNING]
> TODO this is outdated and incomplete. Maybe should move to sphinx docs.

Tasks are logically organised into namespaces. This is used to filter tasks that are compatible with each other.
For example, analysis tasks in the `foo` namespace should all be able to operate on data produced by generators in the `foo` namespace.

At session creation, a new `Benchmark` object is created for each benchmark/instance combination.
This represents a data-generation run and is uniquely identified by an UUID.
The benchmark instance will manage the data-generation and analysis for a benchmark/instance configuration pair.
Each benchmark contains a group of tasks (`ExecutionTask` subclasses) that represent data sources.
These tasks are responsible for generating the commands to produce the data, extract the data from the target cheribsd (or other) instance
once the benchmark is done, and load the data for analysis.

## Example -- using the _generic_ task

In this example, we will use the benchplot `generic` task to collect timing of an arbitrary workload that can be run from the shell.
We start by using the following configuration skeleton:

```
{
    "version": "1.0",
    "concurrent_instances": 1,
    "benchmark_config": {
        "name": "my-generic-example",
        "desc": "Description for humans",
        "iterations": 1,
        "parameterize": {
            "host": [],
            "variant": [],
            "runtime": [],
            "scenario": []
        },
        "system": [],
        "command_hooks": {},
        "generators": []
    }
}
```

First, we will add a `generator` configuration that produces the scripts to run the benchmark workload.
In this case, we are fine with a _generic_ generator, so let's find out how to configure it:

```
$ benchplot-cli.py info task generic.exec
# generic.exec (GenericExecTask):

This is a simple generic executor that uses the run_options configuration entry to run
a custom command and collect the output to the output file.

    Task configuration: GenericTaskConfig
GenericTaskConfig(timing_mode: pycheribenchplot.generic.timing.TimingTool = <TimingTool.HYPERFINE: 'hyperfine'>, command: str = None, collect_stdout: bool = False)
    #: Timing tool to use
    timing_mode: TimingTool = TimingTool.HYPERFINE

    #: Workload command to execute
    command: str = None

    #: Collect command stdout, note this is incompatible with timing
    collect_stdout: bool = False
```

This tells us that the `generic.exec` task expects a `command` configuration parameter, which is the command to run.
So let's start filling the pipeline configuration, in this case, we will use `git clone` as the workload.
Now, we need to collect execution time, to do so, we will use the existing integration of the `generic.exec` task with the `timing` module. Since we don't want to depend on hyperfine for collecting timing results, we will set `timing_tool` to "time".

```
{
    "version": "1.0",
    "concurrent_instances": 1,
    "benchmark_config": {
        "name": "my-generic-example",
        "desc": "Description for humans",
        "iterations": 1,
        "parameterize": {
            "host": [],
            "variant": [],
            "runtime": [],
            "scenario": []
        },
        "system": [],
        "command_hooks": {},
        "generators": [{
            "handler": "generic.exec",
            "task_options": {
                "command": "git clone https://github.com/CTSRD-CHERI/cheribsd-ports.git",
                "timing_mode": "time"
            }
        }]
    }
}
```

Now that we have this ironed out, we need to determine where we are running and what benchmark combinations we have.
In this example, we want to collect the run time depending on temporal safety. There are multiple ways to go about this,
but we will pick one that I think is convenient enough.

> [!TIP]
> Currently, the framework assumes that you always have 4 parameterisation axes that uniquely identify your benchmark run.
> The tuple (target, variant, runtime, scenario) should uniquely identify your run.
> This requirement will be relaxed in the future.

First, we define the benchmark host on which we are going to run, a Morello machine in this case.
In order to do this, we fill the `host` parameter axis and provide a matching `system` descriptor for the host.
The `system.name` field will be used to populate the internal `target` parameterisation axis.
Then, we add two `runtime` values, which represent whether the revocation is enabled or not, and finally we
move the git repository URL in the scenario axis.
The `variant` axis is unused here and we just use a placeholder "default" value there.

> [!TIP]
> While it is possible to specify directly the `target` parameterisation axis in place of `host`,
> this is done to highlight the existence of the host system configuration. Currently the `system`
> configuration is not really used internally, unless the task needs to know the cheri target string or kernel ABI;
> however, the idea is that we could automate the full workflow and the host_system mappings gives us
> a place to specify where to run different benchmark parameterisations
> (i.e. if we have multiple host machines with different architectures).

You may have noticed that as part of moving the repository URL to the `scenario` axis, we have replaced it
in the command with `{scenario}`. This means that when the benchmark combinations are generated, the command
string is substituted with the value of the scenario assigned to the specific benchmark combination.
In this case, there is only a single scenario value so the result is the same.

```
{
    "version": "1.0",
    "concurrent_instances": 1,
    "benchmark_config": {
        "name": "my-generic-example",
        "desc": "Description for humans",
        "iterations": 1,
        "parameterize": {
            "host": ["morello"],
            "variant": ["default"],
            "runtime": ["enabled", "disabled"],
            "scenario": ["https://github.com/CTSRD-CHERI/cheribsd-ports.git"]
        },
        "system": [{
            "matches": { "host": "morello" },
            "host_system": {
                "name": "morello",
                "kernel": "GENERIC-MORELLO-PURECAP-BENCHMARK-NODEBUG",
                "cheri_target": "morello-purecap",
                "kernelabi": "purecap"
            }
        }],
        "command_hooks": {},
        "generators": [{
            "handler": "generic.exec",
            "task_options": {
                "command": "git clone {scenario}",
                "timing_mode": "time"
            }
        }]
    }
}
```

Now, we need to enable and disable revocation based on the `runtime` axis.
In order to do so, we first disable revocation globally and then use `proccontrol` command to switch revocation on.
To disable revocation we want to run a `sysctl` command before starting any of the benchmarking.
This can be accomplished by adding a hook in the setup phase as follows; note the empty `command_hooks.setup[0].matches` field, which means the command hook will run for all the parameterisations.

Finally, we update the `command` to run `proccontrol` and we use template substitution to use the `runtime` axis as an argument to `proccontrol`.

```
{
    "version": "1.0",
    "concurrent_instances": 1,
    "benchmark_config": {
        "name": "my-generic-example",
        "desc": "Description for humans",
        "iterations": 1,
        "parameterize": {
            "host": ["morello"],
            "variant": ["default"],
            "runtime": ["enable", "disable"],
            "scenario": ["https://github.com/CTSRD-CHERI/cheribsd-ports.git"]
        },
        "system": [{
            "matches": { "host": "morello" },
            "host_system": {
                "name": "morello",
                "kernel": "GENERIC-MORELLO-PURECAP-BENCHMARK-NODEBUG",
                "cheri_target": "morello-purecap",
                "kernelabi": "purecap"
            }
        }],
        "command_hooks": {
            "setup": [{
                "matches": {},
                "commands": ["sysctl security.cheri.runtime_revocation_default=0"]
            }]
        },
        "generators": [{
            "handler": "generic.exec",
            "task_options": {
                "command": "proccontrol -m cherirevoke -s {runtime} git clone {scenario}",
                "timing_mode": "time"
            }
        }]
    }
}
```

Armed with this _pipeline configuration_ file, we generate our first _session_:

```
$ benchplot-cli.py session create clone-example.json out/clone-example
[INFO] cheri-benchplot: Create new session cbd72a08-2fc6-47a9-920e-5e00ed385f12
```

We can now inspect what benchmark combinations are present with

```
$ bechplot-cli.py info session out/clone-example
Session clone-demo (cbd72a08-2fc6-47a9-920e-5e00ed385f12):
        my-generic-example (2bbb5428-498b-4848-ad90-953de1d39b10) on morello
                Parameterization:
                 - host = morello
                 - variant = default
                 - runtime = enable
                 - scenario = https://github.com/CTSRD-CHERI/cheribsd-ports.git
                 - target = morello
                Generators:
                 - generic.exec
        my-generic-example (cecb3902-392c-4acd-9fc8-eff60f67a309) on morello
                Parameterization:
                 - host = morello
                 - variant = default
                 - runtime = disable
                 - scenario = https://github.com/CTSRD-CHERI/cheribsd-ports.git
                 - target = morello
                Generators:
                 - generic.exec
```

# Legacy documentation
## Example

 - XXX Rework this example to make it simpler.

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

## Other tracing backends
CHERI-QEMU supports two other tracing backends that may be useful
 - protobuf backend:
   + Build qemu with `--enable-protobuf-log-instr`, run with `--cheri-trace-backend protobuf`
   + Emits a stream of protobufs that represent the trace entries. See `cheri-perfetto/protos/perfetto/trace/track_event/qemu_log_entry.proto` for the proto definitions.
 - json backend:
   + Build qemu with `--enable-json-log-instr`, run with `--cheri-trace-backend json`
   + Emit a json file that contains the log entries. The format should be more stable and easier to parse than the text format, however protobufs or perfetto should
   be used for any trace that is not a toy program.

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
