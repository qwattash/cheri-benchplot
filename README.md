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

Each _session_ is created from a specific _pipeline configuration_ (`core.config.PipelineConfig`) file,
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

Note that the benchmark host has minimal dependencies, we only require that it can
run shell scripts and the benchmark executable. Depending on the data collected,
we may require some auxiliary commands to be available; for example, collecting
hardware performance counters may require the FreeBSD `pmcstat` utility.

### Pipeline Configuration

The pipeline configuration is the user-facing description of the desired
benchmark parameterisation. This describes the benchmark parameterisation axes
and how these are used to generate the benchmark workload combinations to execute.
Below is a worked example that produces a configuration that generates the generic
workload executor.

The JSON pipeline configuration has the following shape:

```
{
    "version": "1.0",
    "concurrent_instances": 1,
    "benchmark_config": {
        "name": "my-generic-example",
        "desc": "Description for humans",
        "iterations": 1,
        "parameterize": {},
        "system": [],
        "command_hooks": {},
        "generators": []
    }
}
```

The key fields in this file are:

 - *parameterize*: describes the parameterization axes as a dictionary of
 "axis name" => [axis values]. For example, `{ "buffer-size"; [1, 1024, 2048 ]}`
 defines a single benchmarking axis named "buffer-size" with values 1, 1024 and 2048.
 This means that the pipeline configuration will be expanded to produce three benchmark
 executions, each with its own buffer-size value.
 - *system*: defines the "target" assignment for different parameter combinations.
  This is intended to describe if/how different parameterisations map to the
  platform on which they run. For instance, if we have a `"platform": ["arm", "riscv"]`
  parameter axis, we probably want all "arm" benchmark workloads to run on an ARM
  benchmark host. Similarly, this can be done to differentiate benchmark workloads that
  require booting the host with different kernels or boot-time tunables.
 - *command_hooks*: describes additional commands to be injected at different
 stages of the benchmark run script. This is useful, for instance, to load kernel
 modules or set sysctl values corresponding to different parameterisations.
 - *generators*: configures the tasks that generate the benchmark executor scripts.
 These can be further inspected using the `benchplot-cli info task` command.

> [!TIP]
> Some parameterisation axis names are reserved and will result in errors.
> The name "iteration" is reserved for internal use (the benchmark iteration number).
> The name "descriptor" is reserved for internal use.
> The name "target" is semi-reserved, the value is automatically generated from the
> `benchmark_config.system` mapping, but can be forced if the axis is explicitly
> specificed.

The `benchplot-cli` tool is your friend, this can be used to show
a description for the various fields as `benchplot-cli info config --desc pipeline`.
This will produce a description of the pipeline configuration; furthermore,
`benchplot-cli info config --generate pipeline` will emit a default empty
pipeline configuration.
Note that the configuration descriptions will include all the nested configuration
objects.

### Analysis

> [!WARNING]
> TODO this needs to be documented.

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
