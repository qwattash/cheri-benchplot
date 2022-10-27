import shlex
import typing
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from .util import new_logger


@dataclass
class CommandHistoryEntry:
    """Internal helper to bind commands to PIDs of running tasks"""
    cmd: str
    pid: int = None


@dataclass
class BenchmarkScriptCommand:
    """
    Represent a single command executed by the benchmark script
    """

    #: The command to execute
    cmd: str

    #: CLI arguments as a list
    args: typing.List[str]

    #: Env variables as a dict ENV_VAR => value
    env: typing.Dict[str, str]

    #: Run the command in background
    background: bool = False

    #: Collect the PID of the command
    collect_pid: bool = False

    #: Redirect remote output to the given file
    remote_output: Path = None

    #: Local path where we will extract the remote output file
    local_output: Path = None

    #: Any extra output paths to extract, maps remote_path => local_path
    extra_output: typing.Dict[Path, Path] = field(default_factory=dict)

    #: Custom data extraction function
    extractfn: typing.Callable[["InstanceInfo", Path, Path], None] = None

    #: Bind command to CPU
    cpu: int = None

    #: Debug mode prints the commands before executing
    debug: bool = False

    def build_sh_command(self, cmd_index: int) -> typing.List[str]:
        cmdargs = f"{self.cmd} " + " ".join(map(str, self.args))
        env_vars = [f"{name}={value}" for name, value in self.env.items()]
        if self.remote_output:
            output_redirect = f" >> {self.remote_output}"
        else:
            output_redirect = ""
        envstr = " ".join(env_vars)
        if self.cpu is None:
            cpuset = ""
        else:
            cpuset = f"cpuset -c -l {cpu}"

        cmd_lines = []
        if self.debug:
            cmd_str = f"{envstr} {cpuset} {cmdargs}".strip()
            cmd_lines.append(f"echo '{cmd_str}'")

        if self.background:
            cmd_str = f"{envstr} {cpuset} {cmdargs} {output_redirect}".strip()
            cmd_lines.append(f"{cmd_str} &")
            if self.collect_pid:
                cmd_lines.append(f"PID_{cmd_index}=$!")
        else:
            cmdline = f"{cmdargs} {output_redirect}"
            if self.collect_pid:
                cmdline = f"PID_{cmd_index}=`{cpuset} sh -c \"echo \\\$\\\$; {envstr} exec {cmdline}\" | head -n 1`".strip(
                )
            else:
                cmdline = f"{envstr} {cpuset} {cmdline}".strip()
            cmd_lines.append(cmdline)
        return cmd_lines


class ShellScriptBuilder:
    """
    Generate a shell script that runs the benchmark steps as they are scheduled by
    datasets.
    When the script is done, data is extracted from the guest as needed and the
    datasets get a chance to perform a post-processing step.
    """
    @dataclass
    class VariableRef:
        name: str

        def __str__(self):
            return f"${{{self.name}}}"

    def __init__(self, benchmark: "Benchmark"):
        self.benchmark = benchmark
        # Main command list
        self._commands = []
        # Intentionally relative to the run-script location.
        # We may want to add a knob to be able to store these in tmpfs or other places.
        self._guest_output = benchmark.config.remote_output_dir

        self._prepare_guest_output_dirs()

    def _add_command(self, command, args, env=None, collect_pid=False):
        env = env or {}
        cmd = BenchmarkScriptCommand(command, args, env)
        cmd.collect_pid = collect_pid
        self._commands.append(cmd)

    def _prepare_guest_output_dirs(self):
        """
        Prepare the guest output environment to store the files to mirror the local benchmark instance
        output directory.
        This makes it easier to share file paths by just using directories relative to the benchmark
        output_path.
        """
        self._add_command("mkdir", ["-p", self._guest_output])
        for i in range(self.benchmark.config.iterations):
            self._add_command("mkdir", ["-p", self._guest_output / str(i)])
        self._add_command("touch", [self.local_to_remote_path(self.command_history_path())])

    def local_to_remote_path(self, host_path: Path) -> Path:
        base_path = self.benchmark.get_benchmark_data_path()
        assert host_path.is_absolute(), f"Ensure host_path is absolute {host_path}"
        assert str(host_path).startswith(str(base_path)), "Ensure host_path is in benchmark output"
        return self._guest_output / host_path.relative_to(base_path)

    def command_history_path(self):
        return self.benchmark.get_benchmark_data_path() / f"command-history-{self.benchmark.uuid}.csv"

    def get_commands_with_pid(self):
        """Return a list of commands for which we recorded the PID in the command history"""
        commands = []
        for cmd in self._commands:
            if cmd.collect_pid:
                commands.append(cmd.cmd)
        return commands

    def get_variable(self, name: str) -> VariableRef:
        return self.VariableRef(name)

    def gen_cmd(self,
                command: str,
                args: list,
                outfile: Path = None,
                env: typing.Dict[str, str] = None,
                extractfn=None,
                extra_outfiles: typing.List[Path] = [],
                pin_cpu: bool = None,
                collect_pid: bool = False):
        """
        Add a foreground command to the run script.
        If the output is to be captured, the outfile argument specifies the host path in which it will be
        extracted. The host path must be within the benchmark instance data path
        (see BenchmarkBase.get_benchmark_data_path()), the guest output path will be derived automatically from it.
        If extra post-processing should be performed upon file extraction, a callback can be given via
        extractfn. This function will be called to extract the remote file to the output file as
        `extractfn(benchmark, remote_path, host_path)`.
        """
        env = env or {}
        cmd = BenchmarkScriptCommand(command, args, env)
        if pin_cpu is not None:
            cmd.cpu = pin_cpu
        if outfile is not None:
            cmd.remote_output = self.local_to_remote_path(outfile)
        cmd.local_output = outfile
        cmd.extra_output = {self.local_to_remote_path(p): p for p in extra_outfiles}
        cmd.extractfn = extractfn
        cmd.collect_pid = collect_pid
        self._commands.append(cmd)

    def gen_bg_cmd(self,
                   command: str,
                   args: list,
                   outfile: Path = None,
                   env: typing.Dict[str, str] = None,
                   extractfn=None,
                   pin_cpu=None) -> VariableRef:
        """
        Similar to add_cmd() but will return an handle that can be used at a later time to terminate the
        background process.
        """
        env = env or {}
        cmd = BenchmarkScriptCommand(command, args, env)
        if pin_cpu is not None:
            cmd.cpu = pin_cpu
        if outfile is not None:
            cmd.remote_output = self.local_to_remote_path(outfile)
        cmd.local_output = outfile
        cmd.extractfn = extractfn
        cmd.background = True
        cmd.collect_pid = True
        cmd.debug = self.benchmark.user_config.verbose
        self._commands.append(cmd)
        cmd_index = len(self._commands) - 1
        return self.VariableRef(f"PID_{cmd_index}")

    def gen_stop_bg_cmd(self, command_handle: VariableRef):
        self._add_command("kill", ["-KILL", command_handle])

    def gen_sleep(self, seconds: int):
        self._add_command("sleep", [seconds])

    def to_shell_script(self, fd: typing.IO[str]):
        fd.write("#!/bin/sh\n\n")
        for i, cmd in enumerate(self._commands):
            lines = cmd.build_sh_command(i)
            for l in lines:
                fd.write(l + "\n")
        # Dump all the collected PIDs in the command history metadata file
        command_history = self.command_history_path()
        pid_history_path = self.local_to_remote_path(command_history)
        for i, cmd in enumerate(self._commands):
            if cmd.collect_pid:
                var = self.get_variable(f"PID_{i}")
                fd.write(f"echo {var} >> {pid_history_path}\n")

    def get_extract_files(self) -> typing.List[typing.Tuple[Path, Path, typing.Optional[typing.Callable]]]:
        """
        Get a list of all files to extract for the benchmark.
        Each list item is a tuple of the form (remote_file, local_file, extract_fn)
        If a custom extract function is given, it will be passed along as the
        last tuple item for each file.
        """
        entries = []
        for cmd in self._commands:
            if cmd.remote_output:
                assert cmd.local_output, "Missing local output file"
                entries.append((cmd.remote_output, cmd.local_output, cmd.extractfn))
            if cmd.extra_output:
                for remote_path, local_path in cmd.extra_output.items():
                    entries.append((remote_path, local_path, cmd.extractfn))
        return entries


@dataclass
class ScriptCommand:
    """
    Represent a single command executed by the generated run script.
    """
    #: The command to execute
    cmd: str

    #: CLI arguments as a list
    args: list[str] = field(default_factory=list)

    #: Env variables as a dict ENV_VAR => value
    env: dict[str, str] = field(default_factory=dict)

    #: path to the output file we should redirect stdout to
    stdout_path: Path | None = None

    #: Run the command in background
    background: bool = False

    #: Bind command to CPU
    cpu: int | None = None

    #: Debug mode prints the commands before executing
    debug: bool = False

    #: Unique name of the PID reference variable. This is useful to kill background processes during teardown
    pid_reference_id: str | None = None

    def generate(self, fd: typing.IO[str]):
        # Base command bits
        cmd_string = shlex.join([str(self.cmd)] + [str(v) for v in self.args])
        env_parts = [f"{name}=" + shlex.quote(str(value)) for name, value in self.env.items()]
        env_string = " ".join(env_parts)

        # Modifiers
        if self.stdout_path:
            stdout = f">> {self.stdout_path}"
        else:
            stdout = ""
        if self.cpu is not None:
            cpuset = ["cpuset", "-c", "-l", str(self.cpu)] + [cmd_string]
            cmd_string = shlex.join(cpuset)

        full_command = f"{env_string} {cmd_string}".strip()
        if self.background:
            background = "&"
        else:
            background = ""
        lines = []

        exec_command = f"{full_command} {stdout}".strip()
        exec_command = f"{exec_command} {background}".strip()
        lines.append(exec_command)

        if self.background and self.pid_reference_id:
            # We want to capture the pid
            lines.append(f"PID_{self.pid_reference_id}=$!")

        if self.debug:
            lines.append(f"echo {full_command}")

        for line in lines:
            fd.write(line + "\n")


class ScriptBuilder:
    """
    Generate a shell script that runs the benchmark steps as they are scheduled by
    datasets.

    The script is organised into sections.
    Each benchmark execution task can take its turn to add commands to each section.
    The sections are separated in steps, so that the order of the commands within a single
    section should not matter. It is inadvisable anyway to perform too many operations
    in between benchmark iterations steps so the number of operations per section should
    generally stay on the low side.

    Currently the section are as follow:
    - pre-benchmark -> common preparatory activity
    - benchmark -> benchmark activity
    - post-benchmark -> system stats extraction steps and data massaging
    - last -> anything that depends on something in post_benchmark

    The benchmark section is further split into iterations:
    - pre-benchmark-iter -> sampling before a benchmark iteration
    - benchmark-iter -> benchmark iteration activity (generally a single command)
    - post-benchmark-iter -> sampling after a benchmark iteration
    """
    class Section:
        """
        A section containing commands.
        """
        def __init__(self, script, name):
            #: Parent script
            self._script = script
            #: Name of the section, mostly for debug output
            self.name = name
            #: Command sequence
            self._commands = []

        def add_cmd(self,
                    command: str | Path,
                    args: list[str] = None,
                    env: dict[str, str] = None,
                    output: "Target" = None,
                    cpu: int | None = None) -> ScriptCommand:
            """
            Add a foreground command to the run script.

            :param cmd_path: Path to the executable (guest path)
            :param args: Command arguments as a list of strings
            :param output: Target specifying a file to which redirect the command output
            :return: An opaque command reference, this is useful to terminate background commands
            """
            cmd = ScriptCommand(cmd=command, args=args or [], env=env or {})
            if output:
                cmd.stdout_path = output.to_remote_path()
            if cpu is not None:
                cmd.cpu = cpu
            cmd.pid_reference_id = str(self._script._next_sequence())
            with self._script._script_lock:
                self._commands.append(cmd)

        def generate(self, fd: typing.IO[str], logger):
            logger.debug("Populating script section %s", self.name)
            for cmd in self._commands:
                cmd.generate(fd)

    def __init__(self, benchmark):
        #: Benchmark that owns this script
        self.benchmark = benchmark
        #: Shell generator logger
        self.logger = new_logger("shellgen", parent=benchmark.logger)
        #: Path to the benchmark data output directory on the guest machine, relative to guest $HOME
        self._guest_output_path = benchmark.config.remote_output_dir
        #: Global command sequence number, protected by the script lock
        self._sequence = 0
        #: Script command list lock
        self._script_lock = Lock()
        #: Sections structure
        self.sections = {
            "pre-benchmark":
            self.Section(self, "pre-benchmark"),
            "benchmark": [{
                "pre-benchmark": self.Section(self, f"pre-benchmark-{i}"),
                "benchmark": self.Section(self, f"benchmark-{i}"),
                "post-benchmark": self.Section(self, f"post-benchmark-{i}")
            } for i in range(benchmark.config.iterations)],
            "post-benchmark":
            self.Section(self, "post-benchmark"),
            "last":
            self.Section(self, "last")
        }
        self._prepare_guest_output_dirs()

    def _prepare_guest_output_dirs(self):
        """
        Prepare the guest output environment to store the files to mirror the local benchmark instance
        output directory.
        This makes it easier to share file paths by just using directories relative to the benchmark
        output_path.
        """
        section = self.sections["pre-benchmark"]
        section.add_cmd("mkdir", ["-p", self._guest_output_path])
        section.add_cmd("mkdir", ["-p", self._guest_output_path])
        for i in range(self.benchmark.config.iterations):
            section.add_cmd("mkdir", ["-p", self._guest_output_path / str(i)])
        # section.add_cmd("touch", [self.local_to_remote_path(self.command_history_path())])

    def _next_sequence(self):
        with self._script_lock:
            self._sequence += 1
            return self._sequence

    @property
    def benchmark_sections(self):
        """Shorthand to access the per-iteration sections list"""
        return self.sections["benchmark"]

    def generate(self, fd: typing.IO[str]):
        fd.write("#!/bin/sh\n\n")
        fd.write("set -e\n")
        # Syncronization here is not really needed since we only do this
        # in the top-level benchmark task, where the script is private to each
        # benchmark.
        with self._script_lock:
            self.sections["pre-benchmark"].generate(fd, self.logger)
            for group in self.sections["benchmark"]:
                group["pre-benchmark"].generate(fd, self.logger)
                group["benchmark"].generate(fd, self.logger)
                group["post-benchmark"].generate(fd, self.logger)
            self.sections["post-benchmark"].generate(fd, self.logger)
            self.sections["last"].generate(fd, self.logger)
