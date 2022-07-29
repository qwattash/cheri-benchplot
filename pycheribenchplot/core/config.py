import argparse as ap
import collections
import itertools as it
import json
import shutil
import typing
from collections import OrderedDict
from dataclasses import (MISSING, Field, asdict, dataclass, field, fields, is_dataclass, replace)
from datetime import date, datetime
from enum import Enum, auto
from pathlib import Path
from uuid import UUID, uuid4

from dataclasses_json import DataClassJsonMixin, config
from typing_inspect import get_args, get_origin, is_generic_type


def _template_safe(temp: str, **kwargs):
    try:
        return temp.format(**kwargs)
    except KeyError:
        return temp


def path_field(default=None):
    return field(default=Path(default) if default else None, metadata=config(encoder=str, decoder=Path))


def lazy_nested_config_field():
    return field(default_factory=dict, metadata={"late_config_bind": True})


def is_lazy_nested_config(f: Field):
    return f.metadata.get("late_config_bind", False)


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, UUID):
            return str(o)
        elif isinstance(o, Path):
            return str(o)
        elif isinstance(o, Enum):
            return o.value
        elif isinstance(o, datetime):
            return o.timestamp()
        return super().default(o)


@dataclass
class Config(DataClassJsonMixin):
    """
    Base class for JSON-based configuration file parsing.
    Each field in this dataclass is deserialized from a json file, with support to
    nested configuration dataclasses.
    Types of the fields are normalized to the type annotation given in the dataclass.
    """
    @classmethod
    def load_json(cls, jsonpath):
        with open(jsonpath, "r") as jsonfile:
            return super().from_json(jsonfile.read())

    @classmethod
    def merge(cls, *other: typing.Tuple["Config"]):
        """
        Similar to dataclass replace but uses fields from another dataclass that must be
        a parent class of the instance type we are replacing into.
        This allows to merge separate dataclasses into a combined view.
        Useful for merging multiple configuration files that specify different fields together.
        """
        init_fields = {}
        other_fields = {}
        for section in other:
            assert issubclass(cls, type(section)), "Can only merge config in child classes"
            for f in fields(section):
                if f.init:
                    init_fields[f.name] = getattr(section, f.name)
                else:
                    other_fields[f.name] = getattr(section, f.name)
        inst = cls(**init_fields)
        for name, val in other_fields.items():
            setattr(inst, name, val)
        return inst

    def emit_json(self) -> str:
        """
        Custom logic to emit json.
        This is required as in older python version pathlib objects are not serializable.
        """
        data = self.to_dict()
        return json.dumps(data, cls=ConfigEncoder, indent=4)

    def _normalize_sequence(self, f, sequence):
        type_args = get_args(f.type)
        item_type = type_args[0]
        if len(sequence) == 0:
            # Nothing to normalize
            return sequence
        if is_dataclass(item_type) and not is_dataclass(sequence[0]):
            items = [item_type(**item) for item in getattr(self, f.name)]
            setattr(self, f.name, items)

    def _normalize_mapping(self, f, mapping):
        type_args = get_args(f.type)
        item_type = type_args[1]
        if len(mapping) == 0:
            # Nothing to normalize
            return mapping
        first_item = next(it.islice(mapping.values(), 1))
        if is_dataclass(item_type) and not is_dataclass(first_item):
            items = {key: item_type(**item) for key, item in mapping.items()}
            setattr(self, f.name, items)

    def __post_init__(self):
        for f in fields(self):
            if not f.init:
                continue
            # Check for existence as this will cause issues down the line
            assert hasattr(self, f.name), f"Missing field {f.name}, use a default value"
            origin = get_origin(f.type)
            type_args = get_args(f.type)
            value = getattr(self, f.name)
            if is_dataclass(f.type):
                if type(value) == dict:
                    setattr(self, f.name, f.type(**value))
            elif is_lazy_nested_config(f):
                if type(value) != dict and not is_dataclass(value):
                    raise ValueError("Lazy config field must either be a dict or a dataclass instance")
            elif type(origin) == type:
                # standard type
                if issubclass(origin, collections.abc.Sequence):
                    self._normalize_sequence(f, value)
                elif issubclass(origin, collections.abc.Mapping):
                    self._normalize_mapping(f, value)
                else:
                    setattr(self, f.name, origin(value))
            elif origin is None:
                # Not a typing class (e.g. Union)
                if issubclass(f.type, Path) and value is not None:
                    setattr(self, f.name, Path(value).expanduser())


class TemplateConfigContext:
    """
    Base class for context that can be used to bind TemplateConfig to.
    """
    def __init__(self):
        self._template_params = {}

    def register_template_subst(self, **kwargs):
        for key, value in kwargs.items():
            self._template_params[key] = value

    def conf_template_params(self):
        return dict(self._template_params)


@dataclass
class TemplateConfig(Config):
    def _bind_one(self, context, dtype, value):
        params = context.conf_template_params()
        if dtype == str:
            return _template_safe(value, **params)
        elif is_dataclass(dtype) and issubclass(dtype, TemplateConfig):
            return value.bind(context)
        elif dtype == Path:
            str_path = _template_safe(str(value), **params)
            return Path(str_path)
        return value

    def bind_field(self, context, f: Field, value):
        """
        Run the template substitution on a config field.
        If the field is a collection or a nested TemplateConfig, we recursively bind
        each value.
        """
        origin = get_origin(f.type)
        if is_dataclass(f.type):
            # Forward the nested bind if the dataclass is a TemplateConfig
            return self._bind_one(context, f.type, value)
        elif is_lazy_nested_config(f):
            # Signals that the field contains a lazily resolved TemplateConfig, if there is one
            # recurse the binding, else skip it.
            if is_dataclass(value):
                return self._bind_one(context, type(value), value)
            return value
        elif origin is typing.Union:
            args = get_args(f.type)
            if len(args) == 2 and args[1] == None:
                # If we have an optional field, bind with the type argument instead
                return self._bind_one(context, args[0], value)
            else:
                # Common union field, use whatever type we have as the argument as we do not
                # know how to parse it
                return self._bind_one(context, type(value), value)
        elif origin is typing.List or origin is list:
            arg_type = get_args(f.type)[0]
            return [self._bind_one(context, arg_type, v) for v in value]
        elif origin is typing.Dict or origin is dict:
            arg_type = get_args(f.type)[1]
            return {key: self._bind_one(context, arg_type, v) for key, v in value.items()}
        else:
            return self._bind_one(context, f.type, value)

    def bind(self, context):
        """
        Run a template substitution pass with the given substitution context.
        This will resolve template strings as "{foo}" for template key/value
        substitutions that have been registerd in the contexv via
        TemplateConfigContext.register_template_subst() and leave the missing
        template parameters unchanged for later passes.
        """
        changes = {}
        for f in fields(self):
            if not f.init:
                continue
            replaced = self.bind_field(context, f, getattr(self, f.name))
            if replaced:
                changes[f.name] = replaced
        return replace(self, **changes)


class InstancePlatform(Enum):
    QEMU = "qemu"
    VCU118 = "vcu118"

    def __str__(self):
        return self.value


class InstanceCheriBSD(Enum):
    RISCV64_PURECAP = "riscv64-purecap"
    RISCV64_HYBRID = "riscv64-hybrid"
    MORELLO_PURECAP = "morello-purecap"
    MORELLO_HYBRID = "morello-hybrid"

    def is_riscv(self):
        return (self == InstanceCheriBSD.RISCV64_PURECAP or self == InstanceCheriBSD.RISCV64_HYBRID)

    def is_morello(self):
        return (self == InstanceCheriBSD.MORELLO_PURECAP or self == InstanceCheriBSD.MORELLO_HYBRID)

    def freebsd_kconf_dir(self):
        if self.is_riscv():
            arch = "riscv"
        elif self.is_morello():
            arch = "arm64"
        else:
            assert False, "Unknown arch"
        return Path("sys") / arch / "conf"

    def __str__(self):
        return self.value


class InstanceKernelABI(Enum):
    NOCHERI = "nocheri"
    HYBRID = "hybrid"
    PURECAP = "purecap"

    def __str__(self):
        return self.value


@dataclass
class PlatformOptions(TemplateConfig):
    """
    Base class for platform-specific options.
    This is internally used during benchmark dataset collection to
    set options for the instance that is to be run.
    """
    # Number of cores in the system
    cores: int = 1
    # The trace file used by default unless one of the datasets overrides it
    qemu_trace_file: typing.Optional[Path] = None
    # Run qemu with tracing enabled
    qemu_trace: bool = False
    # Trace categories to enable for qemu-perfetto
    qemu_trace_categories: typing.Set[str] = field(default_factory=set)
    # VCU118 bios
    vcu118_bios: typing.Optional[Path] = None
    # IP to use for the VCU118 board
    vcu118_ip: str = "10.88.88.2"


@dataclass
class InstanceConfig(TemplateConfig):
    """
    Configuration for a CheriBSD instance to run benchmarks on.
    XXX-AM May need a custom __eq__() if iterable members are added
    """
    kernel: str
    baseline: bool = False
    name: typing.Optional[str] = None
    platform: InstancePlatform = InstancePlatform.QEMU
    cheri_target: InstanceCheriBSD = InstanceCheriBSD.RISCV64_PURECAP
    kernelabi: InstanceKernelABI = InstanceKernelABI.HYBRID
    # Is the kernel config name managed by cheribuild or is it an extra one
    # specified via --cheribsd/extra-kernel-configs?
    cheribuild_kernel: bool = True
    # Internal fields, should not appear in the config file and are missing by default
    platform_options: PlatformOptions = field(default_factory=PlatformOptions)

    @property
    def user_pointer_size(self):
        if (self.cheri_target == InstanceCheriBSD.RISCV64_PURECAP
                or self.cheri_target == InstanceCheriBSD.MORELLO_PURECAP):
            return 16
        elif (self.cheri_target == InstanceCheriBSD.RISCV64_HYBRID
              or self.cheri_target == InstanceCheriBSD.MORELLO_HYBRID):
            return 8
        assert False, "Not reached"

    @property
    def kernel_pointer_size(self):
        if (self.cheri_target == InstanceCheriBSD.RISCV64_PURECAP
                or self.cheri_target == InstanceCheriBSD.MORELLO_PURECAP):
            if self.kernelabi == InstanceKernelABI.PURECAP:
                return self.user_pointer_size
            else:
                return 8
        elif (self.cheri_target == InstanceCheriBSD.RISCV64_HYBRID
              or self.cheri_target == InstanceCheriBSD.MORELLO_HYBRID):
            if self.kernelabi == InstanceKernelABI.PURECAP:
                return 16
            else:
                return self.user_pointer_size
        assert False, "Not reached"

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = f"{self.platform}-{self.cheri_target}-{self.kernelabi}-{self.kernel}"

    def __str__(self):
        return f"{self.name}"


class DatasetName(Enum):
    """
    Public name used to identify dataset parsers (and, by association, generators)
    in the configuration file.
    """
    PMC = "pmc"
    PMC_PROFCLOCK_STACKSAMPLE = "pmc-profclock-stack"
    NETPERF_DATA = "netperf-data"
    PROCSTAT_NETPERF = "procstat-netperf"
    QEMU_STATS_BB_HIT = "qemu-stats-bb-hit"
    QEMU_STATS_BB_ICOUNT = "qemu-stats-bb-icount"
    QEMU_STATS_CALL_HIT = "qemu-stats-call"
    QEMU_UMA_COUNTERS = "qemu-glob-uma-counters"
    QEMU_VM_KERN_COUNTERS = "qemu-glob-vm-kern-counters"
    PIDMAP = "pidmap"
    VMSTAT_UMA_INFO = "vmstat-uma-info"
    VMSTAT_MALLOC = "vmstat-malloc"
    VMSTAT_UMA = "vmstat-uma"
    NETSTAT = "netstat"
    KERNEL_CSETBOUNDS_STATS = "kernel-csetbounds-stats"
    KERNEL_STRUCT_STATS = "kernel-struct-stats"
    KERNEL_STRUCT_MEMBER_STATS = "kernel-struct-member-stats"

    # Test only name
    TEST_FAKE = "test-fake"

    def __str__(self):
        return self.value


class DatasetArtefact(Enum):
    """
    Internal identifier for dataset artifacts that are generated by a dataset object.
    This identifies the artifact that the dataset generates, and is used to avoid
    generating multiple times the same data if multiple datasets reuse the same input
    for processing.
    """
    NETPERF = auto()
    PMC = auto()
    PMC_PROFCLOCK_STACKSAMPLE = auto()
    VMSTAT = auto()
    UMA_ZONE_INFO = auto()
    PROCSTAT = auto()
    PIDMAP = auto()
    QEMU_STATS_BB_HIT = auto()
    QEMU_STATS_BB_ICOUNT = auto()
    QEMU_STATS_CALL_HIT = auto()
    QEMU_COUNTERS = auto()
    NETSTAT = auto()
    KERNEL_CSETBOUNDS_STATS = auto()
    KERNEL_STRUCT_STATS = auto()

    # Test only ID
    TEST_FAKE = auto()

    def __str__(self):
        return self.name


@dataclass
class DatasetConfig(TemplateConfig):
    """
    Define the parameters to run a dataset handler for the benchmark.
    This is shared between the actual benchmark parameters and auxiliary
    datasets.
    """
    #: Identifier (string) of a dataset to add to the run
    handler: DatasetName

    #: Extra options for the dataset handler, depend on the handler
    run_options: typing.Dict[str, any] = lazy_nested_config_field()


@dataclass
class PipelineInstanceConfig(Config):
    """
    Describe the instances on which the benchmarks will be run.
    This is used to generate the interal :class:`InstanceConfig` objects.
    """

    #: Common platform options, depend on the platforms used in the instances
    platform_options: PlatformOptions = field(default_factory=PlatformOptions)
    #: Instance descriptors for each instance to run
    instances: typing.List[InstanceConfig] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if len(self.instances) == 0:
            raise ValueError("There must be at least one instance configuration")


@dataclass
class CommonBenchmarkConfig(TemplateConfig):
    """
    Common benchmark configuration parameters.
    This is shared between the user-facing configuration file and the internal
    benchmark description.
    """

    #: The name of the benchmark
    name: str
    #: The number of iterations to run
    iterations: int
    #: Benchmark configuration
    benchmark: DatasetConfig
    #: Auxiliary data generators/handlers
    aux_dataset_handlers: typing.List[DatasetConfig] = field(default_factory=list)
    #: Number of iterations to drop to reach steady-state
    drop_iterations: int = 0
    #: Benchmark description, used for plot titles (can contain a template), defaults to :attr:`name`.
    desc: typing.Optional[str] = None
    #: Name of the benchmark output directory in the Guest instance OS filesystem
    remote_output_dir: Path = path_field("benchmark-output")
    #: Extra commands to run in the benchmark script. Keys in the dictionary are
    # benchmark setup steps 'pre_benchmark', 'pre_benchmark_iter', 'post_benchmark_iter',
    # 'post_benchmark'.
    command_hooks: typing.Dict[str, typing.List[str]] = field(default_factory=dict)

    @classmethod
    def from_common_conf(cls, other: "CommonBenchmarkConfig"):
        """
        Initialize a child config common fields.
        """
        initializer = {}
        for f in fields(CommonBenchmarkConfig):
            initializer[f.name] = getattr(other, f.name)
        return cls(**initializer)


@dataclass
class PipelineBenchmarkConfig(CommonBenchmarkConfig):
    """
    User-facing benchmark configuration.
    """
    #: Parameterized benchmark generator instructions. This should map (param_name => [values]).
    parameterize: typing.Dict[str, typing.List[any]] = field(default_factory=dict)


@dataclass
class BenchmarkRunConfig(CommonBenchmarkConfig):
    """
    Internal benchmark configuration.
    This represents a resolved benchmark run, associated to an ID and set of parameters.
    """
    #: Unique benchmark run identifier
    uuid: UUID = field(default_factory=uuid4)

    #: Unique benchmark group identifier, links benchmarks that run on the same instance
    g_uuid: typing.Optional[UUID] = None

    #: Benchmark parameters
    parameters: typing.Dict[str, any] = field(default_factory=dict)

    #: Instance configuration
    instance: typing.Optional[InstanceConfig] = None

    def __str__(self):
        if self.g_uuid and self.instance:
            return f"{self.name} ({self.uuid}/{self.g_uuid}) on {self.instance} params={self.parameters}"
        else:
            return f"unallocated {self.name} ({self.uuid}) params={self.parameters}"


@dataclass
class CommonSessionConfig(TemplateConfig):
    """
    Common session configuration.
    This is shared between the user-facing configuration file format and the
    internal session runfile.
    """
    #: Path to the SSH private key to use to access instances
    ssh_key: Path = Path("~/.ssh/id_rsa")

    #: Maximum number of concurrent instances that can be run (0 means unlimted)
    concurrent_instances: int = 0

    #: Allow reusing instances for multiple benchmark runs
    reuse_instances: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.ssh_key = self.ssh_key.resolve()

    @classmethod
    def from_common_conf(cls, other: "CommonSessionConfig"):
        """
        Initialize a child config common fields.
        """
        initializer = {}
        for f in fields(CommonSessionConfig):
            initializer[f.name] = getattr(other, f.name)
        return cls(**initializer)


@dataclass
class PipelineConfig(CommonSessionConfig):
    """
    Describe the benchmarks to run in the current benchplot session.
    Note that this configuration does not allow template substitution,
    the templates will be retained in the session instructions file so that
    the substitution can be replicated with a different user configuration every time.
    """
    #: Instances configuration, required
    instance_config: PipelineInstanceConfig = field(default_factory=PipelineInstanceConfig)

    #: Benchmark configuration, required
    benchmark_config: typing.List[PipelineBenchmarkConfig] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.ssh_key = self.ssh_key.resolve()
        if self.instance_config is None:
            raise ValueError("Missing instance_config")
        if len(self.benchmark_config) == 0:
            raise ValueError("Missing benchmark_config")


@dataclass
class SessionRunConfig(CommonSessionConfig):
    """
    Internal session configuration file, autogenerated from the pipeline configuration.
    This unwraps the benchmark parameterization and generates the full set of benchmarks
    to run with the associated instance configurations.
    """
    #: Session unique ID
    uuid: UUID = field(default_factory=uuid4)

    #: Session name, defaults to the session UUID
    name: typing.Optional[str] = None

    #: Benchmark run configuration, this is essentially the flattened benchmark matrix
    configurations: typing.List[BenchmarkRunConfig] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = str(self.uuid)

    @classmethod
    def generate(cls, mgr: "PipelineManager", config: PipelineConfig) -> "SessionRunConfig":
        """
        Generate a new :class:`SessionRunConfig` from a :class:`PipelineConfig`.
        Manual benchmark parameterization is supported by specifying multiple benchmarks with
        the same set of parameterize keys and different values.

        :param config: The :class:`PipelineConfig` to use.
        :return: A new session runfile configuration
        """
        session = SessionRunConfig.from_common_conf(config)
        mgr.logger.info("Create new session %s", session.uuid)

        # Collect benchmarks and the instances
        param_keys = None
        all_conf = []
        for conf in config.benchmark_config:
            if conf.parameterize:
                mgr.logger.debug("Found parameterized benchmark '%s'", conf.name)
                sorted_params = OrderedDict(conf.parameterize)
                keys = set(conf.parameterize.keys())
                for param_combination in it.product(*sorted_params.values()):
                    parameters = dict(zip(sorted_params.keys(), param_combination))
                    run_conf = BenchmarkRunConfig.from_common_conf(conf)
                    run_conf.parameters = parameters
                    all_conf.append(run_conf)
            else:
                # Only allow no parameterisation if there is a single config
                if len(config.benchmark_config) > 1:
                    mgr.logger.error(
                        "Multiple benchmark configurations must have a parameterization key to permit to distinguish them"
                    )
                    raise ValueError("Invalid configuration")
                mgr.logger.debug("Found benchmark %s", conf.name)
                all_conf = [BenchmarkRunConfig.from_common_conf(conf)]
                break
            if not param_keys:
                param_keys = keys
            elif param_keys != keys:
                mgr.logger.error("Benchmark parameter sets must be the same: expected %s found %s", param_keys, keys)
                raise ValueError("Invalid configuration")

        # Map instances to dataset group IDs
        group_uuids = [uuid4() for _ in config.instance_config.instances]

        # Now create all the full configurations by combining instance and benchmark configurations
        for run_conf in all_conf:
            for gid, inst_conf in zip(group_uuids, config.instance_config.instances):
                final_run_conf = BenchmarkRunConfig(**asdict(run_conf))
                # Generate new unique ID for the parameterized run
                final_run_conf.uuid = uuid4()
                final_run_conf.g_uuid = gid
                final_run_conf.instance = InstanceConfig(**asdict(inst_conf))
                session.configurations.append(final_run_conf)
        return session


@dataclass
class BenchplotUserConfig(Config):
    """
    User-environment configuration.
    This defines system paths for programs and source code we use.
    The main point of the user configuration is to make sessions portable,
    so that a session that is run on a machine can be analysed on another.
    """

    #: Path to write the cheri-benchplot sessions to
    session_path: Path = field(default_factory=Path.cwd)

    #: CHERI sdk path
    sdk_path: Path = path_field("~/cheri/cherisdk")

    #: CHERI projects build directory, expects the format from cheribuild
    build_path: Path = path_field("~/cheri/build")

    #: git repositories path
    src_path: Path = path_field("~/cheri")

    #: Path to the CHERI perfetto fork build directory
    perfetto_path: Path = path_field("~/cheri/cheri-perfetto/build")

    #: Path to openocd, will be inferred if missing (only relevant when running FPGA)
    openocd_path: typing.Optional[Path] = path_field("/usr/bin/openocd")

    #: Path to flamegraph.pl flamegraph generator
    flamegraph_path: typing.Optional[Path] = path_field("flamegraph.pl")

    #: Path to cheribuild, inferred from :attr:`src_path` if missing
    cheribuild_path: Path = field(init=False, default=None)

    #: Path to the CheriBSD sources, inferred from :attr:`src_path` if missing
    cheribsd_path: Path = field(init=False, default=None)

    #: Path to the qemu sources, inferred from :attr:`src_path` if missing
    qemu_path: Path = field(init=False, default=None)

    #: Enable extra debug output
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.sdk_path = self.sdk_path.expanduser().absolute()
        self.build_path = self.build_path.expanduser().absolute()
        self.src_path = self.src_path.expanduser().absolute()
        self.cheribuild_path = self.src_path / "cheribuild"
        self.cheribsd_path = self.src_path / "cheribsd"
        self.qemu_path = self.src_path / "qemu"
        # Try to autodetect openocd
        if self.openocd_path is None:
            self.openocd_path = shutil.which("openocd")

        self.session_path = self.session_path.expanduser().absolute()
        if not self.session_path.is_dir():
            raise ValueError("Session path must be a directory")


@dataclass
class AnalysisConfig(Config):
    #: List of plots/analysis steps to enable
    enable: typing.List[str] = field(default_factory=list)

    #: Tags for group enable
    enable_tags: typing.Set[str] = field(default_factory=set)

    #: Generate multiple split plots instead of combining
    split_subplots: bool = False

    #: Output formats
    plot_output_format: typing.List[str] = field(default_factory=lambda: ["pdf"])

    #: Constants to show in various plots, depending on the X and Y axes.
    # The dictionary maps parameters of the benchmark parameterisation to a dict
    # mapping description -> constant value
    parameter_constants: typing.Dict[str, dict] = field(default_factory=dict)

    #: Baseline dataset group id, defaults to the baseline instance uuid
    baseline_gid: typing.Optional[UUID] = None

    #: Use builtin symbolizer instead of addr2line
    use_buitin_symbolizer: bool=False

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.plot_output_format, list)
