import collections
import dataclasses as dc
import functools as ft
import itertools as it
import json
import logging
import re
import shutil
from collections import OrderedDict
from datetime import date, datetime
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from textwrap import indent, wrap
from typing import Any, Dict, List, Optional, Set, Type, Union
from uuid import UUID, uuid4

import marshmallow.fields as mfields
from git import Repo
from marshmallow import Schema, ValidationError, validates, validates_schema
from marshmallow.validate import And, ContainsOnly, OneOf, Predicate
from marshmallow_dataclass import NewType, class_schema
from typing_extensions import Self
from typing_inspect import get_args, get_origin, is_generic_type, is_union_type

from .error import ConfigurationError
from .util import new_logger, root_logger

# Global configuration logger
# We do not drag around a custom logger during configuration
logger = root_logger()
config_logger = new_logger("config")


def make_uuid() -> str:
    """
    Helper that generates an UUID string
    """
    return str(uuid4())


def resolve_task_options(task_spec: str, task_options: dict, is_exec: bool = False) -> Type["Config"]:
    """
    Helper to lazily coerce task options to the correct type.
    """
    # Need to lazily import this to avoid circular dependencies
    from .task import TaskRegistry

    config_logger.debug("Resolve task options for %s", task_spec)
    if is_exec:
        task_class = TaskRegistry.resolve_exec_task(task_spec)
        if not task_class:
            raise ConfigurationError(f"Invalid task spec: {task_spec}")
    else:
        matches = TaskRegistry.resolve_task(task_spec)
        if not matches:
            raise ConfigurationError(f"Invalid task spec: {task_spec}")
        if len(matches) > 1:
            raise ConfigurationError(f"Task handler should be unique: {task_spec}")
        task_class = matches[0]
    if task_class.task_config_class:
        conf_class = task_class.task_config_class
        config_logger.debug("Coerce %s options to %s", task_spec, conf_class.__name__)
        try:
            return conf_class.schema().load(task_options)
        except ValidationError as err:
            logger.error("Invalid task options, %s validation failed: %s", conf_class, err.normalized_messages())
            raise err
    return task_options


class PathField(mfields.Field):
    """
    Simple wrapper for pathlib.Path fields
    """
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return ""
        return str(value)

    def _deserialize(self, value, attr, data, **kwargs):
        if value == "":
            return None
        try:
            return Path(value).expanduser().resolve()
        except TypeError as ex:
            raise ValidationError(f"Invalid path {value}") from ex


#: Helper to validate that a PathField points to an existing regular file

validate_file_exists = And(Predicate("exists", error="File does not exist"),
                           Predicate("is_file", error="Path is not regular file"))

validate_dir_exists = And(Predicate("exists", error="Directory does not exist"),
                          Predicate("is_dir", error="Path is not a directory"))


class TaskSpecField(mfields.Field):
    """
    Field used to validate a public task specifier.

    See :meth:`TaskRegistry.resolve_task` for details on the format.
    """
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return ""
        return str(value)

    def _validate_taskspec(self, value):
        from .task import TaskRegistry
        matches = TaskRegistry.resolve_task(value)
        if not matches:
            raise ValidationError(f"Task specifier {value} does not name any public tasks")
        if len(matches) > 1:
            raise ValidationError(f"Task specifier {value} must identify an unique task")

    def _deserialize(self, value, attr, data, **kwargs):
        value = str(value)
        if value == "":
            raise ValidationError("Task specifier can not be blank")

        self._validate_taskspec(value)
        return value


class ExecTaskSpecField(TaskSpecField):
    """
    Field used to validate a public execution task name.

    See :meth:`TaskRegistry.resolve_exec_task` for details on the format.
    """
    def _validate_taskspec(self, value):
        from .task import TaskRegistry
        matches = TaskRegistry.resolve_exec_task(value)
        if not matches:
            raise ValidationError(f"Task specifier {value} does not name any public tasks")


class LazyNestedConfigField(mfields.Field):
    """
    Field used to mark lazily-resolved nested configurations.

    This is used for task-dependent configuration types that are not known statically.
    This fields treats its data as a dictionary, however, upon serialization, it accepts
    an arbitrary dataclass that is converted to a dict.
    Note that this automatically defaults to an empty dictionary.
    """
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return {}
        if dc.is_dataclass(value) and isinstance(value, Config):
            return type(value).schema().dump(value)
        return value

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return {}
        return value


class UUIDField(mfields.Field):
    """
    Field used to coerce values to a valid UUID string representation.
    """
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return str(value)

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            raise ValidationError(f"Invalid UUID '{value}'")
        try:
            uid = UUID(str(value))
        except ValueError:
            raise ValidationError(f"Invalid UUID x '{value}'")
        return str(uid)


# Helper type for dataclasses to use the PathField
ConfigTaskSpec = NewType("ConfigTaskSpec", str, field=TaskSpecField)
ConfigExecTaskSpec = NewType("ConfigExecTaskSpec", str, field=ExecTaskSpecField)
ConfigPath = NewType("ConfigPath", Path, field=PathField)
ConfigAny = NewType("ConfigAny", Any, field=mfields.Raw)
LazyNestedConfig = NewType("LazyNestedConfig", dict[str, Any], field=LazyNestedConfigField)
UUIDStr = NewType("UUIDStr", str, field=UUIDField)


class ConfigContext:
    """
    Base class for context that can be used to bind Config to.

    The context describes the mapping between keys and values.
    The substitution occurs in two steps:
    1. lookup the key in the static parameters
    2. search for any namespace matching the first part of a hierarchical key
       and recursively resolve the following parts of a dotted name.

    The "None" namespace is considered the default namespace in which to look up
    substitution keys.
    """
    def __init__(self):
        self._template_params = {}
        self._namespaces = {}
        self._resolved = 0

    @property
    def resolved_count(self) -> int:
        """
        Return the number of keys that have been resolved.

        This can be used to detect when we are done with recursive
        resolution.
        """
        return self._resolved

    def add_values(self, **kwargs):
        """
        Add one or more static template substitution values.

        :param **kwargs: Key-value pairs for template substitution
        """
        self._template_params.update(kwargs)

    def add_namespace(self, config: "Config", name: str = "__global__"):
        """
        Add a source configuration object that is used to look up referenced keys.

        :param config: An existing configuration object, may contain
            unresolved templates.
        :param name: Name to use in the template keys to refer to this namespace.
            For example, if we set name = "user" for :class:`BenchplotUserConfig`, the value for
            :attr:`BenchplotUserConfig.sdk_path` can be referenced as "{user.sdk_path}".
        """
        self._namespaces[name] = config

    def find(self, key: str) -> str | None:
        """
        Resolve a dot-separated key to the corresponding substitution value.

        :param key: The key to lookup.
        :return: The template substitution or None
        """
        if "." in key:
            # Lookup by namespace
            parts = key.split(".")
            value = self._namespaces[parts[0]]
            for name in parts[1:]:
                try:
                    value = getattr(value, name)
                except AttributeError:
                    config_logger.debug("Unresolved template '%s'", key)
                    value = None
                    break
            if isinstance(value, Config):
                config_logger.warning("Template key '%s' is not a leaf value")
                resolved = None
            else:
                resolved = value
        else:
            # Lookup global namespace if available
            if globalns := self._namespaces.get("__global__"):
                if hasattr(globalns, key):
                    resolved = getattr(globalns, key)
                else:
                    resolved = self._template_params.get(key)
            else:
                resolved = self._template_params.get(key)

        if resolved is None:
            config_logger.debug("Unresolved template '%s'", key)
        else:
            config_logger.debug("Resolved template '%s' => '%s'", key, resolved)
            self._resolved += 1
        return resolved


def config_field(default: Any, desc: str = None, field_kwargs: dict = None, **metadata) -> dc.Field:
    """
    Helper to define configuration fields defaults

    Use dataclasses.MISSING as default for required fields.
    """
    kwargs = dict(metadata=metadata)
    if field_kwargs is not None:
        kwargs.update(field_kwargs)
    if callable(default):
        kwargs["default_factory"] = default
    else:
        kwargs["default"] = default
    if issubclass(type(default), Enum):
        kwargs["metadata"].setdefault("by_value", True)
    kwargs["metadata"]["desc"] = desc
    return dc.field(**kwargs)


@dc.dataclass
class Config:
    """
    Base class for configuration data structure that support template substitution.
    Note that this should be used across the whole hierarchy of nested configurations
    to have the expected behaviour.
    Each field in this dataclass is deserialized from a json file, with support to
    nested configuration dataclasses.
    Types of the fields are normalized to the type annotation given in the dataclass.

    Note that the substitution process is designed to be incremental.
    Some template substitutions may become available later during initialization, therefore
    any unmatched template string will be retained unchanged after a call to
    :meth:`Config.bind`.
    """
    class Meta:
        ordered = True

    @classmethod
    def schema(cls):
        return class_schema(cls)()

    @classmethod
    def copy(cls, other):
        return cls.schema().load(cls.schema().dump(other))

    @classmethod
    def load_json(cls, jsonpath):
        with open(jsonpath, "r") as jsonfile:
            data = json.load(jsonfile)
        logger.debug("Parse configuration from %s", jsonpath)
        return cls.schema().load(data)

    @classmethod
    def describe(cls):
        """
        Produce a human-readable description of the configuration.
        """
        desc = StringIO()
        for idx, field in enumerate(dc.fields(cls)):
            help_ = field.metadata.get("desc")
            if help_:
                if idx > 0:
                    # Extra space to separate docstring from previous field
                    desc.write("\n")
                help_string = "".join(map(lambda l: f"#: {l}\n", wrap(help_)))
                desc.write(help_string)

            dtype = field.type if is_generic_type(field.type) else field.type.__name__
            dtype = str(dtype).split(".")[-1]
            desc.write(f"{field.name}: {dtype}")
            if field.default != dc.MISSING:
                desc.write(f" = {field.default}")
            elif field.default_factory != dc.MISSING:
                desc.write(f" = {field.default_factory.__name__}()")
            else:
                desc.write(" <required>")
            desc.write("\n")
        return desc.getvalue()

    def __post_init__(self):
        return

    def _bind_value(self, context: ConfigContext, dtype: Type, value: Any, loc: str) -> Any:
        """
        Resolve all template keys in this configuration value and produce the substituted value.

        :param context: The template configuration context
        :param dtype: Type annotation of the field
        :param value: Current value of the field. At this point, this is assumed to be
        a string-like object or a nested configuration.
        :return: The substituted value
        """
        if value is None:
            return None
        if dtype == ConfigAny or dtype == Any:
            dtype = type(value)

        if dc.is_dataclass(dtype) and issubclass(dtype, Config):
            return value.bind(context)
        if dtype == str:
            template = value
        elif dtype == Path or dtype == ConfigPath or (type(dtype) == type and issubclass(dtype, Path)):
            # Path-like object expected here
            template = str(value)
            dtype = Path
        else:
            # Anything else is assumed to never contain things to bind
            return value

        # Now we can query the context for substitution keys to resolve.
        chunks = []
        last_match = 0
        for m in re.finditer(r"\{([a-zA-Z0-9_.]+)\}", template):
            key = m.group(1)
            subst = context.find(key)
            if subst is None:
                continue
            chunks.append(template[last_match:m.start()])
            chunks.append(str(subst))
            last_match = m.end()

        chunks.append(template[last_match:])
        return dtype("".join(chunks))

    def _bind_field(self,
                    context: ConfigContext,
                    dtype: Type,
                    value: Any,
                    loc: str,
                    metadata: dict | None = None) -> Any:
        """
        Run the template substitution on a config field.
        If the field is a collection or a nested Config, we recursively bind
        each value.
        """
        config_logger.debug("(%s) <%s> -> coerce to %s", loc, self.__class__.__name__, dtype)
        if dc.is_dataclass(dtype) and issubclass(dtype, Config):
            # If it is a nested dataclass, just forward it
            config_logger.debug("Bind recurse into nested config %s", dtype.__name__)
            return value.bind(context, loc=loc)
        elif dtype == LazyNestedConfig:
            # If it is a lazy_nested_config field, treat this either as a dataclass or a dict
            if dc.is_dataclass(value):
                config_logger.debug("Bind lazy config as %s", type(value).__name__)
                return value.bind(context, loc=loc)
            else:
                config_logger.debug("Bind lazy config as dict %s", value)
                return self._bind_generic(context, dict, value, loc)
        elif is_generic_type(dtype) or is_union_type(dtype):
            # Recurse through the type
            config_logger.debug("Bind recurse into generic %s: %s", dtype, value)
            return self._bind_generic(context, dtype, value, loc)
        elif dtype == Any and (isinstance(value, dict) or isinstance(value, list)):
            # We don't have a strict type for the dict/list coercion, but we need
            # to recurse nontheless to bind template values within.
            config_logger.debug("Bind recurse into untyped generic %s: %s", type(value), value)
            return self._bind_generic(context, dtype, value, loc)
        else:
            # Handle as a single value
            config_logger.debug("Bind value '%s' to %s", value, dtype)
            return self._bind_value(context, dtype, value, loc)

    def _bind_generic(self, context: ConfigContext, dtype: Type, value: Any, loc: str) -> Any:
        """
        Recursively bind values in generic container types.
        """
        origin = get_origin(dtype)
        if is_union_type(dtype):
            args = get_args(dtype)
            if value is not None:
                return self._bind_field(context, type(value), value, loc)
            # Check if None is allowed
            for t in args:
                if t == type(None):
                    break
            else:
                raise ConfigurationError("None type is not allowed")
            return None
        elif type(value) == list:
            if origin is List or origin is list:
                inner_type = get_args(dtype)[0]
                config_logger.debug("Bind recurse into list[%s]", inner_type)
            else:
                inner_type = Any
                config_logger.debug("Bind recurse into list: can not determine item dtype")
            return [self._bind_field(context, inner_type, v, f"{loc}[{i}]") for i, v in enumerate(value)]
        elif type(value) == dict:
            if origin is Dict or origin is dict:
                key_type, inner_type = get_args(dtype)
                config_logger.debug("Bind recurse into dict[%s, %s]", key_type, inner_type)
            else:
                config_logger.debug("Bind recurse into dict: can not determine key, value dtypes")
                key_type = str
                inner_type = Any
            return {key: self._bind_field(context, inner_type, v, f"{loc}[{key}]") for key, v in value.items()}
        else:
            return self._bind_value(context, dtype, value, loc)

    def _bind_config(self, source: Self, context: ConfigContext, loc: str) -> Self:
        """
        Run a template substitution pass with the given substitution context.
        This will resolve template strings as "{foo}" for template key/value
        substitutions that have been registerd in the contexv via
        ConfigContext.register_template_subst() and leave the missing
        template parameters unchanged for later passes.
        """
        changes = {}
        config_logger.debug("(%s) -> Bind Config %s", loc, self.__class__.__name__)
        for f in dc.fields(self):
            if not f.init:
                continue
            try:
                metadata = f.metadata.get("metadata", {})
                field_loc = f"{loc}.{f.name}"
                replaced = self._bind_field(context, f.type, getattr(source, f.name), field_loc, metadata)
                config_logger.debug("(%s) <= %s", field_loc, replaced)
            except Exception as ex:
                raise ConfigurationError(f"Failed to bind {f.name} with value "
                                         f"{getattr(source, f.name)}: {ex}")
            if replaced:
                changes[f.name] = replaced
        config_logger.debug("(%s) -> Done binding %s", loc, self.__class__.__name__)
        return dc.replace(source, **changes)

    @property
    def logger(self):
        return config_logger

    def bind(self, context: ConfigContext, loc: str = "*") -> Self:
        """
        Substitute all templates until there is nothing else that we can substitute.

        :param context: The substitution context
        :return: A new configuration instance with the substituted values.
        """
        bound = self
        max_steps = 10
        last_matched = context.resolved_count
        for step in range(max_steps):
            bound = self._bind_config(bound, context, loc)
            if context.resolved_count == last_matched:
                break
            last_matched = context.resolved_count
        else:
            raise RuntimeError("LOOP")
            logger.warning("Configuration template binding exceeded recursion depth limit")
        return bound

    def emit_json(self) -> str:
        """
        Custom logic to emit json.
        This is required as in older python version pathlib objects are not serializable.
        """
        schema = self.schema()
        return json.dumps(schema.dump(self), indent=4)


@dc.dataclass
class BenchplotUserConfig(Config):
    """
    User-environment configuration.
    This defines system paths for programs and source code we use.
    The main point of the user configuration is to make sessions portable,
    so that a session that is run on a machine can be analysed on another.
    """

    #: Path to write the cheri-benchplot sessions to
    session_path: ConfigPath = dc.field(default_factory=Path.cwd, metadata=dict(validate=validate_dir_exists))

    #: CHERI sdk path
    sdk_path: ConfigPath = dc.field(default=Path("~/cheri/cherisdk"), metadata=dict(validate=validate_dir_exists))

    #: CHERI projects build directory, expects the format from cheribuild
    build_path: ConfigPath = dc.field(default=Path("~/cheri/build"), metadata=dict(validate=validate_dir_exists))

    #: git repositories path
    src_path: ConfigPath = dc.field(default=Path("~/cheri"), metadata=dict(validate=validate_dir_exists))

    #: Path to the CHERI perfetto fork build directory
    perfetto_path: ConfigPath = Path("~/cheri/cheri-perfetto/build")

    #: Path to openocd, will be inferred if missing (only relevant when running FPGA)
    openocd_path: ConfigPath = Path("/usr/bin/openocd")

    #: Path to BrendanGregg's flamegraph repository containing flamegraph.pl
    flamegraph_path: ConfigPath = Path("flamegraph.pl")

    #: CHERI rootfs path
    rootfs_path: Optional[ConfigPath] = None

    #: Path to cheribuild, inferred from :attr:`src_path` if missing
    cheribuild_path: Optional[ConfigPath] = dc.field(init=False, default=None)

    #: Path to the CheriBSD sources, inferred from :attr:`src_path` if missing
    cheribsd_path: Optional[ConfigPath] = dc.field(init=False, default=None)

    #: Path to the qemu sources, inferred from :attr:`src_path` if missing
    qemu_path: Optional[ConfigPath] = dc.field(init=False, default=None)

    #: Path to the Cheri LLVM sources, inferred from :attr:`src_path` if missing
    llvm_path: Optional[ConfigPath] = dc.field(init=False, default=None)

    #: Override the maximum number of workers to use
    concurrent_workers: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.sdk_path = self.sdk_path.expanduser().absolute()
        self.build_path = self.build_path.expanduser().absolute()
        self.src_path = self.src_path.expanduser().absolute()
        if self.rootfs_path is None:
            self.rootfs_path = self.sdk_path
        self.cheribuild_path = self.src_path / "cheribuild"
        self.cheribsd_path = self.src_path / "cheribsd"
        self.qemu_path = self.src_path / "qemu"
        self.llvm_path = self.src_path / "llvm-project"
        # Try to autodetect openocd
        if self.openocd_path is None:
            self.openocd_path = shutil.which("openocd")

        self.session_path = self.session_path.expanduser().absolute()
        if not self.session_path.is_dir():
            raise ValueError("Session path must be a directory")


@dc.dataclass
class CommonPlatformOptions(Config):
    """
    Base class for platform-specific options.
    This is internally used during benchmark dataset collection to
    set options for the instance that is to be run.
    """
    #: Number of cores in the system
    cores: int = 1

    #: The trace file used by default unless one of the datasets overrides it
    qemu_trace_file: Optional[ConfigPath] = None

    #: The trace file generated by interceptor
    qemu_interceptor_trace_file: Optional[ConfigPath] = None

    #: Run qemu with tracing enabled ('no', "perfetto", "perfetto-dynamorio")
    qemu_trace: str = dc.field(default="no", metadata={"validate": OneOf(["no", "perfetto", "perfetto-dynamorio"])})

    #: Trace categories to enable for qemu-perfetto
    qemu_trace_categories: Set[str] = dc.field(default_factory=set)

    #: VCU118 bios
    vcu118_bios: Optional[ConfigPath] = None

    #: IP to use for the VCU118 board
    vcu118_ip: str = "10.88.88.2"

    def __post_init__(self):
        assert self.qemu_trace in ["no", "perfetto", "perfetto-dynamorio"]


@dc.dataclass
class PlatformOptions(Config):
    """
    Platform options for an instance.
    This accepts the same fields as :class:`CommonPlatformOptions`, but
    will keep track of which are actively set in the configuration file,
    so that we can go look them up in the common options before setting
    a default value.
    """
    #: Number of cores in the system
    cores: Optional[int] = None

    #: The trace file used by default unless one of the datasets overrides it
    qemu_trace_file: Optional[ConfigPath] = None

    #: The trace file generated by interceptor
    qemu_interceptor_trace_file: Optional[ConfigPath] = None

    #: Run qemu with tracing enabled
    qemu_trace: str = dc.field(default="no", metadata={"validate": OneOf(["no", "perfetto", "perfetto-dynamorio"])})

    #: Trace categories to enable for qemu-perfetto
    qemu_trace_categories: Optional[Set[str]] = None

    #: VCU118 bios
    vcu118_bios: Optional[ConfigPath] = None

    #: IP to use for the VCU118 board
    vcu118_ip: Optional[str] = None

    def replace_common(self, common: CommonPlatformOptions):
        for f in dc.fields(self):
            if getattr(self, f.name) is None:
                setattr(self, f.name, getattr(common, f.name))


@dc.dataclass
class ProfileConfig(Config):
    """
    Common profiling options.
    These are inteded to be embedded into benchmark task_options for those benchmarks
    that support some form of profiling.
    """
    #: Run qemu with tracing enabled
    qemu_trace: Optional[str] = dc.field(default=None,
                                         metadata={"validate": OneOf([None, "perfetto", "perfetto-dynamorio"])})

    #: Trace categories to enable for qemu-perfetto
    qemu_trace_categories: Optional[Set[str]] = None

    #: HWPMC performance counters modes
    hwpmc_trace: Optional[str] = dc.field(default=None, metadata={"validate": OneOf([None, "pmc", "profclock"])})


class InstancePlatform(Enum):
    QEMU = "qemu"
    VCU118 = "vcu118"
    NATIVE = "native"

    def __str__(self):
        return self.value

    def is_fpga(self):
        return self == InstancePlatform.VCU118


class InstanceCheriBSD(Enum):
    NATIVE = "native"
    RISCV64_PURECAP = "riscv64-purecap"
    RISCV64_HYBRID = "riscv64-hybrid"
    MORELLO_PURECAP = "morello-purecap"
    MORELLO_HYBRID = "morello-hybrid"
    MORELLO_BENCHMARK = "morello-benchmark"

    def is_riscv(self):
        return (self == InstanceCheriBSD.RISCV64_PURECAP or self == InstanceCheriBSD.RISCV64_HYBRID)

    def is_morello(self):
        return (self == InstanceCheriBSD.MORELLO_PURECAP or self == InstanceCheriBSD.MORELLO_HYBRID
                or self == InstanceCheriBSD.MORELLO_BENCHMARK)

    def is_hybrid_abi(self):
        return (self == InstanceCheriBSD.RISCV64_HYBRID or self == InstanceCheriBSD.MORELLO_HYBRID)

    def is_purecap_abi(self):
        return (self == InstanceCheriBSD.RISCV64_PURECAP or self == InstanceCheriBSD.MORELLO_PURECAP)

    def is_benchmark_abi(self):
        return self == InstanceCheriBSD.MORELLO_BENCHMARK

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
    BENCHMARK = "benchmark"

    def __str__(self):
        return self.value


class InstanceUserABI(Enum):
    NOCHERI = "nocheri"
    HYBRID = "hybrid"
    PURECAP = "purecap"
    BENCHMARK = "benchmark"

    def __str__(self):
        return self.value


@dc.dataclass
class InstanceConfig(Config):
    """
    Configuration for a CheriBSD instance to run benchmarks on.
    XXX-AM May need a custom __eq__() if iterable members are added
    """
    #: Name of the kernel configuration file used
    kernel: str
    #: Optional name used for user-facing output such as plot legends
    name: str = dc.field(default_factory=lambda: str(uuid4()))
    #: Platform identifier, this affects the strategy used to run execution tasks
    platform: InstancePlatform = dc.field(default=InstancePlatform.QEMU, metadata={"by_value": True})
    #: Userspace ABI identifier
    cheri_target: InstanceCheriBSD = dc.field(default=InstanceCheriBSD.RISCV64_PURECAP, metadata={"by_value": True})
    #: Kernel ABI identifier
    kernelabi: InstanceKernelABI = dc.field(default=InstanceKernelABI.HYBRID, metadata={"by_value": True})
    #: User ABI identifier
    userabi: Optional[InstanceUserABI] = dc.field(default=None, metadata={"by_value": True})
    #: Is the kernel config name managed by cheribuild or is it an extra one
    #: specified via --cheribsd/extra-kernel-configs?
    cheribuild_kernel: bool = True
    #: Additional key-value parameters that can be used in benchmark templates to 'parameterize' on
    #: the platform axis. These are otherwise unused.
    parameters: Dict[str, ConfigAny] = dc.field(default_factory=dict)
    #: Internal fields, should not appear in the config file and are missing by default
    platform_options: PlatformOptions = dc.field(default_factory=PlatformOptions)

    @classmethod
    def native(cls):
        """
        Native instance configuration, which is used to indicate that we do not
        care where the analysis runs.
        E.g. this is used for static analysis and cross builds.
        """
        conf = cls(name="native",
                   kernel="unknown",
                   platform=InstancePlatform.NATIVE,
                   cheri_target=InstanceCheriBSD.NATIVE,
                   kernelabi=InstanceKernelABI.NOCHERI,
                   userabi=InstanceUserABI.NOCHERI,
                   cheribuild_kernel=False)
        return conf

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
        if self.userabi is None:
            # Infer user ABI from the cheri_target
            if self.cheri_target.is_hybrid_abi():
                self.userabi = InstanceUserABI.HYBRID
            elif self.cheri_target.is_purecap_abi():
                self.userabi = InstanceUserABI.PURECAP
            elif self.cheri_target.is_benchmark_abi():
                self.userabi = InstanceUserABI.BENCHMARK
            else:
                self.userabi = InstanceUserABI.NOCHERI

    def __str__(self):
        return f"{self.name}"


@dc.dataclass
class TaskTargetConfig(Config):
    """
    Specify an analysis task and associated options.
    """
    #: Task specifier with format indicated by :meth:`TaskRegistry.resolve_task`
    handler: ConfigTaskSpec

    #: Extra options for the dataset handler, depend on the handler
    task_options: LazyNestedConfig = dc.field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Resolve the lazy task options if this is not already a Config
        if dc.is_dataclass(self.task_options):
            assert isinstance(self.task_options, Config), f"Task options must inherit from Config"
        else:
            self.task_options = resolve_task_options(self.handler, self.task_options, is_exec=False)


@dc.dataclass
class ExecTargetConfig(Config):
    """
    Specify an execution task name.
    Note that the namespace of this task is also used to resolve compatible analysis tasks.
    """
    #: Task specifier with format indicated by :meth:`TaskRegistry.resolve_exec_task`
    handler: ConfigExecTaskSpec

    #: Extra options for the dataset handler, depend on the handler
    task_options: LazyNestedConfig = dc.field(default_factory=dict)  # Dict[str, ConfigAny] = lazy_nested_config_field()

    def __post_init__(self):
        super().__post_init__()
        # Resolve the lazy task options if this is not already a Config
        if dc.is_dataclass(self.task_options):
            assert isinstance(self.task_options, Config), f"Task options must inherit from Config"
        else:
            self.task_options = resolve_task_options(self.handler, self.task_options, is_exec=True)


@dc.dataclass
class PlotConfig(Config):
    """
    Plotting configuration.
    This is separated in case it needs to be propagated separately.
    """
    #: Parallel plotting (hacky and unstable)
    parallel: bool = False
    #: Output formats
    plot_output_format: List[str] = dc.field(default_factory=lambda: ["pdf"])

    def __post_init__(self):
        if self.parallel:
            from .plot import setup_matplotlib_hooks
            setup_matplotlib_hooks()


@dc.dataclass
class AnalysisConfig(Config):
    #: General plot configuration
    plot: PlotConfig = dc.field(default_factory=PlotConfig)

    #: Constants to show in various plots, depending on the X and Y axes.
    # The dictionary maps parameters of the benchmark parameterisation to a dict
    # mapping description -> constant value
    parameter_constants: Dict[str, dict] = dc.field(default_factory=dict)

    #: Baseline dataset identifier.
    #: This can be an UUID or a set of parameter key/values that uniquely identify
    #: a single benchmark run.
    baseline: Optional[UUIDStr | dict] = None

    #: Use builtin symbolizer instead of addr2line
    use_builtin_symbolizer: bool = True

    #: Specify analysis passes to run
    tasks: List[TaskTargetConfig] = dc.field(default_factory=list)


@dc.dataclass
class PipelineInstanceConfig(Config):
    """
    Describe the instances on which the benchmarks will be run.
    This is used to generate the interal :class:`InstanceConfig` objects.
    """

    #: Common platform options, depend on the platforms used in the instances
    platform_options: CommonPlatformOptions = dc.field(default_factory=CommonPlatformOptions)
    #: Instance descriptors for each instance to run
    instances: List[InstanceConfig] = dc.field(default_factory=list)


@dc.dataclass
class CommandHookConfig(Config):
    """
    Configuration for additional command hooks to generate the runner scripts.
    """

    #: Match parameterization key/values to enable this set of commands
    matches: Dict[str, ConfigAny] = dc.field(default_factory=dict)

    #: Set of commands to run when the matcher is satisfied.
    commands: List[str] = dc.field(default_factory=list)


@dc.dataclass
class CommonBenchmarkConfig(Config):
    """
    Common benchmark configuration parameters.
    This is shared between the user-facing configuration file and the internal
    benchmark description.
    """

    #: The name of the benchmark
    name: str

    #: The number of iterations to run
    iterations: int = 1

    #: Data generator tasks.
    #: These are used to produce the benchmark data and loading it during the analysis phase.
    generators: List[ExecTargetConfig] = dc.field(default_factory=list)

    #: Number of iterations to drop to reach steady-state
    drop_iterations: int = 0

    #: Benchmark description, used for plot titles (can contain a template), defaults to :attr:`name`.
    desc: Optional[str] = None

    #: Extra commands to run in the benchmark script.
    #: Keys in the dictionary are shell generator sections (see :class:`ScriptContext`).
    #: Valid hook groups are:
    #: setup -- global benchmark setup phase, before any iteration is run
    #: iter_setup -- per-iteration setup phase
    #: iter_teardown -- per-iteration teardown phase
    #: teardown -- global teardown phase, after all iterations are run
    #:
    #: Each hook group contains a list of matchers that allow to enable certain commands
    #: for specific sets of parameters.
    #:
    #: .. code-block:: python
    #:
    #:    {
    #:        "setup": [{
    #:            "matches": { "my_param": "my_value" },
    #:            "commands": ["echo 'hello world!'"]
    #:        }]
    #:    }
    #:
    command_hooks: Dict[str, List[CommandHookConfig]] = dc.field(
        default_factory=dict,
        metadata=dict(validate=ContainsOnly(["setup", "teardown", "iter_setup", "iter_teardown"])))

    @classmethod
    def from_common_conf(cls, other: "CommonBenchmarkConfig"):
        """
        Initialize a child config common fields.
        """
        initializer = {}
        for f in dc.fields(CommonBenchmarkConfig):
            initializer[f.name] = getattr(other, f.name)
        return cls(**initializer)


@dc.dataclass
class DerivedParamSpec(Config):
    """
    Derived parameter description
    """
    #: Dictionary of parameterization key/values that enable this description.
    matches: Dict[str, ConfigAny]
    #: Value to assign to the derived parameter
    value: str


@dc.dataclass
class ParamOptions(Config):
    """
    Configure parameterization behaviour.
    """
    #: List of parameter combinations to skip.
    #: For instance, the entry {"param1": "x"} will skip any combination
    #: where param1 assumes the value "x"
    skip: List[Dict[str, ConfigAny]] = dc.field(default_factory=list)

    #: Derived parameter substitutions that are made available
    #: during configuration time.
    #: Note that these will not be propagated to the dataframes.
    derived: Dict[str, List[DerivedParamSpec]] = dc.field(default_factory=dict)


@dc.dataclass
class SystemConfig(Config):
    """
    Host system to use for each specific group of parameters.
    """
    #: Match a set of key/values from the parameterization to which apply
    #: the system configuration.
    matches: Dict[str, ConfigAny]

    #: System configuration
    host_system: InstanceConfig


@dc.dataclass
class PipelineBenchmarkConfig(CommonBenchmarkConfig):
    """
    User-facing benchmark configuration.
    """
    #: Parameterized benchmark generator instructions. This should map
    #: (param_name => [values]).
    #: Note that there must be a 'target' parameter axis, otherwise it is implied
    #: and generated from system configurations.
    parameterize: Dict[str, List[ConfigAny]] = dc.field(default_factory=dict)

    #: Parameterization options
    parameterize_options: Optional[ParamOptions] = None

    #: System configuration.
    #: Note that matching is done in-order, therefore the last entry may have
    #: `"matches": {}` to catch-all.
    system: List[SystemConfig] = dc.field(default_factory=list)

    @validates("parameterize")
    def validate_parameterize(self, data, **kwargs):
        if (type(data) != dict):
            raise ValidationError("Must be a dictionary")
        for pk in data.keys():
            if not re.fullmatch(r"[a-zA-Z0-9_]+", pk):
                raise ValidationError(f"Parameterization key '{pk}' must be a valid python property name")


@dc.dataclass
class BenchmarkRunConfig(CommonBenchmarkConfig):
    """
    Internal benchmark configuration.
    This represents a resolved benchmark run, associated to an ID and set of parameters.
    """
    #: Unique benchmark run identifier
    uuid: UUIDStr = dc.field(default_factory=make_uuid)

    #: Unique benchmark group identifier, links benchmarks that run on the same instance
    g_uuid: Optional[UUIDStr] = None

    #: Benchmark parameters
    parameters: Dict[str, ConfigAny] = dc.field(default_factory=dict)

    #: Instance configuration
    instance: Optional[InstanceConfig] = None

    def __str__(self):
        generators = [g.handler for g in self.generators]
        common_info = f"params={self.parameters} gen={generators}"
        if self.g_uuid and self.instance:
            return f"{self.name} ({self.uuid}/{self.g_uuid}) on {self.instance} " + common_info
        else:
            return f"unallocated {self.name} ({self.uuid}) " + common_info

    def __post_init__(self):
        super().__post_init__()
        params = [f"{k}={v}" for k, v in self.parameters.items()]
        config_logger.debug("Resolved BenchmarkRunConfig for %s", ", ".join(params))


@dc.dataclass
class CommonSessionConfig(Config):
    """
    Common session configuration.
    This is shared between the user-facing configuration file format and the
    internal session runfile.
    """
    #: Path to the SSH private key to use to access instances
    ssh_key: ConfigPath = Path("~/.ssh/id_rsa")

    #: Failure policy for workers. If true, when an worker encounters an error, it
    #: causes the scheduler to stop executing tasks and cleaup. If false failures
    #: are tolerated and the tasks that depend on the failed one will be removed
    #: from the schedule.
    abort_on_failure: bool = True

    #: Maximum number of concurrent instances that can be run (0 means unlimted)
    concurrent_instances: int = 0

    #: Maximum number of concurrent workers
    concurrent_workers: int = 0

    #: Allow reusing instances for multiple benchmark runs
    reuse_instances: bool = False

    #: Extract symbols with elftools instead of llvm
    use_builtin_symbolizer: bool = True

    #: Path to the session directory tree in the benchmark host
    remote_session_path: ConfigPath = Path("/opt/benchmark-output")

    #: Default analysis task configuration
    analysis_config: AnalysisConfig = dc.field(default_factory=AnalysisConfig)

    #: Tar and base64-encode the benchmark results to stdout.
    #: This is useful if the benchmark host does not support a filesystem output.
    bundle_results: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.ssh_key = self.ssh_key.resolve()

    @classmethod
    def from_common_conf(cls, other: "CommonSessionConfig"):
        """
        Initialize a child config common fields.
        """
        initializer = {}
        for f in dc.fields(CommonSessionConfig):
            initializer[f.name] = getattr(other, f.name)
        return cls(**initializer)


@dc.dataclass
class PipelineConfig(CommonSessionConfig):
    """
    Describe the benchmarks to run in the current benchplot session.
    Note that this configuration does not allow template substitution,
    the templates will be retained in the session instructions file so that
    the substitution can be replicated with a different user configuration every time.
    """
    #: Configuration format version
    version: str = "0.0",

    #: Benchmark configuration, required
    benchmark_config: Union[PipelineBenchmarkConfig, List[PipelineBenchmarkConfig]] = dc.field(default_factory=list)


@dc.dataclass
class SessionRunConfig(CommonSessionConfig):
    """
    Internal session configuration file, autogenerated from the pipeline configuration.
    This unwraps the benchmark parameterization and generates the full set of benchmarks
    to run with the associated instance configurations.
    """
    #: Session unique ID
    uuid: UUIDStr = dc.field(default_factory=make_uuid)

    #: Snapshots of the relevant git repositories that should be syncronised to this session
    #: These are taken when the session is created.
    git_sha: Dict[str, str] = dc.field(default_factory=dict)

    #: Session name, defaults to the session UUID
    name: Optional[str] = None

    #: Benchmark run configuration, this is essentially the flattened benchmark matrix
    configurations: List[BenchmarkRunConfig] = dc.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = str(self.uuid)

    @validates_schema
    def validate_configuration(self, data, **kwargs):
        # Check that the concurrent instances configuration is compatible with the platform
        if data["concurrent_instances"] != 1:
            for bench_config in data["configurations"]:
                if bench_config.instance.platform == InstancePlatform.VCU118:
                    raise ValidationError("Running on VCU118 instances requires concurrent_instances=1",
                                          "concurrent_instances")

    @classmethod
    def _resolve_template(cls, session_config: Self) -> Self:
        """
        Resolve all template substitutions, except the user_config substitutions that must
        be resolved at session-load time.

        This produces a new session configuration with the substituted values.
        Note that this resolves first the common session template keys, then per-dataset
        configuration keys are introduced and resolved.
        """
        ctx = ConfigContext()
        # Default namespace
        ctx.add_namespace(session_config)
        new_config = session_config.bind(ctx)

        # Replace the default namespace as we have an updated version
        ctx.add_namespace(new_config)
        # Now scan through all the configurations and subsitute per-dataset fields
        new_bench_conf = []
        for bench_conf in new_config.configurations:
            ctx.add_namespace(bench_conf, "benchmark")
            # Shorthand for the corresponding instance configuration
            ctx.add_namespace(bench_conf.instance, "instance")
            # Register parameterization keys, note that these may happen shadow
            # other names, but let it be for now
            ctx.add_values(**bench_conf.parameters)
            # Finally do the binding
            new_bench_conf.append(bench_conf.bind(ctx))
        new_config.configurations = new_bench_conf
        return new_config

    @classmethod
    def _match_params(cls, params: dict[str, Any], matcher: dict[str, str]) -> bool:
        """
        Check whether a given set of parameters satisfies a matcher dictionary.

        This checks that every element in the matcher is present and satisfies the
        match expression.
        An empty matcher acts as catch-all.
        """
        for mkey, mexpr in matcher.items():
            if mkey not in params:
                return False
            if params[mkey] != mexpr:
                return False
        return True

    @classmethod
    def _valid_parameterization(cls, params: dict, opts: ParamOptions | None) -> bool:
        """
        Check whether the given set of parameters is allowed.
        """
        if opts is None:
            return True
        for skip in opts.skip:
            if cls._match_params(params, skip):
                return False
        return True

    @classmethod
    def _match_system_config(cls, params: dict, configs: list[SystemConfig]) -> SystemConfig | None:
        for sys_conf in configs:
            if cls._match_params(params, sys_conf.matches):
                return sys_conf.host_system
        return None

    @classmethod
    def _match_derived_param(cls, params: dict, derived: list[DerivedParamSpec]) -> str | None:
        for spec in derived:
            if cls._match_params(params, spec.matches):
                return spec.value
        return None

    @classmethod
    def generate_v1(cls, user_config: BenchplotUserConfig, config: PipelineConfig) -> Self:
        """
        Generate a new :class:`SessionRunConfig` from a :class:`PipelineConfig`.

        The benchmark configuration parameterization is resolved and we generate
        a BenchmarkRunConfig for each allowed parameter combination.
        System configurations are selected here and associated to each benchmark run.

        If the "target" parameterization level is explicitly present, use its value as is.
        If the "target" level is not given, it is generated by assigning it
        the system configuration UUID.

        :param user_config: The user configuration for the local machine setup.
        :param config: The :class:`PipelineConfig` to use.
        :return: A new session runfile configuration
        """
        session = SessionRunConfig.from_common_conf(config)
        logger.info("Create new session %s", session.uuid)

        bench_config = config.benchmark_config
        param_levels = list(bench_config.parameterize.keys())
        sorted_params = OrderedDict(bench_config.parameterize)

        # Host system names must be unique, warn if this is not the case.
        # It might be useful to have multiple matchers for the same system, so
        # don't make this an error.
        host_system_names = set()
        # Assign dataset g_uuids for backward-compatibility only
        host_system_uuids = {}
        for sys_config in bench_config.system:
            name = sys_config.host_system.name
            if name in host_system_names:
                logger.warning("Host system matcher '%s' has duplicate host system name '%s'", sys_config.matches, name)
            host_system_names.add(name)
            host_system_uuids[name] = uuid4()

        # Generate all parameter combinations
        logger.debug("Generate parameterization for '%s'", bench_config.name)
        for combination in it.product(*sorted_params.values()):
            parameters = dict(zip(sorted_params.keys(), combination))
            if not cls._valid_parameterization(parameters, bench_config.parameterize_options):
                continue

            run_config = BenchmarkRunConfig.from_common_conf(bench_config)
            run_config.parameters = parameters.copy()
            # Resolve system configuration
            if bench_config.system:
                host_system = cls._match_system_config(parameters, bench_config.system)
                if host_system is None:
                    logger.error("Missing system configuration for parameter combination %s", parameters)
                    raise RuntimeError("Invalid configuration")
                if "target" not in parameters:
                    # Generate the target parameter, if not specified
                    run_config.parameters["target"] = host_system.name
            else:
                host_system = InstanceConfig.native()
                # Require the target parameter as it can not be generated
                if "target" not in parameters:
                    logger.error("Missing 'target' parameter in parameterization %s", parameters)
                    raise RuntimeError("Invalid configuration")

            # Resolve custom derived parameters
            derived_specs = bench_config.parameterize_options.derived if bench_config.parameterize_options else dict()
            for name, spec in derived_specs.items():
                if name in parameters:
                    logger.error("Derived parameter name '%s' conflicts with root parameter", name)
                    raise RuntimeError("Invalid configuration")
                if value := cls._match_derived_param(parameters, spec):
                    run_config.parameters[name] = value

            run_config.uuid = uuid4()
            run_config.instance = InstanceConfig.copy(host_system)
            session.configurations.append(run_config)

        # Snapshot all repositories we care about, if they are present.
        # Note that we should support snapshot hooks in the configured tasks.
        def snap_head(repo_path, key):
            if repo_path.exists():
                session.git_sha[key] = Repo(repo_path).head.commit.hexsha
            else:
                logger.warning("No %s repository, skip SHA snapshot", key)

        snap_head(user_config.cheribuild_path, "cheribuild")
        snap_head(user_config.cheribsd_path, "cheribsd")
        snap_head(user_config.qemu_path, "qemu")
        snap_head(user_config.llvm_path, "llvm")

        # Now that we are done with generating the configuration, resolve all
        # templates that do not involve the user configuration
        return cls._resolve_template(session)

    @classmethod
    def generate(cls, user_config: BenchplotUserConfig, config: PipelineConfig) -> Self:
        if config.version == "1.0":
            return cls.generate_v1(user_config, config)
        logger.error("Invalid configuration version %s", config.version)
        raise RuntimeError("Invalid configuration version")
