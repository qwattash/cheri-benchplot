import dataclasses as dc
import itertools as it
import json
import re
import shutil
from collections import OrderedDict
from enum import Enum
from io import StringIO
from pathlib import Path
from textwrap import indent, wrap
from typing import Annotated, Any, Dict, List, Optional, Set, Type, get_args, get_origin
from uuid import UUID, uuid4

import marshmallow.fields as mfields
from git import Repo
from marshmallow import Schema, ValidationError, validates
from marshmallow.validate import And, ContainsOnly, OneOf, Predicate
from marshmallow_dataclass import class_schema
from typing_extensions import Self
from typing_inspect import is_generic_type, is_optional_type, is_union_type

from .error import ConfigTemplateBindError, ConfigurationError
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


def resolve_task_options(
    task_spec: str, task_options: dict, is_exec: bool = False
) -> Type["Config"]:
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
            logger.error(
                "Invalid task options, %s validation failed: %s",
                conf_class,
                err.normalized_messages(),
            )
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
            return Path(value).expanduser()
        except TypeError as ex:
            raise ValidationError(f"Invalid path {value}") from ex


#: Helper to validate that a PathField points to an existing regular file

validate_file_exists = And(
    Predicate("exists", error="File does not exist"),
    Predicate("is_file", error="Path is not regular file"),
)

validate_dir_exists = And(
    Predicate("exists", error="Directory does not exist"),
    Predicate("is_dir", error="Path is not a directory"),
)


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
            raise ValidationError(
                f"Task specifier {value} does not name any public tasks"
            )
        if len(matches) > 1:
            raise ValidationError(
                f"Task specifier {value} must identify an unique task"
            )

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
            raise ValidationError(
                f"Task specifier {value} does not name any public tasks"
            )


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
            return value.schema().dump(value)
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


class TemplateFieldProxy(mfields.Field):
    """
    Generic field that wraps another existing field automatically.

    This field will delay parsing until the template binding
    phase.
    """

    def __init__(self, wrapped):
        super().__init__(
            required=wrapped.required,
            load_default=wrapped.load_default,
            dump_default=wrapped.dump_default,
        )
        self._wrapped_field = wrapped

    def _has_template(self, value: Any) -> bool:
        if type(value) is not str:
            return False
        m = ConfigTemplateSpec.TEMPLATE_REGEX.search(value)
        return m is not None

    def _deserialize(self, value, attr, data, **kwargs):
        has_template = False
        if type(value) is list:
            has_template = any([self._has_template(v) for v in value])
        elif type(value) is dict:
            tmpl_keys = [self._has_template(k) for k in value.keys()]
            tmpl_vals = [self._has_template(v) for v in value.values()]
            has_template = any(tmpl_keys) or any(tmpl_vals)
        else:
            has_template = self._has_template(value)

        if has_template:
            return ConfigTemplateSpec(value)
        return self._wrapped_field._deserialize(value, attr, data, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        if isinstance(value, ConfigTemplateSpec):
            return value.value
        return self._wrapped_field._serialize(value, attr, obj, **kwargs)

    def _validate(self, value: Any):
        # Defer validation if we have a template element.
        if self._has_template(value):
            return
        self._wrapped_field._validate(value)


class BaseConfigSchema(Schema):
    """
    Base schema for Configs.

    This takes care of field wrapping to enable template resolution,
    while retaining type validation accuracy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Wrap all fields with the template proxy field
        replacement = {}
        load_replacement = {}
        dump_replacement = {}
        for field_name, field_obj in self.fields.items():
            if isinstance(field_obj, mfields.Nested):
                continue
            replacement[field_name] = TemplateFieldProxy(field_obj)
            if field_name in self.load_fields:
                load_replacement[field_name] = replacement[field_name]
            if field_name in self.dump_fields:
                dump_replacement[field_name] = replacement[field_name]
        self.fields.update(replacement)
        self.load_fields.update(load_replacement)
        self.dump_fields.update(dump_replacement)


# Helper type for dataclasses to use the PathField
ConfigTaskSpec = Annotated[str, TaskSpecField]
ConfigExecTaskSpec = Annotated[str, ExecTaskSpecField]
ConfigPath = Annotated[Path, PathField]
ConfigAny = Annotated[Any, mfields.Raw]
LazyNestedConfig = Annotated[dict[str, Any], LazyNestedConfigField]
UUIDStr = Annotated[str, UUIDField]


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

    def mark_resolved(self, count):
        self._resolved += count

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
            # self._resolved += 1
        return resolved


class ConfigTemplateSpec:
    """
    Object that wraps a template string in the configuration.

    This will be substituted and re-verified with the configuration
    after template value resolution.
    """

    TEMPLATE_REGEX = re.compile(r"(\{([a-zA-Z0-9_.-]+)(:([dfs]))?\})")

    def __init__(self, value: Any):
        self.value = value

    def _bind_value(self, value: Any, context: ConfigContext, dtype: type):
        """
        Bind a single value, this may occur within a collection.
        """
        # If not a string, there is nothing to bind at this point
        if type(value) is not str:
            return value

        # Query the context for the substitution
        # Now we can query the context for substitution keys to resolve.
        chunks = []
        last_match = 0
        n_bound = 0
        n_patterns = 0
        cast_to = None
        for m in self.TEMPLATE_REGEX.finditer(value):
            n_patterns += 1
            key = m.group(2)
            subst = context.find(key)
            if subst is None:
                continue
            n_bound += 1
            chunks.append(value[last_match : m.start()])
            chunks.append(str(subst))
            last_match = m.end()
            # Somewhat weirdly, we use the type hint from the last match to do the
            # forced type coercion. There should be a better way, but do I care?
            match m.group(4):
                case "d":
                    cast_to = int
                case "f":
                    cast_to = float
                case "s":
                    cast_to = str
                case _:
                    pass

        chunks.append(value[last_match:])
        result = "".join(chunks)

        # If we correctly substituted all patterns, failing the the type coercion
        # is an error and it should be reported as such.
        # If we did not substitute everything, there is no point in trying to coerce.
        if n_patterns == n_bound:
            result_value = self._coerce_value(result, dtype, cast_to)
            context.mark_resolved(n_patterns)
            return result_value
        else:
            raise ValueError("Incomplete template binding")

    def _coerce_union(
        self, value: str, candidate_dtypes: list[type], cast_dtype: type | None
    ):
        # If we get here, optional types have a value, so ignore the optional
        if cast_dtype is not None:
            if cast_dtype not in candidate_dtypes:
                config_logger.error(
                    "Invalid cast type %s for union with types %s",
                    cast_dtype,
                    candidate_dtypes,
                )
                raise ConfigTemplateBindError(
                    f"Failed union coercion to incompatible type {cast_dtype}"
                )
            return cast_dtype(value)

        for value_ty in candidate_dtypes:
            if value_ty is type(None):
                continue
            try:
                return value_ty(value)
            except ValueError:
                config_logger.debug(
                    "Trying to coerce %s to union alternative %s, failed",
                    value,
                    value_ty,
                )
                continue
        raise ValueError(f"Can not coerce {value} to any of the union types")

    def _coerce_value(self, value: str, dtype: type, cast_dtype: type | None = None):
        try:
            if is_union_type(dtype):
                args = get_args(dtype)
                return self._coerce_union(value, args, cast_dtype)
            origin_ty = get_origin(dtype)
            # If binding an Annotated type, convert it back to the base type
            if origin_ty is Annotated:
                dtype = get_args(dtype)[0]

            config_logger.debug("Requested cast %s", cast_dtype)
            if cast_dtype is not None:
                if dtype is Any:
                    return cast_dtype(value)
                # Verify that cast agrees with field type
                if dtype is not cast_dtype:
                    config_logger.error(
                        "Invalid cast type %s for field of type %s", cast_dtype, dtype
                    )
                    raise ConfigTemplateBindError(
                        f"Failed type coercion to incompatible type {cast_dtype}"
                    )
                return cast_dtype(value)
            # No specific cast requrested
            if dtype is Any:
                return value
            else:
                return dtype(value)
        except ValueError:
            config_logger.debug("Failed to coerce %s to %s", value, dtype)
            raise ConfigTemplateBindError(
                f"Failed type coercion for {value} to type {dtype}"
            )

    def _bind_string(
        self, context: ConfigContext, dtype: type
    ) -> str | float | int | Self:
        try:
            config_logger.debug("Bind scalar %s: %s", dtype, self.value)
            return self._bind_value(self.value, context, dtype)
        except ValueError:
            # Bail in the hope that we will come back to it later.
            return self

    def _bind_list(
        self, context: ConfigContext, dtype: type
    ) -> list[str | float | int | Self]:
        origin_ty = get_origin(dtype)
        assert origin_ty is List or origin_ty is list
        element_ty = get_args(dtype)[0]

        config_logger.debug("Bind list[%s]: %s", element_ty, self.value)
        result = []
        remaining = False
        for i, v in enumerate(self.value):
            try:
                result.append(self._bind_value(v, context, element_ty))
            except ValueError:
                remaining = True
                result.append(v)
            except ConfigTemplateBindError as err:
                err.location = f"[{i}]{err.location}"
                raise err
        if remaining:
            # Update the template state with the partial substitution
            self.value = result
            return self
        return result

    def _bind_dict(
        self, context: ConfigContext, dtype: type
    ) -> dict[str, str | float | int | Self]:
        origin_ty = get_origin(dtype)
        assert origin_ty is Dict or origin_ty is dict
        key_ty = get_args(dtype)[0]
        element_ty = get_args(dtype)[1]

        config_logger.debug("Bind dict[%s, %s]: %s", key_ty, element_ty, self.value)
        result = {}
        remaining = False
        for k, v in self.value.items():
            try:
                bound_key = self._bind_value(k, context, key_ty)
                bound_val = self._bind_value(v, context, element_ty)
                result[bound_key] = bound_val
            except ValueError:
                remaining = True
                result[k] = v
            except ConfigTemplateBindError as err:
                err.location = f"[{k}]{err.location}"
                raise err
        if remaining:
            # Update the template state with the partial substition
            self.value = result
            return self
        return result

    def bind(self, context: ConfigContext, dtype: type):
        """
        Attempt to substitute the current value with the given context.

        Currently, the template mini-syntax is simpler than the python
        :func:`format()`. This constrains the complexity, but should
        support most of the common use cases.

        The format string can specify a single key as "{my_key}", or
        a namespaced key in the context as "{my_key.other}".
        Explicit format conversions are specified as:
         - d: cast to integer
         - f: cast to float
         - s: cast to string

        We expect the inner value to be either:
         - a string
         - a list of items, possibly etherogeneous
         - a dict of items, possibly etherogeneous
        """
        vtype = type(self.value)
        if vtype is str:
            return self._bind_string(context, dtype)
        elif vtype is list:
            return self._bind_list(context, dtype)
        elif vtype is dict:
            return self._bind_dict(context, dtype)
        else:
            raise ConfigTemplateBindError(f"Invalid serialized value type: {vtype}")


def config_field(
    default, /, desc: str = None, field_kwargs: dict = None, **metadata
) -> dc.Field:
    """
    Helper to define configuration fields defaults

    If the first argument is not provided, there is no default value and we will
    enforce validation via marshmallow.
    """

    kwargs = dict(metadata=metadata)
    if field_kwargs is not None:
        kwargs.update(field_kwargs)
    if default == Config.REQUIRED:
        # No default, make the field required but pacify dataclass
        kwargs["metadata"]["required"] = True
        kwargs["default"] = None
    else:
        if callable(default):
            kwargs["default_factory"] = default
        else:
            kwargs["default"] = default
        if issubclass(type(default), Enum):
            kwargs["metadata"].setdefault("by_value", True)
    kwargs["metadata"]["metadata"] = dict(desc=desc)
    return dc.field(**kwargs)


def describe_type(dtype) -> tuple[str, list[Type["Config"]]]:
    """
    Given a field type annotation, provide a description of the accepted types.

    This returns a tuple with a textual description of the data type and
    a list containing all the referenced nested Config types.
    """
    if is_optional_type(dtype):
        desc_str, config_types = describe_type(get_args(dtype)[0])
        return f"<{desc_str}>?", config_types
    if is_union_type(dtype):
        desc = [describe_type(t) for t in get_args(dtype)]
        desc_str = " | ".join(map(lambda d: d[0], desc))
        config_types = list(it.chain(*map(lambda d: d[1], desc)))
        return desc_str, config_types
    if is_generic_type(dtype):
        origin = get_origin(dtype)
        config_types = []
        if origin is dict:
            kt, vt = get_args(dtype)
            kt_desc, conf_t = describe_type(kt)
            config_types.extend(conf_t)
            vt_desc, conf_t = describe_type(vt)
            config_types.extend(conf_t)
            return f"dict[{kt_desc}, {vt_desc}]", config_types
        if origin is list:
            lt_desc, config_types = describe_type(get_args(dtype)[0])
            return f"list[{lt_desc}]", config_types

    config_types = []
    if dc.is_dataclass(dtype) and issubclass(dtype, Config):
        config_types.append(dtype)
    type_name = dtype.__name__
    return str(type_name).split(".")[-1], config_types


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

    XXX use custom base schema with on_bind_field to provide session config information
    to analysis configurations? In particular we could detect and resolve the parameterisation
    there and compute defaults accordingly.
    """

    #: Marker for required fields
    def REQUIRED():
        raise ValueError("Attempt to default-construct required field")

    @classmethod
    def schema(cls):
        return class_schema(cls, base_schema=BaseConfigSchema)()

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
    def describe(cls, include_header=False):
        """
        Produce a human-readable description of the configuration.
        """
        nested_configs = set()
        desc = StringIO()
        if include_header:
            desc.write("=" * len(cls.__name__) + "\n")
            desc.write(cls.__name__ + "\n")
            desc.write("=" * len(cls.__name__) + "\n\n")

        for idx, field in enumerate(dc.fields(cls)):
            meta_ = field.metadata.get("metadata", {})
            help_ = meta_.get("desc")
            if help_:
                if idx > 0:
                    # Extra space to separate docstring from previous field
                    desc.write("\n")
                help_string = "".join(map(lambda line: f"#: {line}\n", wrap(help_)))
                desc.write(help_string)

            dtype, nested = describe_type(field.type)
            nested_configs.update(nested)
            desc.write(f"{field.name}: {dtype}")
            if field.metadata.get("required"):
                desc.write(" <required>")
            elif field.default != dc.MISSING:
                desc.write(f" = {field.default}")
            elif field.default_factory != dc.MISSING:
                desc.write(f" = {field.default_factory.__name__}()")
            else:
                logger.debug("Suspicious field documentation: %s: %s", cls, field.name)
            desc.write("\n")

        for config_type in nested_configs:
            desc.write("\n")
            desc.write(config_type.__name__ + "\n")
            desc.write("=" * len(config_type.__name__) + "\n")
            desc.write(config_type.__doc__ + "\n")
            desc.write(indent(config_type.describe(), " " * 4))

        return desc.getvalue()

    def __post_init__(self):
        return

    def _bind_field(
        self, context: ConfigContext, dtype: type, value: Any, loc: str, meta: dict
    ) -> Any:
        """
        Bind values for a template spec or nested configuration objects.
        """
        self.logger.debug("Bind field (%s) with dtype %s", loc, dtype)
        if isinstance(value, ConfigTemplateSpec):
            return value.bind(context, dtype)

        # If this is not a TemplateSpec, this means that we need to recurse into
        # it only if it is a container or union of nested Config types.

        def _has_nested_config_type(target_ty) -> bool:
            origin_ty = get_origin(target_ty)
            if is_union_type(target_ty):
                args_ty = get_args(target_ty)
                return any([_has_nested_config_type(ty) for ty in args_ty])
            elif origin_ty is Annotated:
                data_ty = get_args(target_ty)[0]
                return _has_nested_config_type(data_ty)
            elif is_generic_type(target_ty):
                args_ty = get_args(target_ty)
                return any([_has_nested_config_type(ty) for ty in args_ty])
            elif target_ty is Any:
                # Can not really know, conservatively say yes
                return True
            else:
                is_config = issubclass(target_ty, Config)
                # All Configs must be dataclasses
                assert not is_config or dc.is_dataclass(target_ty)
                return is_config

        def _bind_nested_config(target, nested_loc) -> Any:
            if isinstance(target, list):
                result = []
                for idx, val in enumerate(target):
                    result.append(_bind_nested_config(val, f"{nested_loc}[{idx}]"))
                return result
            elif isinstance(target, dict):
                result = {}
                for key, val in target.items():
                    # Note: can't have a nested config as a dictionary key
                    result[key] = _bind_nested_config(val, f"{nested_loc}[{key}]")
                return result
            elif isinstance(target, Config):
                return target.bind(context, nested_loc)
            else:
                # Undecidable any type does not hold a nested config
                return target

        if not _has_nested_config_type(dtype):
            return value

        self.logger.debug("(%s) recurse into nested generic", loc)
        # This time we base the recursion on the actual data value,
        # not on the expected type
        return _bind_nested_config(value, loc)

    def _bind_config(self, source: Self, context: ConfigContext, loc: str) -> Self:
        """
        Run a template substitution pass with the given substitution context.
        This will resolve template strings as "{foo}" for template key/value
        substitutions that have been registerd in the contexv via
        ConfigContext.register_template_subst() and leave the missing
        template parameters unchanged for later passes.
        """
        changes = {}
        self.logger.debug("(%s) -> Bind Config %s", loc, self.__class__.__name__)
        for f in dc.fields(self):
            if not f.init:
                continue
            field_loc = f"{loc}.{f.name}"
            field_value = getattr(source, f.name)
            meta = f.metadata.get("metadata", {})
            try:
                result = self._bind_field(context, f.type, field_value, field_loc, meta)
                self.logger.debug("(%s) <= %s", field_loc, result)
            except ConfigTemplateBindError as err:
                err.location = f"{field_loc}{err.location}"
                raise err
            except Exception as ex:
                msg = f"Failed to bind {f.name} with value {field_value}"
                raise ConfigurationError(msg) from ex
            if result:
                changes[f.name] = result
        self.logger.debug("(%s) -> Done binding %s", loc, self.__class__.__name__)
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
            logger.warning(
                "Configuration template binding exceeded recursion depth limit"
            )
            raise RuntimeError("Template substitution recursion limit")
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

    session_path: ConfigPath = config_field(
        Path.cwd,
        validate=validate_dir_exists,
        desc="Prefix path where cheri-benchplot sessions are created",
    )

    sdk_path: ConfigPath = config_field(
        Path("~/cheri/cherisdk"),
        validate=validate_dir_exists,
        desc="CHERI SDK path. This should point to the cherisdk directory, not the inner cherisdk/sdk.",
    )

    build_path: ConfigPath = config_field(
        Path("~/cheri/build"),
        validate=validate_dir_exists,
        desc="CHERI projects build directory, the directory layout should match cheribuild.",
    )

    src_path: ConfigPath = config_field(
        Path("~/cheri"),
        validate=validate_dir_exists,
        desc="Source directory, the directory layout should match cheribuild.",
    )

    openocd_path: ConfigPath = config_field(
        Path("/usr/bin/openocd"),
        desc="Path to openocd, will be inferred if missing (only relevant when running FPGA).",
    )

    flamegraph_path: ConfigPath = config_field(
        Path("flamegraph.pl"),
        desc="Path to BrendanGregg's flamegraph repository containing flamegraph.pl.",
    )

    rootfs_path: ConfigPath | None = config_field(
        None,
        desc="CHERI rootfs path, the directory layout should match cheribuild. "
        "If missing, it is inferred from sdk_path.",
    )

    cheribuild_path: ConfigPath | None = config_field(
        None, desc="Path to cheribuild. If missing, it is inferred from src_path."
    )

    cheribsd_path: ConfigPath | None = config_field(
        None,
        desc="Path to the CheriBSD sources. If missing, it is inferred from src_path.",
    )

    qemu_path: ConfigPath | None = config_field(
        None, desc="Path to the qemu sources. If missing, it is inferred from src_path."
    )

    llvm_path: ConfigPath | None = config_field(
        None,
        desc="Path to the Cheri LLVM sources. If missing, it is inferred from src_path.",
    )

    concurrent_workers: int | None = config_field(
        None, desc="Override the maximum number of workers to use."
    )

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
class ProfileConfig(Config):
    """
    Common profiling options.
    These are inteded to be embedded into benchmark task_options for those benchmarks
    that support some form of profiling.

    XXX-AM: These should probably be go away.
    The QEMU tracing targets are heavily outdated and should be reworked.
    """

    #: Run qemu with tracing enabled
    qemu_trace: Optional[str] = dc.field(
        default=None,
        metadata={"validate": OneOf([None, "perfetto", "perfetto-dynamorio"])},
    )

    #: Trace categories to enable for qemu-perfetto
    qemu_trace_categories: Optional[Set[str]] = None

    #: HWPMC performance counters modes
    hwpmc_trace: Optional[str] = dc.field(
        default=None, metadata={"validate": OneOf([None, "pmc", "profclock"])}
    )


class PlatformArch(Enum):
    """
    Describe the CPU platform where the benchmarks run.
    """

    RISCV64 = "riscv64"
    ARM64 = "arm64"

    def __str__(self):
        return self.value


class PlatformABI(Enum):
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
    """

    kernel: str = config_field(
        Config.REQUIRED, desc="Name of the kernel configuration file used"
    )
    name: str = config_field(
        Config.REQUIRED,
        desc="Unique name of the platform where the benchmark is run. "
        "This is used to populate the reserved `target` parameterisation axis.",
    )

    arch: PlatformArch = config_field(PlatformArch.RISCV64, desc="CPU Architecture.")

    kernelabi: PlatformABI = config_field(
        PlatformABI.PURECAP, desc="Kernel ABI identifier."
    )

    userabi: PlatformABI = config_field(
        PlatformABI.PURECAP, desc="User ABI identifier."
    )

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
            assert isinstance(self.task_options, Config), (
                "Task options must inherit from Config"
            )
        else:
            self.task_options = resolve_task_options(
                self.handler, self.task_options, is_exec=False
            )


@dc.dataclass
class ExecTargetConfig(Config):
    """
    Specify an execution task name.
    Note that the namespace of this task is also used to resolve compatible analysis tasks.
    """

    #: Task specifier with format indicated by :meth:`TaskRegistry.resolve_exec_task`
    handler: ConfigExecTaskSpec

    #: Extra options for the dataset handler, depend on the handler
    task_options: LazyNestedConfig = dc.field(
        default_factory=dict
    )  # Dict[str, ConfigAny] = lazy_nested_config_field()

    def __post_init__(self):
        super().__post_init__()
        # Resolve the lazy task options if this is not already a Config
        if dc.is_dataclass(self.task_options):
            assert isinstance(self.task_options, Config), (
                "Task options must inherit from Config"
            )
        else:
            self.task_options = resolve_task_options(
                self.handler, self.task_options, is_exec=True
            )


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
        metadata=dict(
            validate=ContainsOnly(["setup", "teardown", "iter_setup", "iter_teardown"])
        ),
    )

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

    matches: dict[str, str | int] = config_field(
        Config.REQUIRED,
        desc="Dictionary of parameterization key/values that enable this description.",
    )
    value: str | int = config_field(
        Config.REQUIRED, desc="Value to assign to the derived parameter."
    )


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
    parameterize: dict[str, list[Any]] = config_field(
        dict, desc="Parameterization axes. This maps <param_name> => [<values>]"
    )

    parameterize_options: ParamOptions = config_field(
        ParamOptions, desc="Parameterization tunables."
    )

    #: System configuration.
    #: Note that matching is done in-order, therefore the last entry may have
    #: `"matches": {}` to catch-all.
    system: list[SystemConfig] = config_field(
        list, desc="System configuration, this is used to generate the <target> axis."
    )

    @validates("parameterize")
    def validate_parameterize(self, data, **kwargs):
        if type(data) is not dict:
            raise ValidationError("Must be a dictionary")
        for pk in data.keys():
            if not re.fullmatch(r"[a-zA-Z0-9_]+", pk):
                raise ValidationError(
                    f"Parameterization key '{pk}' must be a valid python property name"
                )


@dc.dataclass
class BenchmarkRunConfig(CommonBenchmarkConfig):
    """
    Internal benchmark configuration.
    This represents a resolved benchmark run, associated to an ID and set of parameters.
    """

    #: Unique benchmark run identifier
    uuid: UUIDStr = config_field(
        Config.REQUIRED,
        desc="Unique identifier for the benchmar run. This identifier is stable across "
        "sessions with the same parameterization, even if the session UUID differs.",
    )

    #: Unique benchmark group identifier, links benchmarks that run on the same instance
    g_uuid: UUIDStr | None = config_field(None, desc="DEPRECATED")

    #: Benchmark parameters
    parameters: dict[str, ConfigAny] = config_field(
        dict, desc="Parameterisation tuple for this benchmark run."
    )

    #: Instance configuration
    instance: InstanceConfig | None = config_field(
        None, desc="Resolved host configuration"
    )

    def __str__(self):
        generators = [g.handler for g in self.generators]
        common_info = f"params={self.parameters} gen={generators}"
        if self.g_uuid and self.instance:
            return (
                f"{self.name} ({self.uuid}/{self.g_uuid}) on {self.instance} "
                + common_info
            )
        else:
            return f"unallocated {self.name} ({self.uuid}) " + common_info

    def __post_init__(self):
        super().__post_init__()
        params = [f"{k}={v}" for k, v in self.parameters.items()]
        config_logger.debug("Resolved BenchmarkRunConfig for %s", ", ".join(params))


class AssetImportAction(Enum):
    """
    Actions for importing assets into the session.
    """

    #: Copy directory or files from src to dst. The src path may contain a glob pattern.
    COPY = "copy"


@dc.dataclass
class AssetConfig(Config):
    """
    Describes an asset to import into the session and how.
    """

    action: AssetImportAction = config_field(
        Config.REQUIRED, by_value=True, desc="Import operation"
    )
    src: str = config_field(
        Config.REQUIRED, desc="Source for the asset, depends on the action"
    )
    dst: ConfigPath | None = config_field(
        None, desc="Destination relative to the session assets root"
    )


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
    version: str = config_field("1.0", desc="Session configuration version")

    #: Benchmark configuration, required
    benchmark_config: PipelineBenchmarkConfig = config_field(
        Config.REQUIRED, desc="Benchmark parameterisation configuration"
    )

    #: Assets to import into the session
    assets: dict[str, AssetConfig] = config_field(
        dict, desc="Configure assets to import into the session"
    )


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
    def _match_system_config(
        cls, params: dict, configs: list[SystemConfig]
    ) -> SystemConfig | None:
        for sys_conf in configs:
            if cls._match_params(params, sys_conf.matches):
                return sys_conf.host_system
        return None

    @classmethod
    def _match_derived_param(
        cls, params: dict, derived: list[DerivedParamSpec]
    ) -> str | None:
        for spec in derived:
            if cls._match_params(params, spec.matches):
                return spec.value
        return None

    @classmethod
    def generate_v1(
        cls, user_config: BenchplotUserConfig, config: PipelineConfig
    ) -> Self:
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
                logger.warning(
                    "Host system matcher '%s' has duplicate host system name '%s'",
                    sys_config.matches,
                    name,
                )
            host_system_names.add(name)
            host_system_uuids[name] = uuid4()

        # Generate all parameter combinations
        logger.debug("Generate parameterization for '%s'", bench_config.name)
        for combination in it.product(*sorted_params.values()):
            parameters = dict(zip(sorted_params.keys(), combination))
            if not cls._valid_parameterization(
                parameters, bench_config.parameterize_options
            ):
                continue

            run_config = BenchmarkRunConfig.from_common_conf(bench_config)
            run_config.parameters = parameters.copy()
            # Resolve system configuration
            if bench_config.system:
                host_system = cls._match_system_config(parameters, bench_config.system)
                if host_system is None:
                    logger.error(
                        "Missing system configuration for parameter combination %s",
                        parameters,
                    )
                    raise RuntimeError("Invalid configuration")
                if "target" not in parameters:
                    # Generate the target parameter, if not specified
                    run_config.parameters["target"] = host_system.name
            else:
                host_system = InstanceConfig.native()
                # Require the target parameter as it can not be generated
                if "target" not in parameters:
                    logger.error(
                        "Missing 'target' parameter in parameterization %s", parameters
                    )
                    raise RuntimeError("Invalid configuration")

            # Resolve custom derived parameters
            derived_specs = {}
            if specs := bench_config.parameterize_options.derived:
                derived_specs.update(specs)
            for name, spec in derived_specs.items():
                if name in parameters:
                    logger.error(
                        "Derived parameter name '%s' conflicts with root parameter",
                        name,
                    )
                    raise ConfigurationError("Invalid configuration")
                value = cls._match_derived_param(parameters, spec)
                if value is not None:
                    run_config.parameters[name] = value
                else:
                    logger.error(
                        "Unresolved derived parameter %s: %s (%s)",
                        name,
                        spec,
                        parameters,
                    )
                    raise ConfigurationError(
                        "Unresolved derived parameter %s for %s", name, parameters
                    )

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
