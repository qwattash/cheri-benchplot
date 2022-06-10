import argparse as ap
import collections
import itertools
import json
import typing
import uuid
from dataclasses import (MISSING, Field, dataclass, field, fields, is_dataclass, replace)
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from dataclasses_json import DataClassJsonMixin, config
from typing_inspect import get_args, get_origin, is_generic_type


def _template_safe(temp: str, **kwargs):
    try:
        return temp.format(**kwargs)
    except KeyError:
        return temp


def path_field(default=None):
    return field(default=Path(default) if default else None, metadata=config(encoder=str, decoder=Path))


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, uuid.UUID):
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
        first_item = next(itertools.islice(mapping.values(), 1))
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
