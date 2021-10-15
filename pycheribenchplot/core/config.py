import argparse as ap
import collections
import itertools
import typing
import json
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass, replace, Field, field
from enum import Enum

from dataclasses_json import DataClassJsonMixin, config


def _template_safe(temp: str, **kwargs):
    try:
        return temp.format(**kwargs)
    except KeyError:
        return temp


def path_field(default=None):
    return field(default=Path(default) if default else None, metadata=config(encoder=str, decoder=Path))


@dataclass
class Config(DataClassJsonMixin):
    @classmethod
    def load_json(cls, jsonpath):
        with open(jsonpath, "r") as jsonfile:
            return super().from_json(jsonfile.read())
        # confdata = json.load(jsonfile)
        # return cls(**confdata)

    def _normalize_sequence(self, f, sequence):
        type_args = typing.get_args(f.type)
        item_type = type_args[0]
        if len(sequence) == 0:
            # Nothing to normalize
            return sequence
        if is_dataclass(item_type) and not is_dataclass(sequence[0]):
            items = [item_type(**item) for item in getattr(self, f.name)]
            setattr(self, f.name, items)

    def _normalize_mapping(self, f, mapping):
        type_args = typing.get_args(f.type)
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
            origin = typing.get_origin(f.type)
            type_args = typing.get_args(f.type)
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
            elif type(origin) != typing._SpecialForm:
                # Not a typing class (e.g. Union)
                if issubclass(f.type, Path):
                    setattr(self, f.name, Path(getattr(self, f.name)).expanduser())


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
        origin = typing.get_origin(f.type)
        if is_dataclass(f.type):
            return self._bind_one(context, f.type, value)
        elif type(origin) == type:
            if issubclass(origin, collections.abc.Sequence):
                arg_type = typing.get_args(f.type)[0]
                return [self._bind_one(context, arg_type, v) for v in value]
            if issubclass(origin, collections.abc.Mapping):
                arg_type = typing.get_args(f.type)[1]
                return {key: self._bind_one(context, arg_type, v) for key, v in value.items()}
        else:
            return self._bind_one(context, f.type, value)

    def bind(self, context):
        changes = {}
        for f in fields(self):
            replaced = self.bind_field(context, f, getattr(self, f.name))
            if replaced:
                changes[f.name] = replaced
        return replace(self, **changes)
