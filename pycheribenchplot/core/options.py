
import argparse as ap
import collections
import typing
import json
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass, replace, Field
from enum import Enum

def _template_safe(temp: str, **kwargs):
    try:
        return temp.format(**kwargs)
    except KeyError:
        return temp

@dataclass
class OptionConfig:
    @classmethod
    def from_json(cls, jsonpath):
        jsonfile = open(jsonpath, "r")
        confdata = json.load(jsonfile)
        return cls(**confdata)

    def __post_init__(self):
        for f in fields(self):
            origin = typing.get_origin(f.type)
            type_args = typing.get_args(f.type)
            if is_dataclass(f.type):
                value = getattr(self, f.name)
                if type(value) == dict:
                    setattr(self, f.name, f.type(**value))
            elif type(origin) == type:
                # standard type
                if (origin and issubclass(origin, collections.abc.Sequence) and
                    is_dataclass(type_args[0])):
                    items = [type_args[0](**item) for item in getattr(self, f.name)]
                    setattr(self, f.name, items)
                else:
                    setattr(self, f.name, origin(getattr(self, f.name)))
            elif type(origin) != typing._SpecialForm:
                setattr(self, f.name, f.type(getattr(self, f.name)))


class TemplateConfigContext:
    """
    Base class for context that can be used to bind TemplateConfig to.
    """
    def conf_template_params(self):
        return {}


@dataclass
class TemplateConfig(OptionConfig):

    def _bind_one(self, context, dtype, value):
        params = context.conf_template_params()
        if dtype == str:
            return _template_safe(value, **params)
        elif is_dataclass(dtype) and issubclass(dtype, TemplateConfig):
            return value.bind(context)
        elif dtype == Path:
            str_path = _template_safe(str(value), **params)
            print(str_path, )
            return Path(str_path)
        return value

    def bind_field(self, context, field: Field, value):
        origin = typing.get_origin(field.type)
        if is_dataclass(field.type):
            return self._bind_one(context, field.type, value)
        elif type(origin) == type:
            if origin and issubclass(origin, collections.abc.Sequence):
                arg_type = typing.get_args(field.type)[0]
                return [self._bind_one(context, arg_type, v) for v in value]
        else:
            return self._bind_one(context, field.type, value)

    def bind(self, context):
        changes = {}
        for f in fields(self):
            replaced = self.bind_field(context, f, getattr(self, f.name))
            if replaced:
                changes[f.name] = replaced
        return replace(self, **changes)
