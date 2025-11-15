from dataclasses import dataclass
from typing import Annotated

import pytest
from marshmallow import ValidationError
from marshmallow import fields as mf
from marshmallow import validate as mv

from pycheribenchplot.core.config import (
    Config,
    ConfigAny,
    ConfigContext,
    ConfigTemplateSpec,
    config_field,
)
from pycheribenchplot.core.error import ConfigTemplateBindError


@dataclass
class DemoConfigDefault(Config):
    str_opt: str | None = config_field(None)
    int_opt: int | None = config_field(None)
    list_opt: list[str] | None = config_field(None)
    dict_opt: dict[str, str] | None = config_field(None)
    str_val: str = config_field("default")
    int_val: int = config_field(10)
    list_val: list[str] = config_field(list)
    dict_val: dict[str, str] = config_field(dict)


@dataclass
class DemoConfigRequired(Config):
    something: str | None = config_field(None)
    str_val: str = config_field(Config.REQUIRED)
    int_val: int = config_field(Config.REQUIRED)
    list_val: list[str] = config_field(Config.REQUIRED)
    dict_val: dict[str, str] = config_field(Config.REQUIRED)


@dataclass
class DemoConfigNonString(Config):
    int_val: int = config_field(0)
    float_val: float = config_field(0.0)
    list_val: list[int] = config_field(list)
    dict_val: dict[str, int] = config_field(dict)


@dataclass
class DemoNested(Config):
    str_val: str | None = config_field(None)
    nested_val: DemoConfigDefault | None = config_field(None)


def test_config_default():
    default = DemoConfigDefault.schema().load({})

    assert default.str_opt is None
    assert default.int_opt is None
    assert default.list_opt is None
    assert default.dict_opt is None
    assert default.str_val == "default"
    assert default.int_val == 10
    assert type(default.list_val) is list and len(default.list_val) == 0
    assert type(default.dict_val) is dict and len(default.dict_val) == 0


def test_config_required():
    with pytest.raises(ValidationError):
        DemoConfigRequired.schema().load({})

    with pytest.raises(ValidationError) as e:
        DemoConfigRequired.schema().load(
            {"int_val": 100, "list_val": ["a"], "dict_val": {"x": "y"}}
        )
    errs = e.value.normalized_messages()
    assert "str_val" in errs

    with pytest.raises(ValidationError) as e:
        DemoConfigRequired.schema().load(
            {"str_val": "my_string", "list_val": ["a"], "dict_val": {"x": "y"}}
        )
    errs = e.value.normalized_messages()
    assert "int_val" in errs

    with pytest.raises(ValidationError) as e:
        DemoConfigRequired.schema().load(
            {"str_val": "my_string", "int_val": 100, "dict_val": {"x": "y"}}
        )
    errs = e.value.normalized_messages()
    assert "list_val" in errs

    with pytest.raises(ValidationError) as e:
        DemoConfigRequired.schema().load(
            {"str_val": "my_string", "int_val": 100, "list_val": ["a"]}
        )
    errs = e.value.normalized_messages()
    assert "dict_val" in errs


def test_type_validation():
    with pytest.raises(ValidationError) as e:
        _config = DemoConfigDefault.schema().load({"str_val": 10})
    errs = e.value.normalized_messages()
    assert "str_val" in errs

    with pytest.raises(ValidationError) as e:
        _config = DemoConfigDefault.schema().load({"int_val": "ten"})
    errs = e.value.normalized_messages()
    assert "int_val" in errs

    with pytest.raises(ValidationError) as e:
        _config = DemoConfigDefault.schema().load({"list_val": 10})
    errs = e.value.normalized_messages()
    assert "list_val" in errs

    with pytest.raises(ValidationError) as e:
        _config = DemoConfigDefault.schema().load({"list_val": [10]})
    errs = e.value.normalized_messages()
    assert "list_val" in errs

    with pytest.raises(ValidationError) as e:
        _config = DemoConfigDefault.schema().load({"dict_val": 10})
    errs = e.value.normalized_messages()
    assert "dict_val" in errs

    with pytest.raises(ValidationError) as e:
        _config = DemoConfigDefault.schema().load({"dict_val": {"x": 10}})
    errs = e.value.normalized_messages()
    assert "dict_val" in errs

    with pytest.raises(ValidationError) as e:
        _config = DemoConfigDefault.schema().load({"dict_val": {10: "x"}})
    errs = e.value.normalized_messages()
    assert "dict_val" in errs


def test_config_simple_template():
    data = {
        "str_val": "{param0}",
        "list_val": ["a", "b", "c", "{param1}"],
        "dict_val": {"a": "{param2}", "{param3}": "c"},
    }
    cc = ConfigContext()
    cc.add_values(param0="foo", param1="bar", param2="baz", param3="key")

    config = DemoConfigDefault.schema().load(data)
    bound = config.bind(cc)

    assert bound.str_val == "foo"
    assert bound.list_val == ["a", "b", "c", "bar"]
    assert bound.dict_val == {"a": "baz", "key": "c"}


def test_config_template_dtype_substitution():
    data = {
        "int_val": "{param0}",
        "float_val": "{param1}",
        "list_val": [1, 2, 3, "{param2}"],
        "dict_val": {"a": "{param3}"},
    }
    cc = ConfigContext()
    cc.add_values(param0=100, param1=10.1, param2=300, param3=400)

    config = DemoConfigNonString.schema().load(data)
    bound = config.bind(cc)

    assert type(bound.int_val) is int and bound.int_val == 100
    assert type(bound.float_val) is float and bound.float_val == 10.1
    assert bound.list_val == [1, 2, 3, 300]
    assert bound.dict_val == {"a": 400}


def test_config_template_dtype_cast_validation():
    cc = ConfigContext()
    cc.add_values(
        param0="invalid_int",
        param1="invalid_float",
        param2="invalid_int",
        param3="invalid_int",
    )

    data = {"int_val": "{param0}"}
    config = DemoConfigNonString.schema().load(data)
    with pytest.raises(ConfigTemplateBindError) as err:
        config.bind(cc)

    assert err.value.location == "*.int_val"


def test_config_template_optional():
    cc = ConfigContext()
    cc.add_values(param0="foo")

    data = {"str_opt": "{param0}"}
    config = DemoConfigDefault.schema().load(data)
    bound = config.bind(cc)

    assert bound.str_opt == "foo"


def test_config_template_union():
    @dataclass
    class TestConfig(Config):
        simple_union: int | str | None = config_field(None)
        nested_union: list[int | str] = config_field(list)

    cc = ConfigContext()
    cc.add_values(param0="foo", param1=100)

    data = {"simple_union": "{param0}"}
    config = TestConfig.schema().load(data)
    bound = config.bind(cc)
    assert bound.simple_union == "foo"

    data = {"simple_union": "{param1}"}
    config = TestConfig.schema().load(data)
    bound = config.bind(cc)
    assert bound.simple_union == 100

    data = {"nested_union": [10, "bar", "{param0}"]}
    config = TestConfig.schema().load(data)
    bound = config.bind(cc)
    assert bound.nested_union == [10, "bar", "foo"]

    data = {"nested_union": [10, "bar", "{param1}"]}
    config = TestConfig.schema().load(data)
    bound = config.bind(cc)
    assert bound.nested_union == [10, "bar", 100]


def test_config_incomplete_template_dump():
    cc = ConfigContext()
    cc.add_values(param0="foo", param1=100)

    data = {
        "str_val": "{param_later}",
        "int_val": "{param1}",
        "list_val": ["{param0}", "c", "{param_later}"],
        "dict_val": {"a": "{param0}", "b": "{param_later}"},
    }
    config = DemoConfigRequired.schema().load(data)
    bound = config.bind(cc)

    dump = DemoConfigRequired.schema().dump(bound)

    assert len(dump) == 5
    assert dump["something"] is None
    assert dump["str_val"] == "{param_later}"
    assert dump["int_val"] == 100
    assert dump["list_val"] == ["foo", "c", "{param_later}"]
    assert dump["dict_val"] == {"a": "foo", "b": "{param_later}"}


def test_config_template_multi_pass():
    cc = ConfigContext()
    cc.add_values(param0="foo", param1=100)

    data = {
        "str_val": "{param_later}",
        "int_val": "{param1}",
        "list_val": ["{param0}", "c", "{param_later}"],
        "dict_val": {"a": "{param0}", "b": "{param_later}"},
    }
    config = DemoConfigRequired.schema().load(data)
    bound = config.bind(cc)

    assert bound.something is None
    assert type(bound.str_val) is ConfigTemplateSpec
    assert bound.int_val == 100
    assert type(bound.list_val) is ConfigTemplateSpec
    assert type(bound.dict_val) is ConfigTemplateSpec

    cc.add_values(param_later="bar")
    last = bound.bind(cc)

    assert last.something is None
    assert last.str_val == "bar"
    assert last.int_val == 100
    assert last.list_val == ["foo", "c", "bar"]
    assert last.dict_val == {"a": "foo", "b": "bar"}


def test_nested_config():
    config = DemoNested.schema().load(
        {"str_val": "foo_outer", "nested_val": {"str_val": "foo_inner"}}
    )

    assert config.str_val == "foo_outer"
    assert config.nested_val.str_val == "foo_inner"


def test_nested_template():
    config = DemoNested.schema().load(
        {
            "str_val": "{param0}",
            "nested_val": {"str_val": "{param1}", "int_val": "{param2}"},
        }
    )

    cc = ConfigContext()
    cc.add_values(param0="foo", param1="bar", param2=100)
    bound = config.bind(cc)

    assert bound.str_val == "foo"
    assert bound.nested_val.str_val == "bar"
    assert bound.nested_val.int_val == 100


def test_config_dtype_any_template():
    @dataclass
    class TestConfig(Config):
        simple_any: ConfigAny = config_field(None)
        nested_any: dict[str, ConfigAny] = config_field(dict)

    config = TestConfig.schema().load(
        {"simple_any": "{param0}", "nested_any": {"a": "foo", "b": "{param1}"}}
    )
    cc = ConfigContext()
    cc.add_values(param0=100, param1="bar")
    bound = config.bind(cc)

    assert bound.simple_any == "100"
    assert bound.nested_any["a"] == "foo"
    assert bound.nested_any["b"] == "bar"


def test_config_template_explicit_dtype_cast():
    @dataclass
    class TestConfig(Config):
        union_val_one: str | int = config_field(0)
        union_val_two: int | str = config_field(0)
        any_val_one: ConfigAny = config_field(None)
        any_val_two: ConfigAny = config_field(None)

    config = TestConfig.schema().load(
        {
            "union_val_one": "{param0:d}",
            "union_val_two": "{param0:s}",
            "any_val_one": "{param1:d}",
            "any_val_two": "{param1:f}",
        }
    )

    cc = ConfigContext()
    cc.add_values(param0="100", param1="200")
    bound = config.bind(cc)

    assert bound.union_val_one == 100
    assert bound.union_val_two == "100"
    assert bound.any_val_one == 200
    assert isinstance(bound.any_val_two, float) and bound.any_val_two == 200


CustomType = Annotated[str, mf.String(validate=mv.Regexp(r"^[0-9]+$"))]


def test_validate_annotated_field():
    @dataclass
    class TestConfig(Config):
        custom: CustomType = config_field(Config.REQUIRED)
        opt: CustomType | None = config_field(None)

    with pytest.raises(ValidationError):
        _config = TestConfig.schema().load(
            {
                "custom": "abc",
            }
        )

    with pytest.raises(ValidationError):
        _config = TestConfig.schema().load({"custom": "0123", "opt": "abc"})

    _config = TestConfig.schema().load({"custom": "123"})
    _config = TestConfig.schema().load({"custom": "123", "opt": "987"})
