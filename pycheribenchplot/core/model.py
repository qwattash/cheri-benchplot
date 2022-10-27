import inspect
import typing
import uuid

import numpy as np
import pandas as pd
import typing_inspect
import wrapt
from pandera import Column, Field, SchemaModel
from pandera.dtypes import immutable
from pandera.engines import numpy_engine, pandas_engine
from pandera.errors import SchemaError
from pandera.typing import DataFrame, Index, Series


@pandas_engine.Engine.register_dtype
@immutable
class UUIDType(numpy_engine.Object):
    """
    Pandera data type for UUID values
    """
    def check(self, pandera_dtype, data_container):
        return data_container.map(lambda v: isinstance(v, uuid.UUID))

    def coerce(self, series: pd.Series) -> pd.Series:
        return series.map(lambda uid: uid if isinstance(uid, uuid.UUID) else uuid.UUID(uid))


class DataModel(SchemaModel):
    """
    Empty base data model for all schemas.
    The dataset_id and gid fields are added dynamically from the current session.
    This is necessary until we find a nice way to automate the hooking to add dynamic fields.
    """
    dataset_id: Index[UUIDType] = Field(nullable=False, title="benchmark run ID")
    dataset_gid: Index[UUIDType] = Field(nullable=False, title="platform ID")
    # Extra dynamic param indexes are generated from the configuration after this field

    @classmethod
    def to_schema(cls, session: "PipelineSession" = None):
        """
        Generate a dataframe or series schema.
        This takes care to add the dataset index fields, including the dynamic parameters
        fields from the current configuration.

        :param session: the pipeline session to use to fetch dynamic fields
        :return: dataframe or series schema for validation
        """
        assert session is not None, "Need to specify a session for dynamic fields"
        s = super().to_schema()
        index_names = list(s.index.names)
        param_index = session.benchmark_matrix.index
        extra_columns = {}
        if param_index.names[0] != "RESERVED__unparameterized_index":
            for param in param_index.names:
                try:
                    dt = param_index.dtypes[param]
                except AttributeError:
                    dt = param_index.dtype
                extra_columns[param] = Column(dt,
                                              nullable=False,
                                              coerce=True,
                                              required=True,
                                              title=param,
                                              description=f"Benchmark parameter {param}")
        s = s.add_columns(extra_columns).reset_index()
        # Create new index in the following order [dataset_id, dataset_gid, <param_index_fields>, <other_index_fields>]
        offset = index_names.index("dataset_gid") + 1
        index_names[offset:offset] = extra_columns.keys()
        return s.set_index(index_names)

    class Config:
        # Drop extra columns
        strict = "filter"
        # Coerce data types
        coerce = True


def _resolve_validator(type_target: typing.Type, value: typing.Any) -> typing.Type[DataModel] | None:
    if type_target == inspect.Signature.empty:
        return None
    if not typing_inspect.is_generic_type(type_target):
        return None

    if typing_inspect.is_union_type(type_target):
        if typing_inspect.is_optional_type(type_target):
            type_target = typing.get_args(type_target)[0]
        else:
            # Dynamically resolve union type?
            return None

    type_base = typing.get_origin(type_target)
    if issubclass(DataFrame, type_base):
        return typing.get_args(type_target)[0]
    elif issubclass(Series, type_base):
        return typing.get_args(type_target)[0]
    else:
        return None


def check_data_model(warn=False) -> typing.Callable:
    """
    Class method decorator designed to decorate DatasetTask transformations.

    This will automatically insert model checks for the input and output DataModel dataframes from type hints.
    The caveat is that normally models do not contain the dynamic fields generated for each session from config,
    this makes sure that the schema is generated using the correct session from the current DatasetTask instance.
    """
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        assert instance is not None, "@check_data_model() must be used on a DatasetTask method"
        # assert isinstance(instance, DatasetTask), "@check_data_model() must be used on a DatasetTask method"

        sig = inspect.signature(wrapped)
        bound_args = sig.bind_partial(*args).arguments
        bound_kwargs = sig.bind_partial(**kwargs).arguments
        wrapped_name = wrapped.__qualname__
        if warn:
            log_function = instance.logger.warning
        else:
            log_function = instance.logger.error

        def validate_arg(name, value):
            data_model = _resolve_validator(sig.parameters[name].annotation, value)
            if data_model is None:
                return value
            schema = data_model.to_schema(session=instance.session)
            try:
                return schema.validate(value)
            except SchemaError as ex:
                print(ex)
                log_function("%s: argument '%s' failed validation against %s because %s", wrapped_name, name,
                             data_model, ex)
                if not warn:
                    raise

        checked_args = tuple(validate_arg(arg_name, arg_value) for arg_name, arg_value in bound_args.items())
        checked_kwargs = {arg_name: validate_arg(arg_name, arg_value) for arg_name, arg_value in bound_kwargs.items()}
        result = wrapped(*checked_args, **checked_kwargs)

        data_model = _resolve_validator(sig.return_annotation, result)
        if data_model:
            schema = data_model.to_schema(session=instance.session)
            try:
                return schema.validate(result)
            except SchemaError as ex:
                log_function("%s: returned value failed validation against %s because %s", wrapped_name, data_model, ex)
                if not warn:
                    raise
        return result

    return wrapper
