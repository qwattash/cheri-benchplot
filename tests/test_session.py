import copy
import operator
from uuid import UUID

import pandera as pa
import pytest

EXPECT_NOPARAM_INDEX_NAME = "RESERVED__unparameterized_index"

base_session_conf = {
    "uuid": "17856370-2fd1-4597-937a-42b1277da440",
    "name": "benchplot-selftest-model",
    "configurations": []
}

base_benchmark_conf = {
    "name": "selftest0",
    "iterations": 2,
    "benchmark": {
        "handler": "test-benchmark"  # This is registered in conftest.py
    },
    "instance": {
        "kernel": "selftest-kernel",
        "name": "selftest-instance"
    }
}


def make_bench_config(uuid, g_uuid, baseline=False, params=None):
    conf = copy.deepcopy(base_benchmark_conf)
    if params:
        conf["parameters"] = params
    conf["uuid"] = uuid
    conf["g_uuid"] = g_uuid
    conf["instance"]["baseline"] = baseline
    return conf


def make_session_config(*args):
    conf = copy.deepcopy(base_session_conf)
    conf["configurations"] = list(args)
    return conf


def expect_column(uuid_list, g_uuid, baseline=False):
    get_g_uuid = operator.attrgetter("g_uuid")
    get_uuid = operator.attrgetter("uuid")
    get_baseline = operator.attrgetter("config.instance.baseline")
    uuid_list = map(UUID, uuid_list)
    return pa.Column(object, [
        pa.Check(lambda s: s.map(get_g_uuid) == UUID(g_uuid)),
        pa.Check(lambda s: s.map(get_baseline) == baseline),
        pa.Check(lambda s: set(s.map(get_uuid)) == set(uuid_list))
    ])


#
# Single benchmark configuration/instance with no parameters
#
conf_single_bench = make_session_config(
    make_bench_config("8bc941a3-f6d6-4d37-b193-4738f1da3dae", "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb", baseline=True))
expect_single_bench = pa.DataFrameSchema(
    {
        UUID("2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"):
        expect_column(["8bc941a3-f6d6-4d37-b193-4738f1da3dae"], "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb", baseline=True)
    },
    index=pa.Index(int, name=EXPECT_NOPARAM_INDEX_NAME),
    strict=True,
    unique_column_names=True)

#
# Single benchmark configuration/instance with 1 parameter
#
conf_single_bench_with_params = make_session_config(
    make_bench_config("8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=True,
                      params={"param_foo": "param_foo_value"}))
expect_single_bench_with_params = pa.DataFrameSchema(
    {
        UUID("2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"):
        expect_column(["8bc941a3-f6d6-4d37-b193-4738f1da3dae"], "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb", baseline=True)
    },
    index=pa.MultiIndex([pa.Index(str, pa.Check(lambda s: s == ["param_foo_value"]), name="param_foo")]),
    strict=True,
    unique_column_names=True)

#
# Two benchmark configurations on the same instance with 2 parameters
#
conf_multi_bench_same_instance = make_session_config(
    make_bench_config("8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=True,
                      params={
                          "param_foo": "param_foo_value0",
                          "param_bar": "param_bar_value0"
                      }),
    make_bench_config("1f7e70f5-f15b-4a07-9ab4-d3cb711d7b21",
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=True,
                      params={
                          "param_foo": "param_foo_value1",
                          "param_bar": "param_bar_value1"
                      }),
)
expect_multi_bench_same_instance = pa.DataFrameSchema(
    {
        UUID("2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"):
        expect_column(["8bc941a3-f6d6-4d37-b193-4738f1da3dae", "1f7e70f5-f15b-4a07-9ab4-d3cb711d7b21"],
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=True)
    },
    index=pa.MultiIndex([
        pa.Index(str, pa.Check(lambda s: s.isin(["param_foo_value0", "param_foo_value1"])), name="param_foo"),
        pa.Index(str, pa.Check(lambda s: s.isin(["param_bar_value0", "param_bar_value1"])), name="param_bar"),
    ]),
    strict=True,
    unique_column_names=True)

#
# Two benchmark configurations on two instances with 2 parameters
#
conf_multi_bench_multi_instance = make_session_config(
    make_bench_config("8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=True,
                      params={
                          "param_foo": "param_foo_value0",
                          "param_bar": "param_bar_value0"
                      }),
    make_bench_config("1f7e70f5-f15b-4a07-9ab4-d3cb711d7b21",
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=True,
                      params={
                          "param_foo": "param_foo_value1",
                          "param_bar": "param_bar_value1"
                      }),
    make_bench_config("e412237c-2a89-46c0-8635-cac8f3bf0886",
                      "c73169b7-5797-41c8-9edc-656d666cb45a",
                      baseline=False,
                      params={
                          "param_foo": "param_foo_value0",
                          "param_bar": "param_bar_value0"
                      }),
    make_bench_config("d9a6ee34-eb92-4942-b157-87c9bd6c045c",
                      "c73169b7-5797-41c8-9edc-656d666cb45a",
                      baseline=False,
                      params={
                          "param_foo": "param_foo_value1",
                          "param_bar": "param_bar_value1"
                      }))
expect_multi_bench_multi_instance = pa.DataFrameSchema(
    {
        UUID("2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"):
        expect_column(["8bc941a3-f6d6-4d37-b193-4738f1da3dae", "1f7e70f5-f15b-4a07-9ab4-d3cb711d7b21"],
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=True),
        UUID("c73169b7-5797-41c8-9edc-656d666cb45a"):
        expect_column(["e412237c-2a89-46c0-8635-cac8f3bf0886", "d9a6ee34-eb92-4942-b157-87c9bd6c045c"],
                      "c73169b7-5797-41c8-9edc-656d666cb45a",
                      baseline=False)
    },
    index=pa.MultiIndex([
        pa.Index(str, pa.Check(lambda s: s.isin(["param_foo_value0", "param_foo_value1"])), name="param_foo"),
        pa.Index(str, pa.Check(lambda s: s.isin(["param_bar_value0", "param_bar_value1"])), name="param_bar"),
    ]),
    strict=True,
    unique_column_names=True)


@pytest.mark.parametrize("session_conf,expect", [
    (conf_single_bench, expect_single_bench),
    (conf_single_bench_with_params, expect_single_bench_with_params),
    (conf_multi_bench_same_instance, expect_multi_bench_same_instance),
    (conf_multi_bench_multi_instance, expect_multi_bench_multi_instance),
])
def test_benchmark_matrix(fake_session_factory, session_conf, expect):
    """
    Test benchmark matrix generation from configurations
    """
    assert session_conf is not None
    session = fake_session_factory(session_conf)
    expect.validate(session.benchmark_matrix)


#
# Invalid configuration with missing baseline marker
#
conf_missing_baseline = make_session_config(
    make_bench_config("8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=False,
                      params={"param_foo": "param_foo_value"}))

#
# Invalid configuration with too many baseline markers
#
conf_too_many_baselines = make_session_config(
    make_bench_config("8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                      "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                      baseline=True,
                      params={
                          "param_foo": "param_foo_value0",
                          "param_bar": "param_bar_value0"
                      }),
    make_bench_config("e412237c-2a89-46c0-8635-cac8f3bf0886",
                      "c73169b7-5797-41c8-9edc-656d666cb45a",
                      baseline=True,
                      params={
                          "param_foo": "param_foo_value0",
                          "param_bar": "param_bar_value0"
                      }))


@pytest.mark.parametrize("session_conf,expect", [
    (conf_missing_baseline, pytest.raises(RuntimeError, match=r"Missing baseline")),
    (conf_too_many_baselines, pytest.raises(RuntimeError, match=r"Too many baseline")),
])
def test_benchmark_matrix_invalid(fake_session_factory, session_conf, expect):
    """
    Test benchmark matrix generation from configurations
    """
    assert session_conf is not None
    with expect:
        session = fake_session_factory(session_conf)


@pytest.mark.timeout(5)
def test_session_exec_same_instance(mocker, multi_benchmark_config, fake_session_factory):
    """
    Full test for session.run() interacting with benchmark handlers.
    This should check that we call into the instance manager with the expected set
    of operations for each benchmark
    """
    mock_manager = mocker.patch("pycheribenchplot.core.session.InstanceManager")
    multi_benchmark_config["concurrent_workers"] = 2
    session = fake_session_factory(multi_benchmark_config)

    session.run()

    # Check that the instance_manager.acquire() was called once for each benchmark
    print(mock_manager.mock_calls)


def test_session_exec_two_instances(mocker, multi_benchmark_config, fake_session_factory):
    """
    Full test for session.run() interacting with benchmark handlers.
    This should check that we call into the instance manager with the expected set
    of operations for each benchmark
    """
    return
    mock_manager = mocker.patch("pycheribenchplot.core.session.InstanceManager")
    session = fake_session_factory(conf)

    session.run()
