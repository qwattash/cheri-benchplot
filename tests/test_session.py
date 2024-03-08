import copy
import inspect
import operator
from uuid import UUID

import pytest

from pycheribenchplot.core.benchmark import Benchmark


def check_that(*predicates):
    def _checker(descriptor: Benchmark):
        for pred in predicates:
            assert pred(descriptor), inspect.getsource(pred)

    return _checker


def check_matrix(m, expect):
    assert len(m.columns) == len(expect), "Configuration combinations mismatch"
    for column in m.columns:
        assert column in expect, f"Missing instance column {column}"
        assert len(expect[column]) == len(m[column]), f"Matrix shape mismatch at {column}"
        for checker, entry in zip(expect[column], m[column]):
            if callable(checker):
                checker(entry)
            else:
                assert checker == entry


#
# Very simple session configuration, no parameters
#
conf_single_bench = {
    "uuid":
    "17856370-2fd1-4597-937a-42b1277da440",
    "name":
    "benchplot-selftest-model",
    "configurations": [{
        "name": "simple-session-config",
        "iterations": 2,
        "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
        "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
        "generators": [{
            "handler": "test-benchmark"  # This is registered for tests only
        }],
        "instance": {
            "kernel": "selftest-kernel",
            "name": "selftest-instance"
        }
    }]
}
expect_single_bench = {
    "instance": ["2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"],
    "descriptor": [
        check_that(lambda b: b.config.name == "simple-session-config", lambda b: b.config.iterations == 2,
                   lambda b: b.config.uuid == "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                   lambda b: b.config.g_uuid == "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb")
    ]
}

#
# Single benchmark configuration/instance with 1 parameter
#
conf_single_bench_with_params = {
    "uuid":
    "17856370-2fd1-4597-937a-42b1277da440",
    "name":
    "benchplot-selftest-model",
    "configurations": [{
        "name": "simple-session-config",
        "iterations": 2,
        "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
        "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
        "parameters": {
            "param_foo": "param_foo_value"
        },
        "generators": [{
            "handler": "test-benchmark"  # This is registered for tests only
        }],
        "instance": {
            "kernel": "selftest-kernel",
            "name": "selftest-instance"
        }
    }]
}
expect_single_bench_with_params = {
    "param_foo": ["param_foo_value"],
    "instance": ["2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"],
    "descriptor": [
        check_that(lambda b: b.config.name == "simple-session-config", lambda b: b.config.iterations == 2,
                   lambda b: b.config.uuid == "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                   lambda b: b.config.g_uuid == "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                   lambda b: b.parameters == dict(param_foo="param_foo_value"))
    ]
}

#
# Two benchmark configurations on the same instance with 2 parameter values
#
conf_multi_bench_same_instance = {
    "uuid":
    "17856370-2fd1-4597-937a-42b1277da440",
    "name":
    "benchplot-selftest-model",
    "configurations": [
        {
            "name": "simple-session-config/0",
            "iterations": 2,
            "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
            "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
            "parameters": {
                "param_foo": "param_foo_value0"
            },
            "generators": [{
                "handler": "test-benchmark"  # This is registered for tests only
            }],
            "instance": {
                "kernel": "selftest-kernel",
                "name": "selftest-instance"
            }
        },
        {
            "name": "simple-session-config/1",
            "iterations": 2,
            "uuid": "1f7e70f5-f15b-4a07-9ab4-d3cb711d7b21",
            "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
            "parameters": {
                "param_foo": "param_foo_value1"
            },
            "generators": [{
                "handler": "test-benchmark"  # This is registered for tests only
            }],
            "instance": {
                "kernel": "selftest-kernel",
                "name": "selftest-instance"
            }
        }
    ]
}
expect_multi_bench_same_instance = {
    "param_foo": ["param_foo_value0", "param_foo_value1"],
    "instance": ["2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb", "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb"],
    "descriptor": [
        check_that(lambda b: b.config.name == "simple-session-config/0", lambda b: b.config.iterations == 2,
                   lambda b: b.config.uuid == "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                   lambda b: b.config.g_uuid == "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                   lambda b: b.parameters == dict(param_foo="param_foo_value0")),
        check_that(lambda b: b.config.name == "simple-session-config/1", lambda b: b.config.iterations == 2,
                   lambda b: b.config.uuid == "1f7e70f5-f15b-4a07-9ab4-d3cb711d7b21",
                   lambda b: b.config.g_uuid == "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                   lambda b: b.parameters == dict(param_foo="param_foo_value1"))
    ]
}

#
# Benchmark configurations on two instances with 1 parameters
#
conf_multi_bench_multi_instance = {
    "uuid":
    "17856370-2fd1-4597-937a-42b1277da440",
    "name":
    "benchplot-selftest-model",
    "configurations": [
        {
            "name": "simple-session-config/0",
            "iterations": 2,
            "uuid": "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
            "g_uuid": "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
            "parameters": {
                "param_foo": "param_foo_value0"
            },
            "generators": [{
                "handler": "test-benchmark"  # This is registered for tests only
            }],
            "instance": {
                "kernel": "selftest-kernel",
                "name": "selftest-instance"
            }
        },
        {
            "name": "simple-session-config/1",
            "iterations": 2,
            "uuid": "1f7e70f5-f15b-4a07-9ab4-d3cb711d7b21",
            "g_uuid": "c73169b7-5797-41c8-9edc-656d666cb45a",
            "parameters": {
                "param_foo": "param_foo_value0"
            },
            "generators": [{
                "handler": "test-benchmark"  # This is registered for tests only
            }],
            "instance": {
                "kernel": "selftest-kernel",
                "name": "selftest-instance"
            }
        }
    ]
}
expect_multi_bench_multi_instance = {
    "param_foo": ["param_foo_value0", "param_foo_value0"],
    "instance": ["2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb", "c73169b7-5797-41c8-9edc-656d666cb45a"],
    "descriptor": [
        check_that(lambda b: b.config.name == "simple-session-config/0", lambda b: b.config.iterations == 2,
                   lambda b: b.config.uuid == "8bc941a3-f6d6-4d37-b193-4738f1da3dae",
                   lambda b: b.config.g_uuid == "2d2fe5b2-7f8f-4a52-8f68-d673e60acbfb",
                   lambda b: b.parameters == dict(param_foo="param_foo_value0")),
        check_that(lambda b: b.config.name == "simple-session-config/1", lambda b: b.config.iterations == 2,
                   lambda b: b.config.uuid == "1f7e70f5-f15b-4a07-9ab4-d3cb711d7b21",
                   lambda b: b.config.g_uuid == "c73169b7-5797-41c8-9edc-656d666cb45a",
                   lambda b: b.parameters == dict(param_foo="param_foo_value0")),
    ]
}


@pytest.mark.parametrize("session_conf,expect", [
    (conf_single_bench, expect_single_bench),
    (conf_single_bench_with_params, expect_single_bench_with_params),
    (conf_multi_bench_same_instance, expect_multi_bench_same_instance),
    (conf_multi_bench_multi_instance, expect_multi_bench_multi_instance),
])
def test_param_matrix(fake_session_factory, session_conf, expect):
    """
    Test parameterization matrix generation from configurations
    """
    assert session_conf is not None
    session = fake_session_factory(session_conf)
    check_matrix(session.parameterization_matrix, expect)


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
    # TODO


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
    # TODO
