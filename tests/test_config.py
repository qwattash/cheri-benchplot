import io
import json
import uuid
from dataclasses import dataclass

import pytest
from marshmallow import ValidationError

from pycheribenchplot.core.config import *
from pycheribenchplot.core.task import Task, TaskRegistry


@pytest.fixture
def fake_user_config():
    return BenchplotUserConfig()


def test_common_platform_options():
    default = CommonPlatformOptions.schema().load({})

    assert default.cores == 1
    assert default.qemu_trace_file is None
    assert default.qemu_trace == "no"
    assert default.qemu_trace_categories == set()
    assert default.vcu118_bios is None
    assert default.vcu118_ip == "10.88.88.2"

    test = {
        "cores": 2,
        "qemu_trace_file": "/path/to/qemu.pb",
        "qemu_trace": "perfetto",
        "qemu_trace_categories": ["ctrl", "instructions"],
        "vcu118_bios": "/path/to/bios",
        "vcu118_ip": "10.0.0.1"
    }
    check = CommonPlatformOptions.schema().load(test)

    assert check.cores == 2
    assert check.qemu_trace_file == Path("/path/to/qemu.pb")
    assert check.qemu_trace == "perfetto"
    assert check.qemu_trace_categories == {"ctrl", "instructions"}
    assert check.vcu118_bios == Path("/path/to/bios")
    assert check.vcu118_ip == "10.0.0.1"


def test_platform_options():
    default = PlatformOptions.schema().load({})

    assert default.cores is None
    assert default.qemu_trace_file is None
    assert default.qemu_trace == "no"
    assert default.qemu_trace_categories is None
    assert default.vcu118_bios is None
    assert default.vcu118_ip is None


def test_instance_config():
    default = InstanceConfig.schema().load({"kernel": "GENERIC"})

    assert default.kernel == "GENERIC"
    assert default.baseline == False
    assert default.name == "qemu UserABI:riscv64-purecap KernABI:hybrid KernConf:GENERIC"
    assert default.platform == InstancePlatform.QEMU
    assert default.cheri_target == InstanceCheriBSD.RISCV64_PURECAP
    assert default.kernelabi == InstanceKernelABI.HYBRID
    assert default.cheribuild_kernel == True
    assert isinstance(default.platform_options, PlatformOptions)

    check = InstanceConfig.schema().load({"kernel": "GENERIC", "kernelabi": "hybrid", "cheri_target": "riscv64-hybrid"})
    assert check.kernelabi == InstanceKernelABI.HYBRID
    assert check.cheri_target == InstanceCheriBSD.RISCV64_HYBRID
    assert check.user_pointer_size == 8
    assert check.kernel_pointer_size == 8

    check = InstanceConfig.schema().load({
        "kernel": "GENERIC",
        "kernelabi": "purecap",
        "cheri_target": "riscv64-purecap"
    })
    assert check.kernelabi == InstanceKernelABI.PURECAP
    assert check.cheri_target == InstanceCheriBSD.RISCV64_PURECAP
    assert check.user_pointer_size == 16
    assert check.kernel_pointer_size == 16

    with pytest.raises(ValidationError):
        InstanceConfig.schema().load({})


@pytest.mark.parametrize("task_name", [t.task_name for t in TaskRegistry.public_tasks["exec"].values()])
def test_task_config(task_name):
    check = TaskTargetConfig.schema().load({"handler": task_name, "task_options": {"foo": "bar"}})
    assert check.handler == task_name
    assert check.namespace is None
    assert check.name == task_name
    assert isinstance(check.task_options, dict)
    assert check.task_options["foo"] == "bar"


def test_pipeline_config_missing_benchmark():
    data = {"instance_config": {"instances": [{"kernel": "GENERIC-FAKE-TEST"}]}, "benchmark_config": []}
    with pytest.raises(ValueError):
        input_config = PipelineConfig.schema().load(data)


def test_pipeline_config_missing_instance():
    """
    It is possible to run without any instance when the benchmark configuration uses
    generators that run locally. This is the case for static analysis and source code
    scraping.
    """
    data = {
        "instance_config": {
            "instances": []
        },
        "benchmark_config": [{
            "name": "test-valid",
            "iterations": 1,
            "parameterize": {},
            "benchmark": {
                "handler": "test-benchmark"
            }
        }]
    }

    input_config = PipelineConfig.schema().load(data)


def test_run_config_gen_without_parametrization(fake_user_config):
    # Setup a valid pipeline with just one benchmark
    data = {
        "instance_config": {
            "instances": [{
                "kernel": "GENERIC-FAKE-TEST"
            }]
        },
        "benchmark_config": [{
            "name": "test-valid",
            "iterations": 1,
            "parameterize": {},
            "benchmark": {
                "handler": "test-benchmark"
            }
        }]
    }
    input_config = PipelineConfig.schema().load(data)
    check = SessionRunConfig.generate(fake_user_config, input_config)

    assert len(check.configurations) == 1
    conf0 = check.configurations[0]
    assert conf0.name == "test-valid"
    assert conf0.iterations == 1
    assert conf0.generators[0].handler == "test-benchmark"
    assert conf0.g_uuid is not None
    assert conf0.parameters == {}
    assert conf0.instance.kernel == "GENERIC-FAKE-TEST"


def test_run_config_gen_without_parametrization_fail(fake_user_config):
    # Setup a pipeline with two benchmark configs that should fail
    data = {
        "instance_config": {
            "instances": [{
                "kernel": "GENERIC-FAKE-TEST"
            }]
        },
        "benchmark_config": [{
            "name": "test-valid",
            "iterations": 1,
            "parameterize": {},
            "benchmark": {
                "handler": "test-benchmark"
            }
        }, {
            "name": "test-valid2",
            "iterations": 1,
            "parameterize": {},
            "benchmark": {
                "handler": "test-benchmark"
            }
        }]
    }
    input_config = PipelineConfig.schema().load(data)
    with pytest.raises(ValueError):
        check = SessionRunConfig.generate(fake_user_config, input_config)


def test_run_config_gen_single_parametrization(fake_user_config):
    data = {
        "instance_config": {
            "instances": [{
                "kernel": "GENERIC-FAKE-TEST"
            }]
        },
        "benchmark_config": [{
            "name": "test-valid-{fakeparam}",
            "iterations": 1,
            "parameterize": {
                "fakeparam": ["value0", "value1"]
            },
            "benchmark": {
                "handler": "test-benchmark",
                "task_options": {
                    "fake_arg": "{fakeparam}"
                }
            }
        }]
    }
    input_config = PipelineConfig.schema().load(data)
    check = SessionRunConfig.generate(fake_user_config, input_config)

    assert len(check.configurations) == 2
    conf0 = check.configurations[0]
    assert conf0.name == "test-valid-{fakeparam}"
    assert conf0.iterations == 1
    assert conf0.generators[0].handler == "test-benchmark"
    assert conf0.generators[0].task_options["fake_arg"] == "{fakeparam}"
    assert conf0.g_uuid is not None
    assert conf0.parameters == {"fakeparam": "value0"}
    assert conf0.instance.kernel == "GENERIC-FAKE-TEST"

    conf1 = check.configurations[1]
    assert conf1.name == "test-valid-{fakeparam}"
    assert conf1.iterations == 1
    assert conf1.generators[0].handler == "test-benchmark"
    assert conf1.generators[0].task_options["fake_arg"] == "{fakeparam}"
    assert conf1.g_uuid is not None
    assert conf1.parameters == {"fakeparam": "value1"}
    assert conf1.instance.kernel == "GENERIC-FAKE-TEST"
    # Check same instance config uuid
    assert conf1.g_uuid == conf0.g_uuid


def test_run_config_gen_multi_parametrization(fake_user_config):
    data = {
        "instance_config": {
            "instances": [{
                "kernel": "GENERIC-FAKE-TEST"
            }]
        },
        "benchmark_config": [{
            "name": "test-first-{fakeparam}",
            "iterations": 1,
            "parameterize": {
                "fakeparam": ["value0"]
            },
            "benchmark": {
                "handler": "test-benchmark",
                "task_options": {
                    "fake_arg": "{fakeparam}"
                }
            }
        }, {
            "name": "test-second-{fakeparam}",
            "iterations": 1,
            "parameterize": {
                "fakeparam": ["value1"]
            },
            "benchmark": {
                "handler": "test-benchmark",
                "task_options": {
                    "fake_arg": "{fakeparam}"
                }
            }
        }]
    }
    input_config = PipelineConfig.schema().load(data)
    check = SessionRunConfig.generate(fake_user_config, input_config)

    assert len(check.configurations) == 2
    conf0 = check.configurations[0]
    assert conf0.name == "test-first-{fakeparam}"
    assert conf0.iterations == 1
    assert conf0.generators[0].handler == "test-benchmark"
    assert conf0.generators[0].task_options["fake_arg"] == "{fakeparam}"
    assert conf0.g_uuid is not None
    assert conf0.parameters == {"fakeparam": "value0"}
    assert conf0.instance.kernel == "GENERIC-FAKE-TEST"

    conf1 = check.configurations[1]
    assert conf1.name == "test-second-{fakeparam}"
    assert conf1.iterations == 1
    assert conf1.generators[0].handler == "test-benchmark"
    assert conf1.generators[0].task_options["fake_arg"] == "{fakeparam}"
    assert conf1.g_uuid is not None
    assert conf1.parameters == {"fakeparam": "value1"}
    assert conf1.instance.kernel == "GENERIC-FAKE-TEST"
    # Check same instance config uuid
    assert conf1.g_uuid == conf0.g_uuid


def test_run_config_gen_multi_parametrization_mismatch(fake_user_config):
    data = {
        "instance_config": {
            "instances": [{
                "kernel": "GENERIC-FAKE-TEST"
            }]
        },
        "benchmark_config": [{
            "name": "test-first-{fakeparam}",
            "iterations": 1,
            "parameterize": {
                "fakeparam": ["value0"]
            },
            "benchmark": {
                "handler": "test-benchmark",
                "task_options": {
                    "fake_arg": "{fakeparam}"
                }
            }
        }, {
            "name": "test-second-{otherfakeparam}",
            "iterations": 1,
            "parameterize": {
                "otherfakeparam": ["value1"]
            },
            "benchmark": {
                "handler": "test-benchmark",
                "task_options": {
                    "fake_arg": "{otherfakeparam}"
                }
            }
        }]
    }
    input_config = PipelineConfig.schema().load(data)
    with pytest.raises(ValueError, match="Invalid configuration"):
        SessionRunConfig.generate(fake_user_config, input_config)


def test_run_config_gen_multi_parametrization_missing(fake_user_config):
    data = {
        "instance_config": {
            "instances": [{
                "kernel": "GENERIC-FAKE-TEST"
            }]
        },
        "benchmark_config": [{
            "name": "test-first-{fakeparam}",
            "iterations": 1,
            "parameterize": {
                "fakeparam": ["value0"]
            },
            "benchmark": {
                "handler": "test-benchmark",
                "task_options": {
                    "fake_arg": "{fakeparam}"
                }
            }
        }, {
            "name": "test-second",
            "iterations": 1,
            "parameterize": {},
            "benchmark": {
                "handler": "test-benchmark"
            }
        }]
    }
    input_config = PipelineConfig.schema().load(data)
    with pytest.raises(ValueError, match="Invalid configuration"):
        SessionRunConfig.generate(fake_user_config, input_config)


def test_unified_benchmark_generators(fake_user_config, mock_task_registry):
    data = {
        "instance_config": {},
        "benchmark_config": [{
            "name": "test-unified",
            "iterations": 1,
            "generators": [{
                "handler": "test.generator-1"
            }, {
                "handler": "test.generator-2"
            }]
        }]
    }

    class Gen1(Task):
        public = True
        task_namespace = "test"
        task_name = "generator-1"

    class Gen2(Task):
        public = True
        task_namespace = "test"
        task_name = "generator-2"

    input_config = PipelineConfig.schema().load(data)
    check = SessionRunConfig.generate(fake_user_config, input_config)

    assert len(check.configurations) == 1
    conf0 = check.configurations[0]
    assert len(conf0.generators) == 2
    assert conf0.generators[0].handler == "test.generator-1"
    assert conf0.generators[1].handler == "test.generator-2"
    assert conf0.instance.platform == InstancePlatform.LOCAL
