import io
import json
from dataclasses import dataclass

import pytest
from marshmallow import ValidationError

from pycheribenchplot.core.config import *


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
    assert default.qemu_trace is None
    assert default.qemu_trace_categories is None
    assert default.vcu118_bios is None
    assert default.vcu118_ip is None


def test_instance_config():
    default = InstanceConfig.schema().load({"kernel": "GENERIC"})

    assert default.kernel == "GENERIC"
    assert default.baseline == False
    assert default.name == "qemu-riscv64-purecap-hybrid-GENERIC"
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


@pytest.mark.parametrize("dataset", [d.value for d in DatasetName])
def test_dataset_config(dataset):
    check = DatasetConfig.schema().load({"handler": dataset, "run_options": {"foo": "bar"}})
    assert check.handler == DatasetName(dataset)
    assert isinstance(check.run_options, dict)
    assert check.run_options["foo"] == "bar"
