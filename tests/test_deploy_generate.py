import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from pycheribenchplot.core.config import (
    BenchplotUserConfig,
    PipelineConfig,
    SessionRunConfig,
)
from pycheribenchplot.core.session import Session
from .util.session import benchplot_user_config


def _assert_executable_posix_sh(script_path: Path) -> str:
    """
    Assert ``script_path`` exists, is executable (0755), and passes ``sh -n`` syntax validation.
    """
    assert script_path.is_file(), f"Expected generated script {script_path}"
    mode = stat.S_IMODE(script_path.stat().st_mode)
    assert mode == 0o755, (
        f"Expected 0o755 permissions on {script_path}, got {oct(mode)}"
    )
    res = subprocess.run(
        ["sh", "-n", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, (
        f"POSIX sh syntax check failed for {script_path}:\n{res.stderr}"
    )
    return script_path.read_text()


def test_generate_tier1_no_deploy(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Tier 1 (no deploy block): Session.make_new + Session.generate() produces per-benchmark
    runner-script.sh files and skips run-target-*.sh, auto-run.sh, rc-autorun.sh, and rc.local.
    """
    pipeline = PipelineConfig.schema().load(
        {
            "version": "1.0",
            "benchmark": {
                "name": "local-bench",
                "iterations": 2,
                "parameterize": {"target": ["host-a", "host-b"]},
                "generators": [
                    {
                        "handler": "kernel-build.exec",
                        "task_options": {"kernel_config": "GENERIC"},
                    }
                ],
            },
        }
    )

    session_dir = tmp_path / "tier1-session"
    session = Session.make_new(benchplot_user_config, pipeline, session_dir)
    assert session.config.deploy is None
    for cfg in session.config.configurations:
        assert cfg.system_target is None
        assert cfg.instance is None

    scheduler = session.make_scheduler()
    session.generate(scheduler)
    assert not scheduler.failed_tasks

    data_root = session.get_data_root_path()
    assert list(data_root.glob("run-target-*.sh")) == []
    assert not (data_root / "auto-run.sh").exists()
    assert not (data_root / "rc-autorun.sh").exists()
    assert not (data_root / "rc.local").exists()

    benchmarks = session.all_benchmarks()
    assert len(benchmarks) == 2
    for bench in benchmarks:
        runner_text = _assert_executable_posix_sh(bench.get_run_script_path())
        expected_target = bench.parameters["target"]
        assert f"Target '{expected_target}'" in runner_text
        assert "None" not in runner_text

    # Verify `benchplot-cli info session` works cleanly when instance is None
    user_cfg_file = tmp_path / "cheri-benchplot.json"
    user_cfg_file.write_text(benchplot_user_config.emit_json())
    info_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycheribenchplot.benchplot_cli",
            "-c",
            str(user_cfg_file),
            "info",
            "session",
            str(session_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "on host-a" in info_res.stdout
    assert "on host-b" in info_res.stdout
    assert "on None" not in info_res.stdout


def test_generate_tier2a_deploy_system_without_autorun(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Tier 2a (deploy.system without autorun): produces run-target-<target>.sh for each
    target and skips auto-run.sh, rc-autorun.sh, and rc.local.
    """
    pipeline = PipelineConfig.schema().load(
        {
            "version": "1.0",
            "benchmark": {
                "name": "manual-targets",
                "iterations": 1,
                "parameterize": {"abi": ["nocheri", "purecap"]},
                "generators": [],
            },
            "deploy": {
                "autorun": {
                    "mode": "manual",
                    "bundle_results": False,
                },
                "system": [
                    {
                        "name": "target-nocheri",
                        "matches": {"abi": "nocheri"},
                    },
                    {
                        "name": "target-purecap",
                        "matches": {"abi": "purecap"},
                    },
                ],
            },
        }
    )

    session_dir = tmp_path / "tier2a-session"
    session = Session.make_new(benchplot_user_config, pipeline, session_dir)
    for cfg in session.config.configurations:
        assert cfg.system_target is not None
        assert cfg.system_target.autorun is None
        assert cfg.instance is None

    scheduler = session.make_scheduler()
    session.generate(scheduler)
    assert not scheduler.failed_tasks

    data_root = session.get_data_root_path()
    nocheri_script = _assert_executable_posix_sh(
        data_root / "run-target-target-nocheri.sh"
    )
    purecap_script = _assert_executable_posix_sh(
        data_root / "run-target-target-purecap.sh"
    )
    assert "Run benchmarks for target-nocheri" in nocheri_script
    assert "Run benchmarks for target-purecap" in purecap_script
    assert "%%TARBALL_START%%" not in nocheri_script

    assert not (data_root / "auto-run.sh").exists()
    assert not (data_root / "rc-autorun.sh").exists()
    assert not (data_root / "rc.local").exists()

    for bench in session.all_benchmarks():
        runner_text = _assert_executable_posix_sh(bench.get_run_script_path())
        assert f"Target '{bench.config.system_target.name}'" in runner_text


def test_generate_tier2b_wildcard_arch_and_multikernel(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Tier 2b (deploy.system with autorun, arch=None wildcard + multi-kernel):
    produces run-target-*.sh, auto-run.sh, rc-autorun.sh, rc.local, all valid POSIX sh;
    verifies wildcard targets omit arch guard, empty NEXT_BE="", multi-kernel
    /boot/kernel.${kconf} -> kernel fallback, bundle_results, and settle_delay.
    """
    pipeline = PipelineConfig.schema().load(
        {
            "version": "1.0",
            "benchmark": {
                "name": "multikernel-bench",
                "iterations": 1,
                "parameterize": {"kernel": ["nocheri", "hybrid", "purecap"]},
                "generators": [],
            },
            "deploy": {
                "autorun": {
                    "mode": "rc.local",
                    "bundle_results": True,
                    "settle_delay": 12,
                    "reboot": True,
                    "user": "bench",
                },
                "system": [
                    {
                        "name": "nocheri",
                        "matches": {"kernel": "nocheri"},
                        "autorun": {"kernel": "GENERIC-NODEBUG"},
                    },
                    {
                        "name": "hybrid-base",
                        "matches": {"kernel": "hybrid"},
                        "autorun": {"kernel": "GENERIC-MORELLO-NODEBUG"},
                    },
                    {
                        "name": "purecap-base",
                        "matches": {"kernel": "purecap"},
                        "autorun": {"kernel": "GENERIC-MORELLO-PURECAP-NODEBUG"},
                    },
                ],
            },
        }
    )

    session_dir = tmp_path / "tier2b-session"
    session = Session.make_new(benchplot_user_config, pipeline, session_dir)
    scheduler = session.make_scheduler()
    session.generate(scheduler)
    assert not scheduler.failed_tasks

    data_root = session.get_data_root_path()
    for target_name in ("nocheri", "hybrid-base", "purecap-base"):
        target_text = _assert_executable_posix_sh(
            data_root / f"run-target-{target_name}.sh"
        )
        assert "%%TARBALL_START%%" in target_text

    autorun_text = _assert_executable_posix_sh(data_root / "auto-run.sh")
    rc_autorun_text = _assert_executable_posix_sh(data_root / "rc-autorun.sh")
    rc_local_text = _assert_executable_posix_sh(data_root / "rc.local")

    # Settle delay and autorun_common propagation
    assert "SETTLE_TIME=12" in autorun_text
    assert 'benchplot_user:="bench"' in rc_autorun_text
    assert 'benchplot_settle_delay:="12"' in rc_autorun_text
    assert 'benchplot_user:="bench"' in rc_local_text
    assert 'benchplot_settle_delay:="12"' in rc_local_text
    assert 'benchplot_reboot:="YES"' in rc_local_text

    # Wildcard targets omit [ "${ARCH}" = ... ] guard; flat FOUND_CURRENT chain
    assert '[ "${ARCH}" =' not in autorun_text
    assert 'FOUND_CURRENT=""' in autorun_text
    assert "Unexpected arch/kernel encountered" in autorun_text

    # Empty NEXT_BE="" (never literal "None")
    assert 'NEXT_BE=""' in autorun_text
    assert "None" not in autorun_text

    # Multi-kernel /boot/kernel.${kconf} -> kernel fallback in boot_kernel()
    assert '[ -d "/tmp/be_mount/boot/kernel.${kconf}" ]' in autorun_text
    assert '[ -d "/tmp/be_mount/boot/kernel" ]' in autorun_text
    assert '[ -d "/boot/kernel.${kconf}" ]' in autorun_text
    assert 'kdir="kernel.${kconf}"' in autorun_text
    assert 'kdir="kernel"' in autorun_text

    # Setup/teardown rc.conf.d/benchplot integration
    assert "/etc/rc.conf.d/benchplot" in autorun_text
    assert 'benchplot_user="bench"' in autorun_text
    assert 'benchplot_reboot="YES"' in autorun_text
    assert 'benchplot_settle_delay="12"' in autorun_text

    # All 3 kernels appear in NEXT_KERNEL/KERNEL transitions
    assert 'NEXT_KERNEL="GENERIC-NODEBUG"' in autorun_text
    assert 'NEXT_KERNEL="GENERIC-MORELLO-NODEBUG"' in autorun_text
    assert 'NEXT_KERNEL="GENERIC-MORELLO-PURECAP-NODEBUG"' in autorun_text
    assert 'WORKLOAD_SCRIPT="run-target-nocheri.sh"' in autorun_text
    assert 'WORKLOAD_SCRIPT="run-target-hybrid-base.sh"' in autorun_text
    assert 'WORKLOAD_SCRIPT="run-target-purecap-base.sh"' in autorun_text


def test_generate_tier2c_explicit_arch_and_bootenv(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Tier 2c (deploy.system with explicit arch: "arm64", bootenv, plus optional manual target):
    verify `[ "${ARCH}" = "arm64" ]` guard, Unexpected architecture / arch/kernel fallbacks,
    bootenv activation commands, and omission of manual-only targets from auto-run.sh.
    """
    pipeline = PipelineConfig.schema().load(
        {
            "version": "1.0",
            "benchmark": {
                "name": "arm64-bootenv-bench",
                "iterations": 1,
                "parameterize": {"variant": ["nocheri", "hybrid", "manual-only"]},
                "generators": [],
            },
            "deploy": {
                "autorun": {
                    "mode": "rc.d",
                    "bundle_results": False,
                },
                "system": [
                    {
                        "name": "morello-nocheri",
                        "matches": {"variant": "nocheri"},
                        "autorun": {
                            "kernel": "GENERIC-NODEBUG",
                            "arch": "arm64",
                            "bootenv": "be-nocheri",
                        },
                    },
                    {
                        "name": "morello-hybrid",
                        "matches": {"variant": "hybrid"},
                        "autorun": {
                            "kernel": "GENERIC-MORELLO-NODEBUG",
                            "arch": "arm64",
                            "bootenv": "be-hybrid",
                        },
                    },
                    {
                        "name": "manual-host",
                        "matches": {"variant": "manual-only"},
                    },
                ],
            },
        }
    )

    session_dir = tmp_path / "tier2c-session"
    session = Session.make_new(benchplot_user_config, pipeline, session_dir)
    scheduler = session.make_scheduler()
    session.generate(scheduler)
    assert not scheduler.failed_tasks

    data_root = session.get_data_root_path()
    # Manual target still gets its own run-target script
    _assert_executable_posix_sh(data_root / "run-target-manual-host.sh")
    _assert_executable_posix_sh(data_root / "run-target-morello-nocheri.sh")
    _assert_executable_posix_sh(data_root / "run-target-morello-hybrid.sh")

    autorun_text = _assert_executable_posix_sh(data_root / "auto-run.sh")
    _assert_executable_posix_sh(data_root / "rc-autorun.sh")
    _assert_executable_posix_sh(data_root / "rc.local")

    assert "SETTLE_TIME=5" in autorun_text
    assert '[ "${ARCH}" = "arm64" ]' in autorun_text
    assert "Unexpected architecture ${ARCH}" in autorun_text
    assert "Unexpected arch/kernel encountered" in autorun_text
    assert 'bectl activate -t "${bootenv}"' in autorun_text
    assert 'NEXT_BE="be-nocheri"' in autorun_text
    assert 'NEXT_BE="be-hybrid"' in autorun_text
    assert 'setup_rc_conf "be-nocheri"' in autorun_text
    assert 'setup_rc_conf "be-hybrid"' in autorun_text
    assert "run-target-manual-host.sh" not in autorun_text


def test_generate_mixed_wildcard_and_explicit_arch(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Verify that when deploy.system mixes an arch=None wildcard target with an explicit
    arch target, auto-run.sh guards only the explicit arch entry with `[ "${ARCH}" = ... ]`
    and leaves the wildcard entry unguarded, passing sh -n.
    """
    pipeline = PipelineConfig.schema().load(
        {
            "version": "1.0",
            "benchmark": {
                "name": "mixed-arch",
                "iterations": 1,
                "parameterize": {"sys": ["generic", "arm"]},
                "generators": [],
            },
            "deploy": {
                "system": [
                    {
                        "name": "generic-sys",
                        "matches": {"sys": "generic"},
                        "autorun": {"kernel": "GENERIC"},
                    },
                    {
                        "name": "arm-sys",
                        "matches": {"sys": "arm"},
                        "autorun": {"kernel": "GENERIC-MORELLO", "arch": "arm64"},
                    },
                ]
            },
        }
    )

    session_dir = tmp_path / "mixed-arch-session"
    session = Session.make_new(benchplot_user_config, pipeline, session_dir)
    scheduler = session.make_scheduler()
    session.generate(scheduler)
    assert not scheduler.failed_tasks

    autorun_text = _assert_executable_posix_sh(
        session.get_data_root_path() / "auto-run.sh"
    )
    assert '[ "${ARCH}" = "arm64" ]' in autorun_text
    assert 'WORKLOAD_SCRIPT="run-target-generic-sys.sh"' in autorun_text
    assert 'WORKLOAD_SCRIPT="run-target-arm-sys.sh"' in autorun_text


def test_autorun_and_rc_local_execution_smoke(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Execution smoke test of generated rc.local and auto-run.sh scripts:
    1. Verify rc.local contains `load_rc_config benchplot` and `su -m` user switching,
       passes `sh -n`, and executes cleanly with a stubbed rc.subr.
    2. Execute auto-run.sh across a 2-kernel sequence with stubbed FreeBSD tools (`uname`,
       `nextboot`, `reboot`), verifying workload execution, `.run_completed` creation,
       multi-kernel fallback (`nextboot -k kernel`), and final-target termination without reboot.
    """
    pipeline = PipelineConfig.schema().load(
        {
            "version": "1.0",
            "benchmark": {
                "name": "smoke-bench",
                "iterations": 1,
                "parameterize": {"kernel": ["k0", "k1"]},
                "command_hooks": {
                    "setup": [
                        {
                            "matches": {},
                            "commands": ["echo ran-${ITERATION:-setup} > marker.txt"],
                        }
                    ]
                },
                "generators": [],
            },
            "deploy": {
                "autorun": {
                    "mode": "rc.local",
                    "bundle_results": False,
                    "settle_delay": 0,
                    "reboot": False,
                    "user": "benchuser",
                },
                "system": [
                    {
                        "name": "sys-k0",
                        "matches": {"kernel": "k0"},
                        "autorun": {"kernel": "KERN-ZERO"},
                    },
                    {
                        "name": "sys-k1",
                        "matches": {"kernel": "k1"},
                        "autorun": {"kernel": "KERN-ONE"},
                    },
                ],
            },
        }
    )

    session_dir = tmp_path / "smoke-session"
    session = Session.make_new(benchplot_user_config, pipeline, session_dir)
    scheduler = session.make_scheduler()
    session.generate(scheduler)
    assert not scheduler.failed_tasks

    data_root = session.get_data_root_path()
    rc_local_path = data_root / "rc.local"
    rc_local_text = _assert_executable_posix_sh(rc_local_path)
    assert 'name="benchplot"' in rc_local_text
    assert "load_rc_config benchplot" in rc_local_text
    assert 'su -m "${benchplot_user}"' in rc_local_text

    # Prepare stub FreeBSD utilities on PATH
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    action_log = tmp_path / "actions.log"

    def _write_stub(name: str, body: str) -> None:
        p = fake_bin / name
        p.write_text(f"#!/bin/sh\n{body}\n")
        p.chmod(0o755)

    _write_stub(
        "uname",
        'case "$1" in -m) echo "arm64" ;; -i) echo "${FAKE_KERN}" ;; *) /usr/bin/env uname "$@" ;; esac',
    )
    _write_stub("which", "exit 1")
    _write_stub("nextboot", f'echo "nextboot:$*" >> "{action_log}"')
    _write_stub("reboot", f'echo "reboot" >> "{action_log}"')
    _write_stub("poweroff", f'echo "poweroff" >> "{action_log}"')
    _write_stub(
        "su",
        # Expect: su -m <user> -c <cmd>
        f'echo "su:$2" >> "{action_log}"\nshift 3\nexec sh -c "$1"',
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"

    # Determine actual target/kernel order in generated auto-run.sh
    # (Polars group_by("target") order is not guaranteed)
    autorun_text = (data_root / "auto-run.sh").read_text()
    if autorun_text.index("KERN-ZERO") < autorun_text.index("KERN-ONE"):
        first_target, first_kern = "sys-k0", "KERN-ZERO"
        second_target, second_kern = "sys-k1", "KERN-ONE"
    else:
        first_target, first_kern = "sys-k1", "KERN-ONE"
        second_target, second_kern = "sys-k0", "KERN-ZERO"

    # Boot 1: first_kern -> runs first_target workload, prepares second_kern via nextboot -k kernel, reboots
    env["FAKE_KERN"] = first_kern
    res0 = subprocess.run(
        ["sh", "./auto-run.sh"],
        cwd=data_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"workload run-target-{first_target}.sh" in res0.stdout

    by_target = {b.target_name: b for b in session.all_benchmarks()}
    assert (
        by_target[first_target].get_benchmark_data_path() / session.workload_done_file
    ).is_file()
    assert not (
        by_target[second_target].get_benchmark_data_path() / session.workload_done_file
    ).exists()
    assert action_log.read_text().splitlines() == ["nextboot:-k kernel", "reboot"]

    # Boot 2 via rc.local (with stubbed /etc/rc.subr & /boot/kernel/kernel):
    # second_kern is the final kernel, benchplot_user=benchuser, benchplot_reboot=NO
    action_log.write_text("")
    fake_etc = tmp_path / "etc"
    (fake_etc / "rc.conf.d").mkdir(parents=True)
    (fake_etc / "rc.conf.d" / "benchplot").write_text(
        f'benchplot_enable="YES"\nbenchplot_session="{data_root}"\n'
    )
    fake_rc_subr = fake_etc / "rc.subr"
    fake_rc_subr.write_text(
        f'info() {{ echo "INFO: $*"; }}\n'
        f'err() {{ code="$1"; shift; echo "ERR: $*"; exit "$code"; }}\n'
        f'load_rc_config() {{ [ -f "{fake_etc}/rc.conf.d/$1" ] && . "{fake_etc}/rc.conf.d/$1"; }}\n'
    )
    fake_boot_kernel = tmp_path / "boot" / "kernel" / "kernel"
    fake_boot_kernel.parent.mkdir(parents=True)
    fake_boot_kernel.write_text("")

    sandboxed_rc_local = tmp_path / "rc.local.sandbox"
    sandboxed_rc_local.write_text(
        rc_local_text.replace("/etc/rc.subr", str(fake_rc_subr))
        .replace("/boot/kernel", str(fake_boot_kernel.parent))
        .replace("sleep 2", "sleep 0")
    )

    env["FAKE_KERN"] = second_kern
    subprocess.run(
        ["sh", str(sandboxed_rc_local)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        by_target[second_target].get_benchmark_data_path() / session.workload_done_file
    ).is_file()
    assert action_log.read_text().splitlines() == ["su:benchuser", "poweroff"]

    # Also verify boot_kernel() chooses `kernel.${kconf}` when `/boot/kernel.${kconf}` exists
    action_log.write_text("")
    fake_boot_root = tmp_path / "boot"
    (fake_boot_root / f"kernel.{second_kern}").mkdir(parents=True, exist_ok=True)
    sandboxed_autorun = data_root / "auto-run.sandbox.sh"
    sandboxed_autorun.write_text(
        (data_root / "auto-run.sh")
        .read_text()
        .replace('"/boot/kernel', f'"{fake_boot_root}/kernel')
    )
    sandboxed_autorun.chmod(0o755)
    # Remove first_target done marker so Boot 1 runs again against fake_boot_root
    (
        by_target[first_target].get_benchmark_data_path() / session.workload_done_file
    ).unlink()
    env["FAKE_KERN"] = first_kern
    subprocess.run(
        ["sh", "./auto-run.sandbox.sh"],
        cwd=data_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert action_log.read_text().splitlines() == [
        f"nextboot:-k kernel.{second_kern}",
        "reboot",
    ]
    sandboxed_autorun.unlink()
