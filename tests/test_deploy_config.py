import json
import subprocess
import sys
from pathlib import Path

import pytest
from marshmallow import ValidationError

from pycheribenchplot.core.config import (
    AssembledDistTargetConfig,
    AutorunCommonConfig,
    AutorunMode,
    BenchplotUserConfig,
    CheribuildDeployConfig,
    ConfigContext,
    ConfigTemplateSpec,
    DeployConfig,
    DeployConfigCommon,
    DeployManifest,
    DeploySystemConfig,
    DistHandler,
    DistStrategyConfig,
    DistTargetConfig,
    FetchTargetConfig,
    PipelineConfig,
    PlatformArch,
    SessionRunConfig,
    SystemAutorunMatchConfig,
)
from pycheribenchplot.core.error import ConfigurationError
from pycheribenchplot.core.session import Session
from .util.session import benchplot_user_config


@pytest.fixture
def full_deploy_dict() -> dict:
    """
    Return a comprehensive valid DeployConfig dictionary covering all fields and variants.
    """
    return {
        "autorun": {
            "mode": "rc.local",
            "bundle_results": True,
            "settle_delay": 5,
            "reboot": True,
            "user": "bench",
        },
        "system": [
            {
                "name": "nocheri",
                "matches": {"kernel": "nocheri"},
                "run_on": "aarch64-multikernel",
                "autorun": {
                    "kernel": "GENERIC-NODEBUG",
                    "bootenv": "default",
                    "arch": "arm64",
                },
            },
            {
                "name": "hybrid-base",
                "matches": {"kernel": "hybrid"},
                "run_on": "morello-rootfs",
                "autorun": {
                    "kernel": "GENERIC-MORELLO-NODEBUG",
                },
            },
        ],
        "fetch": {
            "cheribuild": {
                "url": "https://github.com/CTSRD-CHERI/cheribuild.git",
                "revision": "92a432e1c0a047b2b0ff8f87ce5c4830fb09e40e",
            },
            "local-helper": {
                "url": "file:///opt/repos/local-helper",
                "revision": "v1.2.0",
                "dest": "src/custom-helper",
            },
            "ssh-repo": {
                "url": "git@github.com:CTSRD-CHERI/cheribsd.git",
                "revision": "main",
            },
            "morello-llvm": {
                "url": "cheribuild://morello-llvm-native",
                "revision": "ec799dcf3582adff883b37cef569ff9afbd263a8",
            },
            "freebsd-src.tar.gz": {
                "url": "https://github.com/freebsd/freebsd-src/archive/refs/tags/release/15.1.0-p7.tar.gz",
                "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "archive": True,
                "dest": "downloads/freebsd-src.tar.gz",
            },
            "extra-blob": {
                "url": "http://mirror.local/extra.bin",
                "digest": "sha512:deadbeefCAFEBABE0123456789",
            },
        },
        "cheribuild": {
            "targets": ["morello-llvm-native", "cheribsd-aarch64"],
            "config": {
                "cheribsd-aarch64": {
                    "kernel-config": [
                        "GENERIC-NODEBUG",
                        "GENERIC-MORELLO-NODEBUG",
                    ],
                    "clean": False,
                    "make-jobs": 8,
                    "nested": {
                        "enabled": True,
                        "flags": ["-O2", "-pipe"],
                        "empty": [],
                    },
                },
                "pass-dash-k-to-make": False,
            },
        },
        "command_hooks": {
            "post_fetch": ["echo fetched"],
            "pre_build": ["echo pre-build"],
            "post_build": ["echo post-build"],
        },
        "dist": {
            "aarch64-multikernel": {
                "strategy": {
                    "handler": "cheribuild",
                    "target": "disk-image-aarch64",
                },
                "bundle_session": True,
                "enable_autorun": True,
                "command_hooks": {
                    "pre_dist": [
                        "mkdir -p $DEPLOY_EXTRA_FILES/usr/freebsd-src",
                        "tar -xf $DEPLOY_DOWNLOADS/freebsd-src.tar.gz -C $DEPLOY_EXTRA_FILES/usr/freebsd-src",
                    ],
                    "post_dist": ["ls -lh $DEPLOY_DIST_DIR"],
                },
            },
            "morello-rootfs": {
                "strategy": {
                    "handler": "bsd-rootfs-archive",
                    "target": "cheribsd-morello-purecap",
                    "output_name": "morello-purecap-rootfs.tar.zst",
                },
                "bundle_session": False,
                "enable_autorun": False,
            },
        },
    }


@pytest.fixture
def simple_deploy_dict() -> dict:
    """
    Return a minimal representative DeployConfig dictionary for roundtrip and session tests.
    """
    return {
        "autorun": {
            "mode": "rc.local",
            "bundle_results": True,
        },
        "system": [
            {
                "name": "nocheri",
                "matches": {"kernel": "nocheri"},
                "run_on": "aarch64-multikernel",
                "autorun": {
                    "kernel": "GENERIC-NODEBUG",
                    "arch": "arm64",
                },
            },
            {
                "name": "hybrid-base",
                "matches": {"kernel": "hybrid"},
                "run_on": "aarch64-multikernel",
                "autorun": {
                    "kernel": "GENERIC-MORELLO-NODEBUG",
                },
            },
        ],
        "fetch": {
            "cheribuild": {
                "url": "https://github.com/CTSRD-CHERI/cheribuild.git",
                "revision": "92a432e1c0a047b2b0ff8f87ce5c4830fb09e40e",
                "dest": "src/cheribuild",
            },
        },
        "cheribuild": {
            "targets": ["cheribsd-aarch64"],
            "config": {
                "cheribsd-aarch64": {
                    "kernel-config": ["GENERIC-NODEBUG", "GENERIC-MORELLO-NODEBUG"],
                    "clean": False,
                    "nested": {
                        "enabled": True,
                        "empty": [],
                    },
                },
            },
        },
        "dist": {
            "aarch64-multikernel": {
                "strategy": {
                    "handler": "cheribuild",
                    "target": "disk-image-aarch64",
                },
            },
        },
    }


def test_deploy_config_defaults():
    """Verify default values when loading an empty DeployConfig and DeployManifest."""
    cfg = DeployConfig.schema().load({})
    assert isinstance(cfg, DeployConfigCommon)
    assert not isinstance(cfg, DeployManifest)
    assert cfg.autorun.mode == AutorunMode.MANUAL
    assert cfg.autorun.bundle_results is False
    assert cfg.autorun.settle_delay == 5
    assert cfg.autorun.reboot is False
    assert cfg.autorun.user == "root"
    assert cfg.system == []
    assert cfg.fetch == {}
    assert cfg.cheribuild is None
    assert cfg.command_hooks == {}
    assert cfg.dist == {}

    manifest = DeployManifest.schema().load({})
    assert isinstance(manifest, DeployConfigCommon)
    assert not isinstance(manifest, DeployConfig)
    assert manifest.autorun.mode == AutorunMode.MANUAL
    assert manifest.system == []
    assert manifest.fetch == {}
    assert manifest.cheribuild is None
    assert manifest.command_hooks == {}
    assert manifest.dist == {}


def test_parse_full_deploy_config(full_deploy_dict: dict):
    """Verify parsing of a full DeployConfig and DeployManifest with all supported options."""
    raw = full_deploy_dict
    cfg = DeployConfig.schema().load(raw)

    assert cfg.autorun.mode == AutorunMode.RC_LOCAL
    assert cfg.autorun.bundle_results is True
    assert cfg.autorun.settle_delay == 5
    assert cfg.autorun.reboot is True
    assert cfg.autorun.user == "bench"

    assert len(cfg.system) == 2
    sys0, sys1 = cfg.system
    assert sys0.name == "nocheri"
    assert sys0.matches == {"kernel": "nocheri"}
    assert sys0.run_on == "aarch64-multikernel"
    assert sys0.autorun is not None
    assert sys0.autorun.kernel == "GENERIC-NODEBUG"
    assert sys0.autorun.bootenv == "default"
    assert sys0.autorun.arch == PlatformArch.ARM64

    assert sys1.name == "hybrid-base"
    assert sys1.autorun is not None
    assert sys1.autorun.arch is None
    assert sys1.autorun.bootenv is None

    assert set(cfg.fetch.keys()) == {
        "cheribuild",
        "local-helper",
        "ssh-repo",
        "morello-llvm",
        "freebsd-src.tar.gz",
        "extra-blob",
    }
    assert cfg.fetch["local-helper"].dest == Path("src/custom-helper")
    assert cfg.fetch["freebsd-src.tar.gz"].archive is True
    assert cfg.fetch["freebsd-src.tar.gz"].dest == Path("downloads/freebsd-src.tar.gz")
    assert cfg.fetch["extra-blob"].archive is False

    assert cfg.cheribuild is not None
    assert cfg.cheribuild.targets == ["morello-llvm-native", "cheribsd-aarch64"]
    assert cfg.cheribuild.config == raw["cheribuild"]["config"]

    assert cfg.command_hooks["post_fetch"] == ["echo fetched"]
    assert cfg.dist["aarch64-multikernel"].strategy.handler == DistHandler.CHERIBUILD
    assert cfg.dist["aarch64-multikernel"].strategy.output_name is None
    assert cfg.dist["morello-rootfs"].strategy.handler == DistHandler.BSD_ROOTFS_ARCHIVE
    assert (
        cfg.dist["morello-rootfs"].strategy.output_name
        == "morello-purecap-rootfs.tar.zst"
    )
    assert cfg.dist["morello-rootfs"].bundle_session is False
    assert cfg.dist["morello-rootfs"].enable_autorun is False

    # Plain DeployConfig shape also loads cleanly into DeployManifest with defaults
    manifest_plain = DeployManifest.schema().load(raw)
    assert isinstance(
        manifest_plain.dist["aarch64-multikernel"], AssembledDistTargetConfig
    )
    assert manifest_plain.dist["aarch64-multikernel"].output_path is None
    assert manifest_plain.dist["aarch64-multikernel"].config_hash is None

    # Populated DeployManifest with AssembledDistTargetConfig fields
    manifest_raw = {
        **raw,
        "dist": {
            **raw["dist"],
            "aarch64-multikernel": {
                **raw["dist"]["aarch64-multikernel"],
                "output_path": "dist/aarch64-multikernel.img",
                "config_hash": "b" * 64,
            },
        },
    }
    manifest = DeployManifest.schema().load(manifest_raw)
    assert manifest.dist["aarch64-multikernel"].output_path == Path(
        "dist/aarch64-multikernel.img"
    )
    assert manifest.dist["aarch64-multikernel"].config_hash == "b" * 64


@pytest.mark.parametrize("schema_cls", [DeployConfig, DeployManifest])
def test_deploy_duplicate_system_name_rejected(schema_cls: type):
    """Verify DeployConfig and DeployManifest reject duplicate or missing DeploySystemConfig names."""
    with pytest.raises(ValidationError, match="Duplicate deploy system name 'dup'"):
        schema_cls.schema().load(
            {
                "system": [
                    {"name": "dup", "matches": {"abi": "nocheri"}},
                    {"name": "dup", "matches": {"abi": "purecap"}},
                ]
            }
        )

    with pytest.raises(ValidationError):
        schema_cls.schema().load({"system": [{"matches": {"abi": "nocheri"}}]})


@pytest.mark.parametrize("schema_cls", [DeployConfig, DeployManifest])
def test_deploy_unknown_run_on_rejected(full_deploy_dict: dict, schema_cls: type):
    """Verify DeployConfig and DeployManifest reject system.run_on when target is missing from dist."""
    # Non-empty dist missing the referenced key
    raw = {
        **full_deploy_dict,
        "system": [
            {**full_deploy_dict["system"][0], "run_on": "nonexistent-image"},
            full_deploy_dict["system"][1],
        ],
    }
    with pytest.raises(ValidationError, match="unknown dist target"):
        schema_cls.schema().load(raw)

    # Empty dist when run_on is specified
    with pytest.raises(ValidationError, match="unknown dist target"):
        schema_cls.schema().load(
            {
                "system": [
                    {
                        "name": "sys0",
                        "matches": {},
                        "run_on": "missing-dist",
                    }
                ],
                "dist": {},
            }
        )


@pytest.mark.parametrize("forbidden_key", ["pretend", "include-dependencies"])
def test_cheribuild_cli_only_keys_rejected_in_schema(forbidden_key: str):
    """Verify CheribuildDeployConfig rejects CLI-only keys at root and inside target blocks."""
    with pytest.raises(ValidationError, match="CLI-only cheribuild option"):
        CheribuildDeployConfig.schema().load(
            {"targets": ["cheribsd-aarch64"], "config": {forbidden_key: True}}
        )

    with pytest.raises(ValidationError, match="CLI-only cheribuild option"):
        CheribuildDeployConfig.schema().load(
            {
                "targets": ["cheribsd-aarch64"],
                "config": {"cheribsd-aarch64": {forbidden_key: True}},
            }
        )


@pytest.mark.parametrize(
    "bad_fetch",
    [
        # Empty URLs
        {"url": "", "revision": "main"},
        {"url": "   ", "revision": "main"},
        # cheribuild:// errors
        {"url": "cheribuild://", "revision": "main"},
        {"url": "cheribuild://   ", "revision": "main"},
        {"url": "cheribuild://morello-llvm"},
        {"url": "cheribuild://morello-llvm", "revision": ""},
        {
            "url": "cheribuild://morello-llvm",
            "revision": "main",
            "digest": "sha256:deadbeef",
        },
        {
            "url": "cheribuild://morello-llvm",
            "revision": "main",
            "archive": True,
        },
        # Archive / digest errors
        {"url": "https://example.com/src.tar.gz", "archive": True},
        {
            "url": "https://example.com/src.tar.gz",
            "digest": "sha256:deadbeef",
            "revision": "main",
        },
        {"url": "https://example.com/src.tar.gz", "digest": "md5:deadbeef"},
        {"url": "https://example.com/src.tar.gz", "digest": "sha1:deadbeef"},
        {"url": "https://example.com/src.tar.gz", "digest": "SHA256:deadbeef"},
        {"url": "https://example.com/src.tar.gz", "digest": "deadbeef"},
        {"url": "https://example.com/src.tar.gz", "digest": "sha256:"},
        {"url": "https://example.com/src.tar.gz", "digest": "sha256:not-valid-hex!"},
        # Git URLs missing revision
        {"url": "https://github.com/CTSRD-CHERI/cheribuild.git"},
        {"url": "http://example.com/repo.git"},
        {"url": "ssh://git@example.com/repo.git"},
        {"url": "git://example.com/repo.git"},
        {"url": "git@github.com:CTSRD-CHERI/cheribuild.git"},
        {"url": "file:///tmp/local-repo"},
        {"url": "https://github.com/CTSRD-CHERI/cheribuild.git", "revision": "  "},
        # Malformed scheme
        {"url": "cheribuild:/typo", "revision": "main"},
    ],
)
def test_fetch_target_validation_failures(bad_fetch: dict):
    """Verify FetchTargetConfig rejects invalid URL, revision, archive, and digest combinations."""
    with pytest.raises(ValidationError):
        FetchTargetConfig.schema().load(bad_fetch)


def test_fetch_target_templated_fields():
    """Verify templated URL, revision, and digest fields defer validation until binding."""
    templated = FetchTargetConfig.schema().load(
        {
            "url": "file://{user.src_path}/repo",
            "revision": "main",
        }
    )
    assert isinstance(templated.url, ConfigTemplateSpec)

    templated_archive = FetchTargetConfig.schema().load(
        {
            "url": "https://{host}/archive.tar.gz",
            "digest": "sha256:{digest_var}",
            "archive": True,
        }
    )
    assert isinstance(templated_archive.url, ConfigTemplateSpec)
    assert isinstance(templated_archive.digest, ConfigTemplateSpec)


def test_command_hooks_phase_validation():
    """Verify ContainsOnly validation on global and per-dist command_hooks phases."""
    with pytest.raises(ValidationError):
        DeployConfig.schema().load(
            {"command_hooks": {"pre_dist": ["echo wrong-scope"]}}
        )

    with pytest.raises(ValidationError):
        DistTargetConfig.schema().load(
            {
                "strategy": {
                    "handler": "cheribuild",
                    "target": "disk-image-aarch64",
                },
                "command_hooks": {"post_build": ["echo wrong-scope"]},
            }
        )


def test_enum_and_strategy_validation_failures():
    """Verify invalid enum values and empty strategy targets fail schema validation."""
    with pytest.raises(ValidationError):
        AutorunCommonConfig.schema().load({"mode": "systemd"})

    with pytest.raises(ValidationError):
        DistStrategyConfig.schema().load(
            {"handler": "docker", "target": "disk-image-aarch64"}
        )

    with pytest.raises(ValidationError):
        DistStrategyConfig.schema().load({"handler": "cheribuild", "target": "   "})

    with pytest.raises(ValidationError):
        SystemAutorunMatchConfig.schema().load({"kernel": "GENERIC", "arch": "x86_64"})


def test_pipeline_config_roundtrip_and_bind(tmp_path: Path, simple_deploy_dict: dict):
    """
    Verify PipelineConfig emit_json -> load_json/loads round-trip and that nested
    cheribuild.config dicts/lists/booleans survive Config.bind() intact.
    """
    raw_pipeline = {
        "version": "1.0",
        "benchmark": {
            "name": "kbuild-test",
            "iterations": 1,
            "parameterize": {"kernel": ["nocheri", "hybrid"]},
            "generators": [
                {
                    "handler": "kernel-build.exec",
                    "task_options": {"kernel_config": "GENERIC"},
                }
            ],
        },
        "deploy": simple_deploy_dict,
    }

    pipeline = PipelineConfig.schema().load(raw_pipeline)
    emitted = pipeline.emit_json()

    # Round-trip via schema().loads
    reloaded_loads = PipelineConfig.schema().loads(emitted)
    assert reloaded_loads.deploy is not None
    assert (
        reloaded_loads.deploy.cheribuild.config
        == raw_pipeline["deploy"]["cheribuild"]["config"]
    )
    assert reloaded_loads.deploy.system[0].autorun.arch == PlatformArch.ARM64
    assert reloaded_loads.deploy.system[1].autorun.arch is None
    assert reloaded_loads.deploy.fetch["cheribuild"].dest == Path("src/cheribuild")

    # Round-trip via load_json on disk
    json_path = tmp_path / "pipeline.json"
    json_path.write_text(emitted)
    reloaded_file = PipelineConfig.load_json(json_path)
    assert (
        reloaded_file.deploy.cheribuild.config
        == raw_pipeline["deploy"]["cheribuild"]["config"]
    )

    # Verify Config.bind() preserves nested dicts, lists, bools cleanly
    ctx = ConfigContext()
    ctx.add_values(kernel="nocheri")
    bound = reloaded_file.bind(ctx)
    assert (
        bound.deploy.cheribuild.config == raw_pipeline["deploy"]["cheribuild"]["config"]
    )
    assert bound.deploy.cheribuild.config["cheribsd-aarch64"]["clean"] is False
    assert bound.deploy.cheribuild.config["cheribsd-aarch64"]["nested"]["empty"] == []


def test_session_run_config_with_deploy_roundtrip_and_session_generate(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, simple_deploy_dict: dict
):
    """
    Verify SessionRunConfig.generate_v1 resolves deploy.system, exposes system_target
    templates, round-trips through JSON, and works with Session.generate().
    """
    raw_pipeline = {
        "version": "1.0",
        "benchmark": {
            "name": "kbuild-session",
            "desc": "Build on {system_target.name}",
            "iterations": 1,
            "parameterize": {"kernel": ["nocheri", "hybrid"]},
            "generators": [
                {
                    "handler": "kernel-build.exec",
                    "task_options": {"kernel_config": "{system_target.autorun.kernel}"},
                }
            ],
        },
        "deploy": simple_deploy_dict,
    }

    pipeline = PipelineConfig.schema().load(raw_pipeline)
    run_cfg = SessionRunConfig.generate(benchplot_user_config, pipeline)

    assert isinstance(run_cfg.deploy, DeployManifest)
    assert isinstance(
        run_cfg.deploy.dist["aarch64-multikernel"], AssembledDistTargetConfig
    )
    assert run_cfg.deploy.dist["aarch64-multikernel"].output_path is None
    assert run_cfg.deploy.dist["aarch64-multikernel"].config_hash is None
    assert len(run_cfg.configurations) == 2

    by_kernel = {c.parameters["kernel"]: c for c in run_cfg.configurations}
    assert by_kernel["nocheri"].parameters["target"] == "nocheri"
    assert by_kernel["nocheri"].system_target is not None
    assert by_kernel["nocheri"].system_target.name == "nocheri"
    assert by_kernel["nocheri"].system_target.run_on == "aarch64-multikernel"
    assert by_kernel["nocheri"].desc == "Build on nocheri"

    assert by_kernel["hybrid"].parameters["target"] == "hybrid-base"
    assert by_kernel["hybrid"].system_target is not None
    assert by_kernel["hybrid"].system_target.name == "hybrid-base"
    assert by_kernel["hybrid"].desc == "Build on hybrid-base"

    # Verify SessionRunConfig JSON round-trip
    run_json = run_cfg.emit_json()
    run_file = tmp_path / "session-run.json"
    run_file.write_text(run_json)
    reloaded_run = SessionRunConfig.load_json(run_file)
    assert isinstance(reloaded_run.deploy, DeployManifest)
    assert (
        reloaded_run.deploy.cheribuild.config
        == raw_pipeline["deploy"]["cheribuild"]["config"]
    )
    assert reloaded_run.configurations[0].system_target is not None

    # Verify Session.make_new + Session.generate() succeed with deploy.system
    session_dir = tmp_path / "test-session"
    session = Session.make_new(benchplot_user_config, pipeline, session_dir)
    scheduler = session.make_scheduler()
    session.generate(scheduler)
    assert not scheduler.failed_tasks

    # Verify unmatched parameter combination raises ConfigurationError
    unmatched_raw = simple_deploy_dict
    unmatched_raw["system"] = [unmatched_raw["system"][0]]
    bad_pipeline = PipelineConfig.schema().load(
        {**raw_pipeline, "deploy": unmatched_raw}
    )
    with pytest.raises(ConfigurationError):
        SessionRunConfig.generate(benchplot_user_config, bad_pipeline)


def test_deploy_deferred_user_template_resolution(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Verify {user.*} templates in DeployConfig survive SessionRunConfig.generate()
    as portable templates in session-run.json and resolve on Session load.
    """
    pipeline = PipelineConfig.schema().load(
        {
            "version": "1.0",
            "benchmark": {
                "name": "portable-deploy",
                "parameterize": {"target": ["local"]},
                "generators": [],
            },
            "deploy": {
                "fetch": {
                    "local-repo": {
                        "url": "file://{user.src_path}/my-repo",
                        "revision": "main",
                    }
                }
            },
        }
    )

    session_dir = tmp_path / "portable-session"
    Session.make_new(benchplot_user_config, pipeline, session_dir)

    raw_saved = json.loads((session_dir / "session-run.json").read_text())
    assert (
        raw_saved["deploy"]["fetch"]["local-repo"]["url"]
        == "file://{user.src_path}/my-repo"
    )

    loaded_session = Session.from_path(benchplot_user_config, session_dir)
    assert loaded_session is not None
    assert loaded_session.config.deploy is not None
    assert (
        loaded_session.config.deploy.fetch["local-repo"].url
        == f"file://{benchplot_user_config.src_path}/my-repo"
    )


def test_cli_info_config_pipeline(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Verify `benchplot-cli info config --desc pipeline` and `--generate pipeline` succeed cleanly.
    """
    user_cfg_file = tmp_path / "cheri-benchplot.json"
    user_cfg_file.write_text(benchplot_user_config.emit_json())

    desc_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycheribenchplot.benchplot_cli",
            "-c",
            str(user_cfg_file),
            "info",
            "config",
            "--desc",
            "pipeline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DeployConfig" in desc_res.stdout
    assert "FetchTargetConfig" in desc_res.stdout
    assert "DistStrategyConfig" in desc_res.stdout
    assert "AutorunMode(manual, rc.d, rc.local)" in desc_res.stdout
    assert "DistHandler(cheribuild, bsd-rootfs-archive)" in desc_res.stdout

    gen_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycheribenchplot.benchplot_cli",
            "-c",
            str(user_cfg_file),
            "info",
            "config",
            "--generate",
            "pipeline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(gen_res.stdout)
    assert "deploy" in generated
    assert generated["deploy"] is None
