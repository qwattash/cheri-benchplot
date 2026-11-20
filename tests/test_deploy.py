"""
Unit tests for SessionDeployment orchestration, source fetching, and cheribuild
integration. External tools (git, cheribuild, downloads) are mocked.
"""

import hashlib
import io
import json
from pathlib import Path

import pytest
from marshmallow import ValidationError

from pycheribenchplot.core.config import (
    AssembledDistTargetConfig,
    BenchplotUserConfig,
    DeployConfig,
    DeployManifest,
    PipelineConfig,
    SessionRunConfig,
)
from pycheribenchplot.core.deploy import (
    DeployBuilder,
    DeployDistAssembler,
    DeploymentContext,
    DeploySourceFetcher,
    SessionDeployment,
)
from pycheribenchplot.core.error import ConfigurationError
from pycheribenchplot.core.session import SESSION_RUN_FILE
from .util.deploy import FakeGit, make_deploy_session
from .util.session import benchplot_user_config

SHA1 = "a" * 40
SHA2 = "b" * 40


# ---------------------------------------------------------------------------
# Orchestration: progressive optionality, validation, and phase dispatch
# ---------------------------------------------------------------------------


def test_deploy_noop_without_actions(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Tier 1 (deploy=None) and Tier 2 (deploy.system + autorun only) sessions are
    no-ops: deploy() returns None and no deploy/ directory is created.
    """
    tier1 = make_deploy_session(benchplot_user_config, tmp_path, "s-tier1", None)
    assert SessionDeployment(tier1).deploy() is None
    assert not tier1.get_deploy_root_path().exists()
    assert tier1.config.deploy is None

    tier2 = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-tier2",
        parameterize={"abi": ["nocheri"]},
        deploy={
            "autorun": {"mode": "rc.local"},
            "system": [
                {
                    "name": "sys-nocheri",
                    "matches": {"abi": "nocheri"},
                    "autorun": {"kernel": "GENERIC-NODEBUG"},
                }
            ],
        },
    )
    assert SessionDeployment(tier2).deploy() is None
    assert not tier2.get_deploy_root_path().exists()


def test_deploy_unknown_dist_targets_rejected(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    session = make_deploy_session(benchplot_user_config, tmp_path, "s-bad-dist", None)
    with pytest.raises(ConfigurationError, match="Unknown deployment dist target"):
        SessionDeployment(session).deploy(dist_targets=["missing-img"])


def test_deploy_sub_object_composition(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    session = make_deploy_session(benchplot_user_config, tmp_path, "s-compose", None)
    deployer = SessionDeployment(session)
    assert isinstance(deployer.context, DeploymentContext)
    assert isinstance(deployer.fetcher, DeploySourceFetcher)
    assert isinstance(deployer.builder, DeployBuilder)
    assert isinstance(deployer.assembler, DeployDistAssembler)


def test_deploy_dispatch_calls_phase_objects(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    Verify deploy() routes work to the phase sub-objects and honours
    skip_fetch / skip_build / skip_dist / dist_targets.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-dispatch",
        deploy={
            "dist": {
                "dist-a": {
                    "strategy": {
                        "handler": "cheribuild",
                        "target": "disk-image-aarch64",
                    },
                    "bundle_session": False,
                    "enable_autorun": False,
                },
                "dist-b": {
                    "strategy": {
                        "handler": "cheribuild",
                        "target": "disk-image-aarch64",
                    },
                    "bundle_session": False,
                    "enable_autorun": False,
                },
            }
        },
    )
    fetcher_cls = mocker.patch("pycheribenchplot.core.deploy.DeploySourceFetcher")
    builder_cls = mocker.patch("pycheribenchplot.core.deploy.DeployBuilder")
    assembler_cls = mocker.patch("pycheribenchplot.core.deploy.DeployDistAssembler")

    deployer = SessionDeployment(session)
    manifest = session.config.deploy
    result = deployer.deploy(skip_build=True, dist_targets=["dist-b"])

    assert result is manifest
    fetcher_cls.return_value.fetch_all.assert_called_once_with(manifest)
    builder_cls.return_value.build.assert_not_called()
    assembler_cls.return_value.assemble.assert_called_once_with(manifest, ["dist-b"])

    fetcher_cls.return_value.fetch_all.reset_mock()
    assembler_cls.return_value.assemble.reset_mock()
    deployer.deploy(skip_fetch=True, skip_dist=True)
    fetcher_cls.return_value.fetch_all.assert_not_called()
    builder_cls.return_value.build.assert_called_once_with(manifest)
    assembler_cls.return_value.assemble.assert_not_called()


def test_deploy_converts_plain_deploy_config(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    A plain DeployConfig assigned to session.config.deploy is converted to a
    DeployManifest in place on deploy().
    """
    payload = b"archive-bytes"
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    session = make_deploy_session(benchplot_user_config, tmp_path, "s-plain", None)
    session.config.deploy = DeployConfig.schema().load(
        {
            "fetch": {
                "blob": {
                    "url": "https://example.com/blob.bin",
                    "digest": digest,
                    "archive": True,
                }
            }
        }
    )
    mocker.patch("urllib.request.urlopen", return_value=io.BytesIO(payload))

    manifest = SessionDeployment(session).deploy()
    assert isinstance(manifest, DeployManifest)
    assert session.config.deploy is manifest
    assert manifest.fetch["blob"].digest == digest


# ---------------------------------------------------------------------------
# Fetch phase: git repositories
# ---------------------------------------------------------------------------


def test_fetch_git_clone_by_tag_and_pin(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    Cloning by tag runs git clone + detached checkout and pins the resolved SHA
    in the manifest and session-run.json.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-tag",
        deploy={
            "fetch": {
                "repo": {"url": "https://example.com/repo.git", "revision": "v1.0.0"}
            }
        },
    )
    fake_git = FakeGit(refs={"v1.0.0": SHA1})
    mocker.patch.object(DeploymentContext, "run_git", side_effect=fake_git)

    manifest = SessionDeployment(session).deploy()

    assert manifest.fetch["repo"].revision == SHA1
    assert ("clone", "--no-checkout", "https://example.com/repo.git") in [
        args[:3] for _, args in fake_git.calls
    ]
    assert ("checkout", "--detach", SHA1) in [args for _, args in fake_git.calls]
    repo_dir = session.get_deploy_root_path() / "src" / "repo"
    assert (repo_dir / ".git").is_dir()

    saved = SessionRunConfig.load_json(session.session_root_path / SESSION_RUN_FILE)
    assert saved.deploy.fetch["repo"].revision == SHA1


def test_fetch_git_dest_redirect(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    fetch.<key>.dest redirects the clone destination relative to deploy/.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-dest",
        deploy={
            "fetch": {
                "repo": {
                    "url": "https://example.com/repo.git",
                    "revision": SHA1,
                    "dest": "src/custom-pinned",
                }
            }
        },
    )
    fake_git = FakeGit()
    mocker.patch.object(DeploymentContext, "run_git", side_effect=fake_git)

    manifest = SessionDeployment(session).deploy()

    custom_dir = session.get_deploy_root_path() / "src" / "custom-pinned"
    assert (custom_dir / ".git").is_dir()
    clone_call = next(c for c in fake_git.calls if c[1][0] == "clone")
    assert Path(clone_call[1][-1]) == custom_dir
    assert manifest.fetch["repo"].dest == Path("src/custom-pinned")


def test_fetch_git_cached_head_skips_clone(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    When the manifest already pins a full SHA and the local checkout matches,
    no clone/fetch/checkout commands are issued.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-cached",
        deploy={
            "fetch": {"repo": {"url": "https://example.com/repo.git", "revision": SHA1}}
        },
    )
    repo_dir = session.get_deploy_root_path() / "src" / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    fake_git = FakeGit(head=SHA1)
    mocker.patch.object(DeploymentContext, "run_git", side_effect=fake_git)

    manifest = SessionDeployment(session).deploy()

    assert manifest.fetch["repo"].revision == SHA1
    assert "clone" not in fake_git.ops()
    assert "fetch" not in fake_git.ops()
    assert "checkout" not in fake_git.ops()


def test_fetch_git_force_refetches_moved_tag(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    With force=True, a tag that moved upstream is re-resolved and checked out.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-force",
        deploy={
            "fetch": {
                "repo": {"url": "https://example.com/repo.git", "revision": "v1.0.0"}
            }
        },
    )
    mocker.patch.object(
        DeploymentContext, "run_git", side_effect=FakeGit(refs={"v1.0.0": SHA1})
    )
    manifest = SessionDeployment(session).deploy()
    assert manifest.fetch["repo"].revision == SHA1

    # Tag moved upstream: re-resolve with force=True
    manifest.fetch["repo"].revision = "v1.0.0"
    moved_git = FakeGit(refs={"v1.0.0": SHA2}, head=SHA1)
    mocker.patch.object(DeploymentContext, "run_git", side_effect=moved_git)
    forced = SessionDeployment(session, force=True).deploy()

    assert forced.fetch["repo"].revision == SHA2
    assert "fetch" in moved_git.ops()
    assert ("checkout", "--detach", SHA2) in [args for _, args in moved_git.calls]


def test_fetch_git_unresolvable_revision_raises(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-unresolvable",
        deploy={
            "fetch": {
                "repo": {"url": "https://example.com/repo.git", "revision": "v9.9.9"}
            }
        },
    )
    mocker.patch.object(DeploymentContext, "run_git", side_effect=FakeGit())
    with pytest.raises(RuntimeError, match="Failed to resolve git revision"):
        SessionDeployment(session).deploy()


# ---------------------------------------------------------------------------
# Fetch phase: verified downloads
# ---------------------------------------------------------------------------


def test_fetch_archive_download_and_digest(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    Archives are streamed into deploy/downloads/, verified, and uppercase
    digest hex is normalized to lowercase in the manifest.
    """
    payload = b"deterministic-binary-payload-0123456789"
    digest = f"sha512:{hashlib.sha512(payload).hexdigest().upper()}"
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-archive",
        deploy={
            "fetch": {
                "weights": {"url": "https://example.com/weights.bin", "digest": digest}
            }
        },
    )
    urlopen = mocker.patch("urllib.request.urlopen", return_value=io.BytesIO(payload))

    manifest = SessionDeployment(session).deploy()

    dest = session.get_deploy_root_path() / "downloads" / "weights.bin"
    assert dest.read_bytes() == payload
    assert manifest.fetch["weights"].digest == digest.lower()
    urlopen.assert_called_once_with("https://example.com/weights.bin")


def test_fetch_archive_cached_skips_download(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    payload = b"already-here"
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-archive-cached",
        deploy={
            "fetch": {"blob": {"url": "https://example.com/blob.bin", "digest": digest}}
        },
    )
    dest = session.get_deploy_root_path() / "downloads" / "blob.bin"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    urlopen = mocker.patch("urllib.request.urlopen")

    manifest = SessionDeployment(session).deploy()

    urlopen.assert_not_called()
    assert manifest.fetch["blob"].digest == digest


def test_fetch_archive_digest_mismatch_removes_part(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-bad-digest",
        deploy={
            "fetch": {
                "corrupt": {
                    "url": "https://example.com/corrupt.tar.gz",
                    "digest": "sha256:" + ("0" * 64),
                }
            }
        },
    )
    mocker.patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong-bytes"))

    with pytest.raises(RuntimeError, match="Digest mismatch"):
        SessionDeployment(session).deploy()

    downloads = session.get_deploy_root_path() / "downloads"
    assert not (downloads / "corrupt.tar.gz").exists()
    assert not (downloads / "corrupt.tar.gz.part").exists()


def test_fetch_dest_path_traversal_rejected(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-escape",
        deploy={
            "fetch": {
                "escape": {
                    "url": "https://example.com/x.bin",
                    "digest": "sha256:" + ("0" * 64),
                    "dest": "../escaped.bin",
                }
            }
        },
    )
    with pytest.raises(ConfigurationError, match="escapes directory"):
        SessionDeployment(session).deploy()


# ---------------------------------------------------------------------------
# Command hooks
# ---------------------------------------------------------------------------


def test_hook_env_and_run_hooks(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    session = make_deploy_session(benchplot_user_config, tmp_path, "s-hooks", None)
    context = DeploymentContext.from_session(session)

    env = context.hook_env()
    assert env["SESSION_ROOT"] == str(session.session_root_path)
    assert env["DEPLOY_ROOT"] == str(context.deploy_root)
    assert env["DEPLOY_SRC_ROOT"] == str(context.src_root)
    assert env["DEPLOY_DOWNLOADS"] == str(context.downloads_dir)
    assert env["DEPLOY_BUILD_ROOT"] == str(context.build_root)
    assert env["DEPLOY_SDK_ROOT"] == str(context.sdk_root)
    assert env["DEPLOY_DIST_DIR"] == str(context.dist_dir)
    assert env["CHERIBUILD"].endswith("cheribuild.py")

    dist_env = context.hook_env(
        dist_id="img",
        extra_files_dir=Path("/x/ef"),
        output_path=Path("/x/out.img"),
    )
    assert dist_env["DEPLOY_DIST_ID"] == "img"
    assert dist_env["DEPLOY_EXTRA_FILES"] == "/x/ef"
    assert dist_env["DEPLOY_DIST_OUTPUT"] == "/x/out.img"

    run_mock = mocker.patch("pycheribenchplot.core.deploy.subprocess.run")
    context.run_hooks("post_fetch", ["echo hi"], env)
    run_mock.assert_called_once_with(
        "echo hi", shell=True, cwd=context.deploy_root, env=env, check=True
    )


# ---------------------------------------------------------------------------
# Cheribuild fetch, configuration, and build
# ---------------------------------------------------------------------------


def _attach_ordered_mocks(mocker, fake_git: FakeGit, query_result: str):
    git_mock = mocker.patch.object(DeploymentContext, "run_git", side_effect=fake_git)
    cb_mock = mocker.patch.object(DeploymentContext, "run_cheribuild")
    query_mock = mocker.patch.object(
        DeploymentContext, "query_cheribuild_option", return_value=query_result
    )
    parent = mocker.MagicMock()
    parent.attach_mock(git_mock, "run_git")
    parent.attach_mock(cb_mock, "run_cheribuild")
    parent.attach_mock(query_mock, "query_cheribuild_option")
    return parent, git_mock, cb_mock, query_mock


def test_fetch_cheribuild_before_cheribuild_urls(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    The cheribuild repository is fetched before any cheribuild:// target,
    regardless of fetch dict order; cheribuild:// clones go through a
    fetch-only cheribuild invocation.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-cb-order",
        deploy={
            "fetch": {
                "morello-llvm": {
                    "url": "cheribuild://morello-llvm-native",
                    "revision": "v1.0.0",
                },
                "cheribuild": {
                    "url": "https://example.com/cheribuild.git",
                    "revision": SHA1,
                },
            }
        },
    )
    llvm_src = tmp_path / "cb-src" / "morello-llvm-project"
    fake_git = FakeGit(refs={"v1.0.0": SHA2})
    parent, git_mock, cb_mock, query_mock = _attach_ordered_mocks(
        mocker, fake_git, str(llvm_src)
    )

    manifest = SessionDeployment(session).deploy()

    calls = parent.mock_calls
    clone_idx = next(
        i
        for i, c in enumerate(calls)
        if c[0] == "run_git" and c.args and c.args[0] == "clone"
    )
    first_cb_idx = next(
        i
        for i, c in enumerate(calls)
        if c[0] in ("run_cheribuild", "query_cheribuild_option")
    )
    assert clone_idx < first_cb_idx

    cb_mock.assert_called_once()
    assert cb_mock.call_args[0][1] == [
        "--force",
        "--skip-configure",
        "--skip-build",
        "--skip-install",
        "morello-llvm-native",
    ]
    assert manifest.fetch["cheribuild"].revision == SHA1
    assert manifest.fetch["morello-llvm"].revision == SHA2


def test_cheribuild_url_fetch_existing_checkout(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    A cheribuild:// source with an existing checkout resolves and pins without
    invoking cheribuild fetch-only.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-cb-existing",
        deploy={
            "fetch": {
                "llvm": {
                    "url": "cheribuild://morello-llvm-native",
                    "revision": "v1.0.0",
                }
            }
        },
    )
    llvm_src = tmp_path / "cb-src" / "morello-llvm-project"
    (llvm_src / ".git").mkdir(parents=True)
    fake_git = FakeGit(refs={"v1.0.0": SHA2})
    mocker.patch.object(DeploymentContext, "run_git", side_effect=fake_git)
    cb_mock = mocker.patch.object(DeploymentContext, "run_cheribuild")
    mocker.patch.object(
        DeploymentContext, "query_cheribuild_option", return_value=str(llvm_src)
    )

    manifest = SessionDeployment(session).deploy()

    cb_mock.assert_not_called()
    assert "clone" not in fake_git.ops()
    assert manifest.fetch["llvm"].revision == SHA2


def test_cheribuild_url_fetch_cached(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    A cheribuild:// source already pinned at a matching SHA is skipped entirely.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-cb-cached",
        deploy={
            "fetch": {
                "llvm": {"url": "cheribuild://morello-llvm-native", "revision": SHA2}
            }
        },
    )
    llvm_src = tmp_path / "cb-src" / "morello-llvm-project"
    (llvm_src / ".git").mkdir(parents=True)
    fake_git = FakeGit(head=SHA2)
    mocker.patch.object(DeploymentContext, "run_git", side_effect=fake_git)
    cb_mock = mocker.patch.object(DeploymentContext, "run_cheribuild")
    mocker.patch.object(
        DeploymentContext, "query_cheribuild_option", return_value=str(llvm_src)
    )

    manifest = SessionDeployment(session).deploy()

    cb_mock.assert_not_called()
    assert "clone" not in fake_git.ops()
    assert "checkout" not in fake_git.ops()
    assert manifest.fetch["llvm"].revision == SHA2


def test_cheribuild_json_written(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    cheribuild.json contains the deploy root paths, user configuration, and
    source-directory overrides for cheribuild:// fetch entries with dest.
    """
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-cb-json",
        deploy={
            "fetch": {
                "llvm": {
                    "url": "cheribuild://morello-llvm-native",
                    "revision": SHA1,
                    "dest": "src/llvm-custom",
                }
            },
            "cheribuild": {
                "targets": ["cheribsd-aarch64"],
                "config": {
                    "cheribsd-aarch64": {
                        "kernel-config": [
                            "GENERIC-NODEBUG",
                            "GENERIC-MORELLO-NODEBUG",
                        ]
                    }
                },
            },
        },
    )
    deployer = SessionDeployment(session)
    deployer.context.ensure_dirs()
    cfg_path = deployer.context.ensure_cheribuild_config(session.config.deploy)

    data = json.loads(cfg_path.read_text())
    assert data["source-root"] == str(deployer.context.src_root)
    assert data["build-root"] == str(deployer.context.build_root)
    assert data["output-root"] == str(deployer.context.sdk_root)
    assert data["cheribsd-aarch64"]["kernel-config"] == [
        "GENERIC-NODEBUG",
        "GENERIC-MORELLO-NODEBUG",
    ]
    assert data["morello-llvm-native"]["source-directory"] == str(
        deployer.context.deploy_root / "src" / "llvm-custom"
    )


def test_build_invokes_cheribuild_force_skip_update(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-build",
        deploy={"cheribuild": {"targets": ["morello-llvm-native", "cheribsd-aarch64"]}},
    )
    cb_mock = mocker.patch.object(DeploymentContext, "run_cheribuild")

    deployer = SessionDeployment(session)
    deployer.context.ensure_dirs()
    deployer.builder.build(session.config.deploy)

    cb_mock.assert_called_once()
    assert cb_mock.call_args[0][1] == [
        "--force",
        "--skip-update",
        "morello-llvm-native",
        "cheribsd-aarch64",
    ]


@pytest.mark.parametrize("forbidden_key", ["pretend", "include-dependencies"])
@pytest.mark.parametrize(
    "config_shape",
    [
        lambda key: {key: True},
        lambda key: {"cheribsd-aarch64": {key: True}},
    ],
)
def test_cheribuild_cli_only_config_keys_rejected(forbidden_key: str, config_shape):
    """
    CLI-only cheribuild options inside deploy.cheribuild.config fail schema validation.
    """
    with pytest.raises(ValidationError, match="CLI-only cheribuild option"):
        PipelineConfig.schema().load(
            {
                "version": "1.0",
                "benchmark": {
                    "name": "bad-cb-key",
                    "iterations": 1,
                    "parameterize": {"target": ["local"]},
                    "generators": [],
                },
                "deploy": {
                    "cheribuild": {
                        "targets": ["cheribsd-aarch64"],
                        "config": config_shape(forbidden_key),
                    }
                },
            }
        )


# ---------------------------------------------------------------------------
# Dist phase: cheribuild image strategy and incremental caching
# ---------------------------------------------------------------------------


def _make_image_cb_mock(mocker):
    """
    Mock run_cheribuild that materializes the image file for /path arguments.
    """

    def fake_run_cheribuild(cfg_path, args):
        args = [str(a) for a in args]
        for i, arg in enumerate(args[:-1]):
            if arg.endswith("/path"):
                out = Path(args[i + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("image")

    return mocker.patch.object(
        DeploymentContext, "run_cheribuild", side_effect=fake_run_cheribuild
    )


def _make_image_session(benchplot_user_config, tmp_path):
    return make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-img",
        parameterize={"kconf": ["nocheri"]},
        deploy={
            "autorun": {"mode": "rc.local", "user": "bench", "settle_delay": 3},
            "system": [
                {
                    "name": "sys0",
                    "matches": {"kconf": "nocheri"},
                    "run_on": "img",
                    "autorun": {"kernel": "GENERIC-NODEBUG"},
                }
            ],
            "cheribuild": {"targets": ["cheribsd-aarch64"]},
            "dist": {
                "img": {
                    "strategy": {
                        "handler": "cheribuild",
                        "target": "disk-image-aarch64",
                    },
                    "bundle_session": True,
                }
            },
        },
    )


def test_dist_cheribuild_invocation_and_extra_files(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    The disk-image cheribuild invocation passes extra-files and output path;
    the extra-files overlay contains the staged session bundle and autorun rc files.
    """
    session = _make_image_session(benchplot_user_config, tmp_path)
    cb_mock = _make_image_cb_mock(mocker)

    manifest = SessionDeployment(session).deploy()

    assert cb_mock.call_count == 2
    build_args = cb_mock.call_args_list[0][0][1]
    assert build_args == ["--force", "--skip-update", "cheribsd-aarch64"]
    dist_args = cb_mock.call_args_list[1][0][1]
    extra_files_dir = session.get_deploy_root_path() / "extra-files" / "img"
    assert dist_args == [
        "--force",
        "--skip-update",
        "--disk-image-aarch64/extra-files",
        extra_files_dir,
        "--disk-image-aarch64/path",
        session.get_deploy_root_path() / "dist" / "img.img",
        "disk-image-aarch64",
    ]

    assert (extra_files_dir / "etc" / "rc.conf.d" / "benchplot").is_file()
    assert (extra_files_dir / "etc" / "rc.local").is_file()
    guest = extra_files_dir / "opt" / "benchmark-output"
    assert (guest / "session-run.json").is_file()
    assert (guest / "run" / "auto-run.sh").is_file()
    assert (guest / "run" / "run-target-sys0.sh").is_file()

    record = manifest.dist["img"]
    assert isinstance(record, AssembledDistTargetConfig)
    assert record.output_path == Path("dist/img.img")
    assert record.config_hash is not None


def test_dist_skipped_when_hash_unchanged(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    A second deploy with unchanged inputs skips dist assembly but still
    delegates target builds to cheribuild.
    """
    session = _make_image_session(benchplot_user_config, tmp_path)
    cb_mock = _make_image_cb_mock(mocker)
    manifest = SessionDeployment(session).deploy()
    initial_hash = manifest.dist["img"].config_hash

    cb_mock.reset_mock()
    manifest2 = SessionDeployment(session).deploy()

    assert manifest2.dist["img"].config_hash == initial_hash
    cb_mock.assert_called_once()
    assert cb_mock.call_args[0][1] == ["--force", "--skip-update", "cheribsd-aarch64"]


def test_dist_reassembled_when_config_changes(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    Mutating deploy.cheribuild.config invalidates the dist config hash and
    triggers re-assembly.
    """
    session = _make_image_session(benchplot_user_config, tmp_path)
    cb_mock = _make_image_cb_mock(mocker)
    manifest = SessionDeployment(session).deploy()
    initial_hash = manifest.dist["img"].config_hash

    session.config.deploy.cheribuild.config["cheribsd-aarch64"] = {"make-jobs": 4}
    cb_mock.reset_mock()
    manifest2 = SessionDeployment(session).deploy()

    assert manifest2.dist["img"].config_hash != initial_hash
    assert cb_mock.call_count == 2
    assert cb_mock.call_args_list[1][0][1][-1] == "disk-image-aarch64"
