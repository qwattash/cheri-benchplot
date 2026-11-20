"""
Tests for dist staging and the bsd-rootfs-archive strategy, using real bsdtar
against mock rootfs trees.
"""

import json
from pathlib import Path

import pytest

from pycheribenchplot.core.config import (
    AssembledDistTargetConfig,
    BenchplotUserConfig,
    DistHandler,
)
from pycheribenchplot.core.deploy import (
    DeploymentContext,
    SessionDeployment,
    _write_extra_files_mtree,
)
from pycheribenchplot.core.error import ConfigurationError
from .util.deploy import (
    assert_mode,
    assert_tar_all_root_wheel,
    create_mock_bsd_rootfs,
    extract_tar,
    make_deploy_session,
    make_tar_gz,
)
from .util.session import benchplot_user_config


def test_bsd_rootfs_rc_local(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    bsd-rootfs-archive produces a valid .tar.zst that preserves METALOG-owned
    files, and overlays /etc/rc.conf.d/benchplot, /etc/rc.local, and the
    bundled session scripts with root:wheel ownership.
    """
    rootfs_dir = tmp_path / "mock-freebsd-rootfs"
    create_mock_bsd_rootfs(rootfs_dir)

    asset_src = tmp_path / "input-assets"
    asset_src.mkdir()
    (asset_src / "config.ini").write_text("threads=4\n")

    # Note: session scripts are intentionally not pre-generated so that
    # auto-generation before bundling is exercised.
    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-rootfs",
        parameterize={"kernel": ["nocheri"]},
        generators=[
            {
                "handler": "kernel-build.exec",
                "task_options": {"kernel_config": "GENERIC"},
            }
        ],
        assets={"bench-cfg": {"action": "copy", "src": str(asset_src)}},
        deploy={
            "autorun": {"mode": "rc.local", "user": "bench", "settle_delay": 7},
            "system": [
                {
                    "name": "freebsd-nocheri",
                    "matches": {"kernel": "nocheri"},
                    "run_on": "morello-archive",
                    "autorun": {"kernel": "GENERIC-NODEBUG"},
                }
            ],
            "dist": {
                "morello-archive": {
                    "strategy": {
                        "handler": "bsd-rootfs-archive",
                        "target": str(rootfs_dir),
                    },
                    "bundle_session": True,
                }
            },
        },
    )

    manifest = SessionDeployment(session).deploy()

    archive_out = session.get_deploy_root_path() / "dist" / "morello-archive.tar.zst"
    assert archive_out.is_file()
    record = manifest.dist["morello-archive"]
    assert isinstance(record, AssembledDistTargetConfig)
    assert record.strategy.handler == DistHandler.BSD_ROOTFS_ARCHIVE
    assert record.output_path == Path("dist/morello-archive.tar.zst")
    assert record.config_hash is not None

    assert_tar_all_root_wheel(archive_out)
    extracted = extract_tar(archive_out, tmp_path / "extracted")

    # METALOG.world /etc/rc.conf and kernel are preserved
    assert (extracted / "etc" / "rc.conf").read_text() == (
        'ifconfig_DEFAULT="DHCP"\nsshd_enable="YES"\n'
    )
    assert (extracted / "boot" / "kernel" / "kernel").is_file()

    # Autorun rc configuration overlay
    rc_conf_d = extracted / "etc" / "rc.conf.d" / "benchplot"
    assert_mode(rc_conf_d, 0o644)
    rc_conf_text = rc_conf_d.read_text()
    assert 'benchplot_enable="YES"' in rc_conf_text
    assert 'benchplot_session="/opt/benchmark-output/run"' in rc_conf_text
    assert 'benchplot_user="bench"' in rc_conf_text
    assert 'benchplot_reboot="NO"' in rc_conf_text
    assert 'benchplot_settle_delay="7"' in rc_conf_text

    rc_local = extracted / "etc" / "rc.local"
    assert_mode(rc_local, 0o755)
    assert "load_rc_config benchplot" in rc_local.read_text()

    # Bundled session files with executable permissions
    guest = extracted / "opt" / "benchmark-output"
    assert (guest / "session-run.json").is_file()
    assert (guest / "assets" / "config.ini").read_text() == "threads=4\n"
    assert_mode(guest / "run" / "auto-run.sh", 0o755)
    assert_mode(guest / "run" / "run-target-freebsd-nocheri.sh", 0o755)


def test_bsd_rootfs_rc_d(tmp_path: Path, benchplot_user_config: BenchplotUserConfig):
    """
    With autorun mode rc.d, the rc script is staged as
    /usr/local/etc/rc.d/benchplot (0755) and no /etc/rc.local is produced.
    """
    rootfs_dir = tmp_path / "mock-freebsd-rootfs"
    create_mock_bsd_rootfs(rootfs_dir)

    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-rcd",
        parameterize={"kernel": ["nocheri"]},
        deploy={
            "autorun": {"mode": "rc.d", "reboot": True},
            "system": [
                {
                    "name": "freebsd-nocheri",
                    "matches": {"kernel": "nocheri"},
                    "run_on": "rcd-archive",
                    "autorun": {"kernel": "GENERIC-NODEBUG"},
                }
            ],
            "dist": {
                "rcd-archive": {
                    "strategy": {
                        "handler": "bsd-rootfs-archive",
                        "target": str(rootfs_dir),
                        "output_name": "custom-rcd.tar.zst",
                    },
                }
            },
        },
    )

    SessionDeployment(session).deploy()

    rcd_tar = session.get_deploy_root_path() / "dist" / "custom-rcd.tar.zst"
    assert rcd_tar.is_file()
    extracted = extract_tar(rcd_tar, tmp_path / "extracted-rcd")

    rcd_service = extracted / "usr" / "local" / "etc" / "rc.d" / "benchplot"
    assert_mode(rcd_service, 0o755)
    assert not (extracted / "etc" / "rc.local").exists()
    assert (
        'benchplot_reboot="YES"'
        in (extracted / "etc" / "rc.conf.d" / "benchplot").read_text()
    )


def test_bsd_rootfs_pre_post_dist_hooks(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    pre_dist hooks can unpack fetched archives into the extra-files overlay
    (symlinks survive archiving); post_dist hooks receive dist env vars.
    """
    rootfs_dir = tmp_path / "mock-freebsd-rootfs"
    create_mock_bsd_rootfs(rootfs_dir)

    src_tar = tmp_path / "freebsd-src.tar.gz"
    src_digest = make_tar_gz(
        src_tar,
        {
            "Makefile": "# FreeBSD top-level Makefile\n",
            "sys/conf/NOTES": "# Kernel notes\n",
        },
    )
    post_dist_marker = tmp_path / "post-dist.txt"

    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-hooks",
        parameterize={"kernel": ["nocheri"]},
        deploy={
            "autorun": {"mode": "manual"},
            "system": [
                {
                    "name": "freebsd-nocheri",
                    "matches": {"kernel": "nocheri"},
                    "run_on": "morello-archive",
                }
            ],
            "fetch": {
                "freebsd-src.tar.gz": {
                    "url": f"file://{src_tar}",
                    "digest": src_digest,
                }
            },
            "dist": {
                "morello-archive": {
                    "strategy": {
                        "handler": "bsd-rootfs-archive",
                        "target": str(rootfs_dir),
                    },
                    "bundle_session": False,
                    "enable_autorun": False,
                    "command_hooks": {
                        "pre_dist": [
                            "mkdir -p $DEPLOY_EXTRA_FILES/usr/src",
                            "tar -xf $DEPLOY_DOWNLOADS/freebsd-src.tar.gz -C $DEPLOY_EXTRA_FILES/usr/src",
                            "ln -s Makefile $DEPLOY_EXTRA_FILES/usr/src/BSDmakefile",
                        ],
                        "post_dist": [
                            f'printf "%s|%s" "$DEPLOY_DIST_ID" "$DEPLOY_DIST_OUTPUT" > {post_dist_marker}'
                        ],
                    },
                }
            },
        },
    )

    SessionDeployment(session).deploy()

    archive_out = session.get_deploy_root_path() / "dist" / "morello-archive.tar.zst"
    assert post_dist_marker.read_text() == f"morello-archive|{archive_out}"

    extracted = extract_tar(archive_out, tmp_path / "extracted")
    assert (extracted / "usr" / "src" / "Makefile").is_file()
    assert (extracted / "usr" / "src" / "sys" / "conf" / "NOTES").is_file()
    assert (extracted / "usr" / "src" / "BSDmakefile").is_symlink()
    assert (extracted / "usr" / "src" / "BSDmakefile").readlink() == Path("Makefile")


def test_bsd_rootfs_missing_metalog_kernel_rejected(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    A rootfs with only METALOG.world (without METALOG.kernel) is rejected.
    """
    broken_rootfs = tmp_path / "broken-rootfs"
    broken_rootfs.mkdir()
    (broken_rootfs / "METALOG.world").write_text("#mtree\n")

    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-broken-metalog",
        deploy={
            "dist": {
                "bad-rootfs": {
                    "strategy": {
                        "handler": "bsd-rootfs-archive",
                        "target": str(broken_rootfs),
                    },
                    "bundle_session": False,
                    "enable_autorun": False,
                }
            }
        },
    )
    with pytest.raises(RuntimeError, match="METALOG.world / METALOG.kernel"):
        SessionDeployment(session).deploy()


def test_dist_autorun_manual_conflict_rejected(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    enable_autorun=True conflicts with deploy.autorun.mode = 'manual'.
    """
    rootfs_dir = tmp_path / "mock-freebsd-rootfs"
    create_mock_bsd_rootfs(rootfs_dir)

    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-manual-conflict",
        deploy={
            "autorun": {"mode": "manual"},
            "dist": {
                "conflict-dist": {
                    "strategy": {
                        "handler": "bsd-rootfs-archive",
                        "target": str(rootfs_dir),
                    },
                    "enable_autorun": True,
                }
            },
        },
    )
    with pytest.raises(ConfigurationError, match="deploy.autorun.mode is 'manual'"):
        SessionDeployment(session).deploy()


def test_write_extra_files_mtree_escapes_special_chars(tmp_path: Path):
    """
    _write_extra_files_mtree octal-escapes spaces, '=', '#' and records
    symlink, directory, executable, and regular file entries.
    """
    extra = tmp_path / "extra-files"
    (extra / "opt" / "dir=1").mkdir(parents=True)
    (extra / "opt" / "dir=1" / "file #1.txt").write_text("ok\n")
    (extra / "bin").mkdir()
    script = extra / "bin" / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    (extra / "link.sh").symlink_to("bin/run.sh")

    mtree_path = tmp_path / "extra.mtree"
    assert _write_extra_files_mtree(extra, mtree_path)
    lines = mtree_path.read_text().splitlines()

    assert lines[0] == "#mtree"
    by_path = {line.split(" ", 1)[0]: line for line in lines[1:]}
    esc_file = "./opt/dir\\0751/file\\040\\0431.txt"
    assert esc_file in by_path
    assert "type=file" in by_path[esc_file]
    assert "mode=0644" in by_path[esc_file]
    assert "contents=" in by_path[esc_file]
    assert "type=dir" in by_path["./opt/dir\\0751"]
    assert "mode=0755" in by_path["./bin/run.sh"]
    assert "type=link" in by_path["./link.sh"]
    assert "link=bin/run.sh" in by_path["./link.sh"]

    # An empty overlay produces no mtree file
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not _write_extra_files_mtree(empty, tmp_path / "empty.mtree")


def test_bsd_rootfs_resolves_install_directory(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig, mocker
):
    """
    Non-absolute rootfs targets are resolved via
    cheribuild --get-config-option <target>/install-directory before bsdtar runs.
    """
    rootfs_dir = tmp_path / "rootfs-install"
    create_mock_bsd_rootfs(rootfs_dir)

    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-install-dir",
        deploy={
            "dist": {
                "morello-tar": {
                    "strategy": {
                        "handler": "bsd-rootfs-archive",
                        "target": "cheribsd-morello-purecap",
                    },
                    "bundle_session": False,
                    "enable_autorun": False,
                }
            }
        },
    )
    query_mock = mocker.patch.object(
        DeploymentContext, "query_cheribuild_option", return_value=str(rootfs_dir)
    )
    bsdtar_calls = []

    def fake_run(cmd, **kwargs):
        bsdtar_calls.append(cmd)
        # Materialize the temporary output file so the rename succeeds
        out = Path(cmd[cmd.index("-cf") + 1])
        out.write_bytes(b"fake-tar")

    mocker.patch("pycheribenchplot.core.deploy.subprocess.run", side_effect=fake_run)

    manifest = SessionDeployment(session).deploy()

    query_mock.assert_called_once()
    assert query_mock.call_args[0][0] == "cheribsd-morello-purecap/install-directory"
    assert len(bsdtar_calls) == 1
    cmd = [str(a) for a in bsdtar_calls[0]]
    assert Path(cmd[cmd.index("-C") + 1]) == rootfs_dir.resolve()
    assert any(a.endswith("METALOG.world") for a in cmd)
    assert any(a.endswith("METALOG.kernel") for a in cmd)
    assert manifest.dist["morello-tar"].output_path == Path("dist/morello-tar.tar.zst")
