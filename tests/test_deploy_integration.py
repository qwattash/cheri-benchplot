"""
End-to-end deployment integration test using real local git repositories,
real archives, and real bsdtar. No cheribuild targets are involved.
"""

import json
from pathlib import Path

import pytest

from pycheribenchplot.core.config import BenchplotUserConfig, DeployManifest
from pycheribenchplot.core.deploy import SessionDeployment
from pycheribenchplot.core.session import SESSION_RUN_FILE, Session
from .util.deploy import (
    create_mock_bsd_rootfs,
    extract_tar,
    init_git_repo,
    make_deploy_session,
    make_tar_gz,
)
from .util.session import benchplot_user_config


@pytest.mark.slow
def test_deploy_end_to_end_local(
    tmp_path: Path, benchplot_user_config: BenchplotUserConfig
):
    """
    Full fetch -> dist pipeline against local file:// sources: revisions are
    pinned and persisted, the rootfs archive contains the staged overlay, and
    re-deploying a session reloaded from session-run.json skips all fetch and
    dist work even with the upstream sources removed.
    """
    upstream = tmp_path / "upstream-repo"
    sha1, _ = init_git_repo(upstream)
    helper = tmp_path / "helper-repo"
    helper_sha1, _ = init_git_repo(helper)
    tar_path = tmp_path / "payload.tar.gz"
    tar_digest = make_tar_gz(tar_path, {"data.txt": "payload\n"})
    rootfs_dir = tmp_path / "mock-rootfs"
    create_mock_bsd_rootfs(rootfs_dir)
    marker = tmp_path / "post-dist.log"

    session = make_deploy_session(
        benchplot_user_config,
        tmp_path,
        "s-e2e",
        parameterize={"kconf": ["nocheri"]},
        deploy={
            "autorun": {"mode": "rc.local"},
            "system": [
                {
                    "name": "sys0",
                    "matches": {"kconf": "nocheri"},
                    "run_on": "rootfs-tar",
                    "autorun": {"kernel": "GENERIC-NODEBUG"},
                }
            ],
            "fetch": {
                "tagged": {"url": f"file://{upstream}", "revision": "v1.0.0"},
                "helper": {
                    "url": f"file://{helper}",
                    "revision": helper_sha1,
                    "dest": "src/helper-custom",
                },
                "payload": {"url": f"file://{tar_path}", "digest": tar_digest},
            },
            "dist": {
                "rootfs-tar": {
                    "strategy": {
                        "handler": "bsd-rootfs-archive",
                        "target": str(rootfs_dir),
                    },
                    "bundle_session": True,
                    "command_hooks": {
                        "pre_dist": [
                            "mkdir -p $DEPLOY_EXTRA_FILES/usr/payload",
                            "tar -xf $DEPLOY_DOWNLOADS/payload.tar.gz -C $DEPLOY_EXTRA_FILES/usr/payload",
                        ],
                        "post_dist": [f'echo done >> "{marker}"'],
                    },
                }
            },
        },
    )

    manifest = SessionDeployment(session).deploy()
    assert manifest is not None
    assert isinstance(manifest, DeployManifest)

    # Pinned revisions persisted to session-run.json
    raw = json.loads((session.session_root_path / SESSION_RUN_FILE).read_text())
    assert raw["deploy"]["fetch"]["tagged"]["revision"] == sha1
    assert raw["deploy"]["fetch"]["helper"]["revision"] == helper_sha1
    assert raw["deploy"]["fetch"]["payload"]["digest"] == tar_digest
    assert raw["deploy"]["dist"]["rootfs-tar"]["config_hash"]
    assert marker.read_text() == "done\n"

    # Real clone + detached checkout content
    deploy_root = session.get_deploy_root_path()
    assert (deploy_root / "src" / "tagged" / "README.md").read_text() == "rev 1\n"
    assert (deploy_root / "src" / "helper-custom" / "README.md").read_text() == (
        "rev 1\n"
    )

    # Archive contents: preserved METALOG files, autorun overlay, session bundle,
    # and pre_dist-unpacked payload
    archive = deploy_root / "dist" / "rootfs-tar.tar.zst"
    extracted = extract_tar(archive, tmp_path / "extracted")
    assert (extracted / "etc" / "rc.conf").read_text() == (
        'ifconfig_DEFAULT="DHCP"\nsshd_enable="YES"\n'
    )
    assert (extracted / "etc" / "rc.conf.d" / "benchplot").is_file()
    assert (extracted / "etc" / "rc.local").is_file()
    guest = extracted / "opt" / "benchmark-output"
    assert (guest / "session-run.json").is_file()
    assert (guest / "run" / "auto-run.sh").is_file()
    assert (guest / "run" / "run-target-sys0.sh").is_file()
    assert (extracted / "usr" / "payload" / "data.txt").read_text() == "payload\n"

    # Reload from session-run.json with all upstream sources removed:
    # fetch cache and dist cache must make the re-deploy fully offline.
    upstream.rename(tmp_path / "upstream-repo.bak")
    helper.rename(tmp_path / "helper-repo.bak")
    tar_path.rename(tmp_path / "payload.tar.gz.bak")
    marker.unlink()

    reloaded = Session.from_path(benchplot_user_config, session.session_root_path)
    assert reloaded is not None
    assert isinstance(reloaded.config.deploy, DeployManifest)
    manifest2 = SessionDeployment(reloaded).deploy()
    assert manifest2 is not None
    assert not marker.exists()  # dist was skipped: post_dist hook did not run
    assert (
        manifest2.dist["rootfs-tar"].config_hash
        == manifest.dist["rootfs-tar"].config_hash
    )
