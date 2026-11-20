"""
Shared helpers for deployment tests: fakes for external tools (git, bsdtar),
local fixture builders (git repos, archives, mock rootfs), and assertion helpers.
"""

import hashlib
import stat
import subprocess
import tarfile
from pathlib import Path

from pycheribenchplot.core.config import BenchplotUserConfig, PipelineConfig
from pycheribenchplot.core.session import Session


class FakeGit:
    """
    Callable stand-in for :meth:`DeploymentContext.run_git` backed by an
    in-memory ref -> SHA table. Records every invocation in ``calls`` as
    ``(repo_dir, args)`` tuples.
    """

    def __init__(self, refs: dict[str, str] | None = None, head: str | None = None):
        self.refs = dict(refs or {})
        self.head = head
        self.calls: list[tuple[Path | None, tuple[str, ...]]] = []

    def __call__(
        self,
        *args,
        repo_dir: Path | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        str_args = tuple(str(a) for a in args)
        self.calls.append((repo_dir, str_args))
        op = str_args[0]
        if op == "clone":
            # Mark the repository as existing on disk for .git checks
            (Path(str_args[-1]) / ".git").mkdir(parents=True, exist_ok=True)
            return self._completed(str_args, 0)
        if op in ("remote", "fetch"):
            return self._completed(str_args, 0)
        if op == "checkout":
            self.head = self._resolve(str_args[-1]) or str_args[-1]
            return self._completed(str_args, 0)
        if op == "rev-parse":
            ref = str_args[-1]
            if ref == "--is-shallow-repository":
                return self._completed(str_args, 0, "false\n")
            sha = self._resolve(ref)
            if sha is None:
                return self._completed(str_args, 128)
            return self._completed(str_args, 0, f"{sha}\n")
        raise AssertionError(f"FakeGit does not implement 'git {op}'")

    def _resolve(self, ref: str) -> str | None:
        ref = ref.removesuffix("^{commit}").removeprefix("origin/")
        if ref == "HEAD":
            return self.head
        if ref in self.refs:
            return self.refs[ref]
        if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
            # Full commit SHAs resolve to themselves
            return ref
        return None

    @staticmethod
    def _completed(args, returncode: int, stdout: str = ""):
        return subprocess.CompletedProcess(list(args), returncode, stdout, "")

    def ops(self) -> list[str]:
        return [args[0] for _, args in self.calls]


def init_git_repo(repo_dir: Path) -> tuple[str, str]:
    """
    Create a local git repository with two commits and tag ``v1.0.0`` on the
    first commit. Returns ``(sha1, sha2)``.
    """
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", repo_dir], check=True)
    subprocess.run(
        ["git", "-C", repo_dir, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo_dir, "config", "user.name", "Benchplot Test"],
        check=True,
    )

    (repo_dir / "README.md").write_text("rev 1\n")
    subprocess.run(["git", "-C", repo_dir, "add", "README.md"], check=True)
    subprocess.run(["git", "-C", repo_dir, "commit", "-m", "commit 1"], check=True)
    sha1 = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", repo_dir, "tag", "v1.0.0"], check=True)

    (repo_dir / "README.md").write_text("rev 2\n")
    subprocess.run(["git", "-C", repo_dir, "commit", "-am", "commit 2"], check=True)
    sha2 = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return sha1, sha2


def make_tar_gz(archive_path: Path, files: dict[str, str]) -> str:
    """
    Create a ``.tar.gz`` archive containing ``files`` and return its ``sha256:<hex>`` digest.
    """
    src_root = archive_path.parent / f".staging-{archive_path.stem}"
    src_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tf:
        for rel_name, content in files.items():
            file_path = src_root / rel_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            tf.add(file_path, arcname=rel_name)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def create_mock_bsd_rootfs(rootfs_dir: Path) -> None:
    """
    Populate a mock FreeBSD rootfs directory with default ``/etc/rc.conf``,
    ``/bin/sh``, ``/boot/kernel/kernel``, and valid ``METALOG.world`` + ``METALOG.kernel``.
    """
    (rootfs_dir / "etc").mkdir(parents=True, exist_ok=True)
    (rootfs_dir / "bin").mkdir(parents=True, exist_ok=True)
    (rootfs_dir / "boot" / "kernel").mkdir(parents=True, exist_ok=True)

    (rootfs_dir / "etc" / "rc.conf").write_text(
        'ifconfig_DEFAULT="DHCP"\nsshd_enable="YES"\n'
    )
    sh_bin = rootfs_dir / "bin" / "sh"
    sh_bin.write_text("#!/bin/sh\n")
    sh_bin.chmod(0o555)
    kern_bin = rootfs_dir / "boot" / "kernel" / "kernel"
    kern_bin.write_bytes(b"\x7fELFmock-freebsd-kernel")
    kern_bin.chmod(0o555)

    (rootfs_dir / "METALOG.world").write_text(
        "#mtree\n"
        ". type=dir uname=root gname=wheel mode=0755\n"
        "./bin type=dir uname=root gname=wheel mode=0755\n"
        "./bin/sh type=file uname=root gname=wheel mode=0555\n"
        "./etc type=dir uname=root gname=wheel mode=0755\n"
        "./etc/rc.conf type=file uname=root gname=wheel mode=0644\n"
    )
    (rootfs_dir / "METALOG.kernel").write_text(
        "#mtree\n"
        ". type=dir uname=root gname=wheel mode=0755\n"
        "./boot type=dir uname=root gname=wheel mode=0755\n"
        "./boot/kernel type=dir uname=root gname=wheel mode=0755\n"
        "./boot/kernel/kernel type=file uname=root gname=wheel mode=0555\n"
    )


def extract_tar(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bsdtar", "-xf", archive, "-C", dest], check=True)
    return dest


def assert_tar_all_root_wheel(archive: Path) -> None:
    res = subprocess.run(
        ["bsdtar", "-tvf", archive],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in res.stdout.splitlines():
        assert " root " in line and " wheel " in line, (
            f"Expected root:wheel ownership in tar entry: {line}"
        )


def assert_mode(path: Path, expected: int) -> None:
    actual = stat.S_IMODE(path.stat().st_mode)
    assert actual == expected, f"Expected {oct(expected)} on {path}, got {oct(actual)}"


def make_deploy_session(
    user_config: BenchplotUserConfig,
    tmp_path: Path,
    name: str,
    deploy: dict | None,
    parameterize: dict[str, list] | None = None,
    generators: list[dict] | None = None,
    assets: dict | None = None,
) -> Session:
    """
    Build a session from a minimal pipeline with the given ``deploy`` block.
    """
    raw = {
        "version": "1.0",
        "benchmark": {
            "name": name,
            "iterations": 1,
            "parameterize": parameterize or {"target": ["local"]},
            "generators": generators or [],
        },
        "deploy": deploy,
    }
    if assets:
        raw["assets"] = assets
    pipeline = PipelineConfig.schema().load(raw)
    return Session.make_new(user_config, pipeline, tmp_path / name)
