import dataclasses as dc
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Self

from .config import (
    AssembledDistTargetConfig,
    AutorunMode,
    DeployConfig,
    DeployManifest,
    DistHandler,
    DistStrategyConfig,
    DistTargetConfig,
    FetchTargetConfig,
)
from .error import ConfigurationError
from .session import SESSION_RUN_FILE, Session
from .util import new_logger, resolve_system_command

_HASH_CHUNK_SIZE = 65536
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], val)
        else:
            merged[key] = val
    return merged


def _is_git_sha(revision: str | None) -> bool:
    return revision is not None and _GIT_SHA_RE.fullmatch(revision) is not None


def _parse_digest_spec(digest_spec: str) -> tuple[str, str]:
    algo, sep, expected_hex = digest_spec.partition(":")
    if sep != ":" or algo not in {"sha256", "sha512"} or not expected_hex:
        raise ConfigurationError(f"Invalid digest specification '{digest_spec}'")
    return algo, expected_hex.lower()


def _compute_file_digest(path: Path, algo: str) -> str:
    hasher = hashlib.new(algo)
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def _filename_from_url(url: str, fallback: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if name := Path(urllib.parse.unquote(parsed.path)).name:
        return name
    if fallback_name := Path(fallback).name:
        return fallback_name
    raise ConfigurationError(f"Cannot infer download filename from URL '{url}'")


def _mtree_escape(value: str) -> str:
    """
    Escape an mtree token using FreeBSD strsvis(3) VIS_OCTAL rules so spaces,
    tabs, backslashes, comments, key separators, glob characters, and control
    or non-ASCII bytes are parsed unambiguously by libarchive / bsdtar.
    """
    escaped: list[str] = []
    for ch in value:
        code = ord(ch)
        if code < 33 or code > 126 or ch in ("\\", "[", "]", "?", "*", "#", "="):
            for byte in ch.encode("utf-8"):
                escaped.append(f"\\{byte:03o}")
        else:
            escaped.append(ch)
    return "".join(escaped)


def _write_extra_files_mtree(extra_files_dir: Path, mtree_path: Path) -> bool:
    entries = sorted(
        extra_files_dir.rglob("*"),
        key=lambda p: p.relative_to(extra_files_dir).as_posix(),
    )
    if not entries:
        return False

    lines = ["#mtree"]
    for entry in entries:
        esc_rel = _mtree_escape("./" + entry.relative_to(extra_files_dir).as_posix())
        if entry.is_symlink():
            esc_target = _mtree_escape(str(entry.readlink()))
            lines.append(
                f"{esc_rel} type=link uname=root gname=wheel mode=0755 link={esc_target}"
            )
        elif entry.is_dir():
            lines.append(f"{esc_rel} type=dir uname=root gname=wheel mode=0755")
        elif entry.is_file():
            mode = "0755" if (entry.stat().st_mode & 0o111) != 0 else "0644"
            esc_src = _mtree_escape(str(entry.absolute()))
            lines.append(
                f"{esc_rel} type=file uname=root gname=wheel mode={mode} contents={esc_src}"
            )

    mtree_path.write_text("\n".join(lines) + "\n")
    return True


def _resolve_metalog_args(rootfs_dir: Path) -> list[str]:
    abs_rootfs = rootfs_dir.resolve()
    world = abs_rootfs / "METALOG.world"
    kernel = abs_rootfs / "METALOG.kernel"
    single = abs_rootfs / "METALOG"

    if world.is_file() and kernel.is_file():
        return [f"@{world}", f"@{kernel}"]
    if world.is_file() or kernel.is_file():
        raise RuntimeError(
            f"Found only one of METALOG.world / METALOG.kernel in {abs_rootfs} (both are required)"
        )
    if single.is_file():
        return [f"@{single}"]
    raise RuntimeError(
        f"No METALOG.world/METALOG.kernel or METALOG found in {abs_rootfs}"
    )


def _stage_session_bundle(
    session: Session, extra_files_dir: Path, guest_rel: PurePosixPath
) -> None:
    guest_session_dir = extra_files_dir / guest_rel.as_posix()
    guest_session_dir.mkdir(parents=True, exist_ok=True)

    session_run_src = session.session_root_path / SESSION_RUN_FILE
    session_run_dst = guest_session_dir / SESSION_RUN_FILE
    if session_run_src.is_file():
        shutil.copy2(session_run_src, session_run_dst)
    else:
        session_run_dst.write_text(session.config.emit_json() + "\n")

    for src, subdir in (
        (session.get_data_root_path(), "run"),
        (session.get_asset_root_path(), "assets"),
    ):
        if src.is_dir() and any(src.iterdir()):
            shutil.copytree(
                src, guest_session_dir / subdir, symlinks=True, dirs_exist_ok=True
            )


def _stage_autorun_files(
    session: Session,
    extra_files_dir: Path,
    guest_rel: PurePosixPath,
    manifest: DeployManifest,
    logger: logging.Logger,
) -> None:
    data_root = session.get_data_root_path()
    if not (data_root / "auto-run.sh").is_file():
        logger.info(
            "No auto-run.sh generated for session %s; skipping autorun rc overlay",
            session.name,
        )
        return

    autorun_cfg = manifest.autorun
    guest_run_dir = PurePosixPath("/") / guest_rel / "run"
    reboot_val = "YES" if autorun_cfg.reboot else "NO"
    rc_conf_file = extra_files_dir / "etc" / "rc.conf.d" / "benchplot"
    rc_conf_file.parent.mkdir(parents=True, exist_ok=True)
    rc_conf_file.write_text(
        'benchplot_enable="YES"\n'
        f'benchplot_session="{guest_run_dir}"\n'
        f'benchplot_user="{autorun_cfg.user}"\n'
        f'benchplot_reboot="{reboot_val}"\n'
        f'benchplot_settle_delay="{autorun_cfg.settle_delay}"\n'
    )
    rc_conf_file.chmod(0o644)

    rc_map = {
        AutorunMode.RC_LOCAL: ("etc/rc.local", "rc.local"),
        AutorunMode.RC_D: ("usr/local/etc/rc.d/benchplot", "rc-autorun.sh"),
    }
    if rc_pair := rc_map.get(autorun_cfg.mode):
        rc_dst = extra_files_dir / rc_pair[0]
        rc_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_root / rc_pair[1], rc_dst)
        rc_dst.chmod(0o755)


def _assemble_bsd_rootfs_archive(
    context: DeploymentContext,
    dist_id: str,
    target: str,
    extra_files_dir: Path,
    output_path: Path,
    manifest: DeployManifest,
    logger: logging.Logger,
) -> None:
    target_path = Path(target)
    if target_path.is_absolute():
        rootfs_dir = target_path
    else:
        cfg_path = context.ensure_cheribuild_config(manifest)
        rootfs_dir = Path(
            context.query_cheribuild_option(f"{target}/install-directory", cfg_path)
        )
    if not rootfs_dir.is_dir():
        raise RuntimeError(
            f"Rootfs directory for dist '{dist_id}' does not exist: {rootfs_dir}"
        )

    bsdtar_bin = resolve_system_command("bsdtar", logger=logger)
    abs_rootfs = rootfs_dir.resolve()
    tar_args = _resolve_metalog_args(abs_rootfs)

    tmp_output = output_path.with_name(f".tmp.{output_path.name}")
    mtree_path = context.extra_files_root / f"{dist_id}.mtree"
    try:
        if _write_extra_files_mtree(extra_files_dir, mtree_path):
            tar_args.append(f"@{mtree_path}")
        cmd = [bsdtar_bin, "-a", "-cf", tmp_output, "-C", abs_rootfs, *tar_args]
        logger.info("Archiving rootfs %s -> %s", abs_rootfs, output_path)
        subprocess.run(cmd, check=True)
        tmp_output.replace(output_path)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise
    finally:
        mtree_path.unlink(missing_ok=True)


@dc.dataclass
class DeploymentContext:
    """
    Shared deployment directory layout, hook environment, and cheribuild helpers.
    """

    session: Session
    deploy_root: Path
    src_root: Path
    downloads_dir: Path
    build_root: Path
    sdk_root: Path
    extra_files_root: Path
    dist_dir: Path
    cheribuild_config_path: Path
    logger: logging.Logger = dc.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = new_logger("deploy", parent=self.session.logger)

    @classmethod
    def from_session(cls, session: Session) -> Self:
        deploy_root = session.get_deploy_root_path()
        return cls(
            session=session,
            deploy_root=deploy_root,
            src_root=deploy_root / "src",
            downloads_dir=deploy_root / "downloads",
            build_root=deploy_root / "build",
            sdk_root=deploy_root / "cherisdk",
            extra_files_root=deploy_root / "extra-files",
            dist_dir=deploy_root / "dist",
            cheribuild_config_path=deploy_root / "cheribuild.json",
        )

    def ensure_dirs(self) -> None:
        for directory in (
            self.deploy_root,
            self.src_root,
            self.downloads_dir,
            self.build_root,
            self.sdk_root,
            self.extra_files_root,
            self.dist_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_confined_path(self, root: Path, rel_path: Path, label: str) -> Path:
        if rel_path.is_absolute():
            raise ConfigurationError(
                f"{label} '{rel_path}' must be a relative path inside '{root}'"
            )
        root_resolved = root.resolve()
        resolved = (root_resolved / rel_path).resolve()
        if resolved == root_resolved or not resolved.is_relative_to(root_resolved):
            raise ConfigurationError(f"{label} '{rel_path}' escapes directory '{root}'")
        return resolved

    def resolve_dest_path(self, dest: Path | None, default: Path) -> Path:
        if dest is None:
            return default
        return self.resolve_confined_path(self.deploy_root, dest, "Fetch destination")

    def resolve_cheribuild_bin(self) -> Path:
        deploy_cfg = self.session.config.deploy
        if deploy_cfg is not None and "cheribuild" in deploy_cfg.fetch:
            cb_fetch = deploy_cfg.fetch["cheribuild"]
            cb_dir = self.resolve_dest_path(cb_fetch.dest, self.src_root / "cheribuild")
            return cb_dir / "cheribuild.py"
        cb_base = self.session.user_config.cheribuild_path or (
            self.session.user_config.src_path / "cheribuild"
        )
        return cb_base / "cheribuild.py"

    def require_cheribuild_bin(self) -> Path:
        cb_bin = self.resolve_cheribuild_bin()
        if not cb_bin.is_file():
            raise RuntimeError(f"cheribuild executable not found at {cb_bin}")
        return cb_bin

    def build_cheribuild_json(self, manifest: DeployManifest) -> dict[str, Any]:
        base: dict[str, Any] = {
            "source-root": str(self.src_root),
            "build-root": str(self.build_root),
            "output-root": str(self.sdk_root),
        }
        for target_cfg in manifest.fetch.values():
            if (
                target_cfg.url.startswith("cheribuild://")
                and target_cfg.dest is not None
            ):
                cb_target = target_cfg.url.removeprefix("cheribuild://").strip()
                src_dir = self.resolve_dest_path(target_cfg.dest, self.src_root)
                base.setdefault(cb_target, {})["source-directory"] = str(src_dir)

        if manifest.cheribuild is None:
            return base
        return _deep_merge_dict(base, dict(manifest.cheribuild.config))

    def ensure_cheribuild_config(self, manifest: DeployManifest) -> Path:
        cfg_data = json.dumps(self.build_cheribuild_json(manifest), indent=4) + "\n"
        self.cheribuild_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.cheribuild_config_path.write_text(cfg_data)
        return self.cheribuild_config_path

    def query_cheribuild_option(self, option_key: str, cfg_path: Path) -> str:
        cb_bin = self.require_cheribuild_bin()
        cmd = [
            sys.executable,
            cb_bin,
            "--config-file",
            cfg_path,
            "--get-config-option",
            option_key,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to query cheribuild option '{option_key}': {res.stderr.strip()}"
            )
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(
                f"cheribuild returned empty output for option '{option_key}'"
            )
        return lines[-1]

    def run_cheribuild(self, cfg_path: Path, args: list[str | Path]) -> None:
        cb_bin = self.require_cheribuild_bin()
        cmd: list[str | Path] = [
            sys.executable,
            cb_bin,
            "--config-file",
            cfg_path,
            *args,
        ]
        self.logger.info("Running cheribuild: %s", " ".join(map(str, cmd)))
        subprocess.run(cmd, check=True)

    def run_git(
        self,
        *args: str | Path,
        repo_dir: Path | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        """
        Run a git command, optionally confined to a repository directory.
        """
        cmd: list[str | Path] = ["git"]
        if repo_dir is not None:
            cmd += ["-C", repo_dir]
        cmd += list(args)
        return subprocess.run(
            cmd, check=check, capture_output=capture_output, text=True
        )

    def hook_env(
        self,
        dist_id: str | None = None,
        extra_files_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "SESSION_ROOT": str(self.session.session_root_path),
                "DEPLOY_ROOT": str(self.deploy_root),
                "DEPLOY_SRC_ROOT": str(self.src_root),
                "DEPLOY_DOWNLOADS": str(self.downloads_dir),
                "DEPLOY_BUILD_ROOT": str(self.build_root),
                "DEPLOY_SDK_ROOT": str(self.sdk_root),
                "DEPLOY_DIST_DIR": str(self.dist_dir),
                "CHERIBUILD": str(self.resolve_cheribuild_bin()),
            }
        )
        if dist_id is not None:
            env["DEPLOY_DIST_ID"] = dist_id
        if extra_files_dir is not None:
            env["DEPLOY_EXTRA_FILES"] = str(extra_files_dir)
        if output_path is not None:
            env["DEPLOY_DIST_OUTPUT"] = str(output_path)
        return env

    def run_hooks(self, phase: str, commands: list[str], env: dict[str, str]) -> None:
        for cmd in commands:
            self.logger.info("Running %s hook: %s", phase, cmd)
            subprocess.run(cmd, shell=True, cwd=self.deploy_root, env=env, check=True)


class DeploySourceFetcher:
    """
    Fetch and pin external git repositories, cheribuild targets, and verified archives.
    """

    def __init__(
        self, session: Session, context: DeploymentContext, force: bool = False
    ):
        self.session = session
        self.context = context
        self.force = force
        self.logger = new_logger("fetch", parent=context.logger)

    def fetch_all(self, manifest: DeployManifest) -> None:
        fetch_entries = manifest.fetch
        if "cheribuild" in fetch_entries:
            self._fetch_single_target(
                "cheribuild", fetch_entries["cheribuild"], manifest
            )
        has_cheribuild = any(
            key == "cheribuild" or cfg.url.startswith("cheribuild://")
            for key, cfg in fetch_entries.items()
        )
        if has_cheribuild:
            self.context.ensure_cheribuild_config(manifest)

        for key, target_cfg in fetch_entries.items():
            if key != "cheribuild":
                self._fetch_single_target(key, target_cfg, manifest)

        if hooks := manifest.command_hooks.get("post_fetch"):
            self.context.run_hooks("post_fetch", hooks, self.context.hook_env())

    def _fetch_single_target(
        self, key: str, target_cfg: FetchTargetConfig, manifest: DeployManifest
    ) -> None:
        if target_cfg.url.startswith("cheribuild://"):
            self._fetch_cheribuild_target(key, target_cfg, manifest)
        elif target_cfg.archive or target_cfg.digest is not None:
            self._fetch_archive(key, target_cfg)
        else:
            self._fetch_git_repo(key, target_cfg)

    def _fetch_cheribuild_target(
        self, key: str, target_cfg: FetchTargetConfig, manifest: DeployManifest
    ) -> None:
        cb_target = target_cfg.url.removeprefix("cheribuild://").strip()
        cfg_path = self.context.ensure_cheribuild_config(manifest)
        opt_key = f"{cb_target}/source-directory"
        src_dir = Path(self.context.query_cheribuild_option(opt_key, cfg_path))

        cached = self._check_cached_git_revision(key, target_cfg.revision, src_dir)
        if cached is not None:
            target_cfg.revision = cached
            return

        need_clone = (
            not (src_dir / ".git").exists()
            or self._resolve_git_commit(src_dir, target_cfg.revision) is None
        )
        if need_clone:
            fetch_flags = [
                "--force",
                "--skip-configure",
                "--skip-build",
                "--skip-install",
            ]
            self.context.run_cheribuild(cfg_path, [*fetch_flags, cb_target])
            src_dir = Path(self.context.query_cheribuild_option(opt_key, cfg_path))

        target_cfg.revision = self._checkout_pinned_revision(
            src_dir, target_cfg.revision, force_fetch=self.force and not need_clone
        )

    def _fetch_archive(self, key: str, target_cfg: FetchTargetConfig) -> None:
        filename = _filename_from_url(target_cfg.url, fallback=key)
        resolved_dest = self.context.resolve_dest_path(
            target_cfg.dest, self.context.downloads_dir
        )
        if resolved_dest == self.context.downloads_dir or resolved_dest.is_dir():
            dest_file = resolved_dest / filename
        else:
            dest_file = resolved_dest

        algo, expected_hex = _parse_digest_spec(target_cfg.digest)
        if not self.force and dest_file.is_file():
            actual_hex = _compute_file_digest(dest_file, algo).lower()
            if actual_hex == expected_hex:
                self.logger.info(
                    "Skipping download '%s': '%s' already verified (%s)",
                    key,
                    dest_file,
                    algo,
                )
                target_cfg.digest = f"{algo}:{actual_hex}"
                return

        self.logger.info(
            "Downloading '%s' from %s to %s", key, target_cfg.url, dest_file
        )
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest_file.with_name(f"{dest_file.name}.part")
        hasher = hashlib.new(algo)
        try:
            with (
                urllib.request.urlopen(target_cfg.url) as response,
                part_path.open("wb") as out_file,
            ):
                while chunk := response.read(_HASH_CHUNK_SIZE):
                    out_file.write(chunk)
                    hasher.update(chunk)
            actual_hex = hasher.hexdigest().lower()
            if actual_hex != expected_hex:
                raise RuntimeError(
                    f"Digest mismatch for '{key}' ({target_cfg.url}): "
                    f"expected {algo}:{expected_hex}, got {algo}:{actual_hex}"
                )
            part_path.replace(dest_file)
        except Exception:
            part_path.unlink(missing_ok=True)
            raise

        target_cfg.digest = f"{algo}:{actual_hex}"

    def _fetch_git_repo(self, key: str, target_cfg: FetchTargetConfig) -> None:
        repo_dir = self.context.resolve_dest_path(
            target_cfg.dest, self.context.src_root / key
        )
        cached = self._check_cached_git_revision(key, target_cfg.revision, repo_dir)
        if cached is not None:
            target_cfg.revision = cached
            return

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        freshly_cloned = not (repo_dir / ".git").exists()
        if freshly_cloned:
            self.logger.info("Cloning '%s' (%s) into %s", key, target_cfg.url, repo_dir)
            self.context.run_git("clone", "--no-checkout", target_cfg.url, repo_dir)
        else:
            self.context.run_git(
                "remote", "set-url", "origin", target_cfg.url, repo_dir=repo_dir
            )

        target_cfg.revision = self._checkout_pinned_revision(
            repo_dir, target_cfg.revision, force_fetch=self.force and not freshly_cloned
        )

    def _check_cached_git_revision(
        self, key: str, revision: str | None, repo_dir: Path
    ) -> str | None:
        if self.force or not _is_git_sha(revision) or not (repo_dir / ".git").exists():
            return None
        head_sha = self._git_rev_parse(repo_dir, "HEAD")
        if head_sha is not None and head_sha.lower() == revision.lower():
            self.logger.info(
                "Skipping fetch '%s': '%s' already at %s", key, repo_dir, head_sha[:12]
            )
            return head_sha
        return None

    def _git_rev_parse(self, repo_dir: Path, rev: str) -> str | None:
        res = self.context.run_git(
            "rev-parse",
            "--verify",
            "--quiet",
            rev,
            repo_dir=repo_dir,
            check=False,
            capture_output=True,
        )
        if res.returncode != 0:
            return None
        return res.stdout.strip() or None

    def _resolve_git_commit(self, repo_dir: Path, revision: str) -> str | None:
        return self._git_rev_parse(
            repo_dir, f"{revision}^{{commit}}"
        ) or self._git_rev_parse(repo_dir, f"origin/{revision}^{{commit}}")

    def _is_shallow_repo(self, repo_dir: Path) -> bool:
        res = self.context.run_git(
            "rev-parse",
            "--is-shallow-repository",
            repo_dir=repo_dir,
            check=False,
            capture_output=True,
        )
        return res.returncode == 0 and res.stdout.strip() == "true"

    def _checkout_pinned_revision(
        self, repo_dir: Path, revision: str, force_fetch: bool
    ) -> str:
        if force_fetch:
            self._git_fetch_updates(repo_dir)

        commit = self._resolve_git_commit(repo_dir, revision)
        if commit is None and not force_fetch:
            self._git_fetch_updates(repo_dir)
            commit = self._resolve_git_commit(repo_dir, revision)

        if commit is None:
            fetch_rev = self.context.run_git(
                "fetch",
                "origin",
                revision,
                repo_dir=repo_dir,
                check=False,
                capture_output=True,
            )
            if fetch_rev.returncode == 0:
                commit = self._git_rev_parse(repo_dir, "FETCH_HEAD^{commit}")

        if commit is None:
            raise RuntimeError(
                f"Failed to resolve git revision '{revision}' in {repo_dir}"
            )

        self.context.run_git("checkout", "--detach", commit, repo_dir=repo_dir)
        resolved = self._git_rev_parse(repo_dir, "HEAD")
        if resolved is None:
            raise RuntimeError(f"Failed to determine HEAD commit in {repo_dir}")
        return resolved

    def _git_fetch_updates(self, repo_dir: Path) -> None:
        args = ["fetch", "--all", "--tags", "--force", "--prune"]
        if self._is_shallow_repo(repo_dir):
            args.append("--unshallow")
        self.context.run_git(*args, repo_dir=repo_dir)


class DeployBuilder:
    """
    Generate cheribuild configuration and execute configured cheribuild targets.
    """

    def __init__(
        self, session: Session, context: DeploymentContext, force: bool = False
    ):
        self.session = session
        self.context = context
        self.force = force
        self.logger = new_logger("build", parent=context.logger)

    def build(self, manifest: DeployManifest) -> None:
        hook_env = self.context.hook_env()
        self.context.run_hooks(
            "pre_build", manifest.command_hooks.get("pre_build", []), hook_env
        )

        cb_cfg = manifest.cheribuild
        if cb_cfg is not None and cb_cfg.targets:
            cfg_path = self.context.ensure_cheribuild_config(manifest)
            self.context.run_cheribuild(
                cfg_path, ["--force", "--skip-update", *cb_cfg.targets]
            )

        self.context.run_hooks(
            "post_build", manifest.command_hooks.get("post_build", []), hook_env
        )


class DeployDistAssembler:
    """
    Stage extra-files overlays and assemble deployable disk images or rootfs archives.
    """

    def __init__(
        self, session: Session, context: DeploymentContext, force: bool = False
    ):
        self.session = session
        self.context = context
        self.force = force
        self.logger = new_logger("dist", parent=context.logger)

    def assemble(
        self, manifest: DeployManifest, dist_targets: list[str] | None = None
    ) -> None:
        if dist_targets is not None:
            selected_ids = dist_targets
        else:
            selected_ids = list(manifest.dist.keys())
        if not selected_ids:
            return

        autorun_flags = {}
        needs_scripts = False
        for dist_id in selected_ids:
            dist_cfg = manifest.dist[dist_id]
            autorun_flags[dist_id] = self._should_enable_autorun(
                dist_id, dist_cfg, manifest
            )
            if dist_cfg.bundle_session or autorun_flags[dist_id]:
                needs_scripts = True
        if needs_scripts:
            self._ensure_session_scripts(manifest)

        cb_json = self.context.build_cheribuild_json(manifest)
        deploy_root_resolved = self.context.deploy_root.resolve()
        for dist_id in selected_ids:
            dist_cfg = manifest.dist[dist_id]
            output_path = self._resolve_dist_output_path(dist_id, dist_cfg.strategy)
            relative_path = output_path.relative_to(deploy_root_resolved)
            dist_hash = self._compute_dist_config_hash(
                dist_id, dist_cfg, cb_json, manifest
            )

            if (
                not self.force
                and dist_cfg.output_path == relative_path
                and dist_cfg.config_hash == dist_hash
                and output_path.is_file()
            ):
                self.logger.info(
                    "Skipping dist '%s': '%s' is up to date (%s)",
                    dist_id,
                    output_path,
                    dist_hash[:12],
                )
                continue

            extra_files_dir = self.context.extra_files_root / dist_id
            self._prepare_extra_files_dir(
                extra_files_dir,
                manifest,
                dist_cfg.bundle_session,
                autorun_flags[dist_id],
            )
            hook_env = self.context.hook_env(dist_id, extra_files_dir, output_path)
            self.context.run_hooks(
                "pre_dist", dist_cfg.command_hooks.get("pre_dist", []), hook_env
            )
            self._assemble_dist_artefact(
                dist_id, dist_cfg, extra_files_dir, output_path, manifest
            )
            self.context.run_hooks(
                "post_dist", dist_cfg.command_hooks.get("post_dist", []), hook_env
            )
            manifest.dist[dist_id] = AssembledDistTargetConfig(
                strategy=dist_cfg.strategy,
                bundle_session=dist_cfg.bundle_session,
                enable_autorun=dist_cfg.enable_autorun,
                command_hooks=dist_cfg.command_hooks,
                output_path=relative_path,
                config_hash=dist_hash,
            )

    def _resolve_dist_output_path(
        self, dist_id: str, strategy: DistStrategyConfig
    ) -> Path:
        ext = "img" if strategy.handler == DistHandler.CHERIBUILD else "tar.zst"
        name = Path(strategy.output_name or f"{dist_id}.{ext}")
        return self.context.resolve_confined_path(
            self.context.dist_dir, name, "Dist output_name"
        )

    def _should_enable_autorun(
        self, dist_id: str, dist_cfg: DistTargetConfig, manifest: DeployManifest
    ) -> bool:
        mode = manifest.autorun.mode
        if dist_cfg.enable_autorun is None:
            return mode != AutorunMode.MANUAL
        if dist_cfg.enable_autorun and mode == AutorunMode.MANUAL:
            raise ConfigurationError(
                f"Dist target '{dist_id}' enables autorun, but deploy.autorun.mode is 'manual'"
            )
        return dist_cfg.enable_autorun

    def _session_scripts_ready(self, manifest: DeployManifest) -> bool:
        data_root = self.session.get_data_root_path()
        if not data_root.is_dir():
            return False
        if not all(
            b.get_run_script_path().is_file() for b in self.session.all_benchmarks()
        ):
            return False
        if not manifest.system:
            return True
        need_autorun = any(s.autorun is not None for s in manifest.system)
        return any(data_root.glob("run-target-*.sh")) and (
            not need_autorun or (data_root / "auto-run.sh").is_file()
        )

    def _ensure_session_scripts(self, manifest: DeployManifest) -> None:
        if self._session_scripts_ready(manifest):
            return
        self.logger.info("Generating session scripts before bundling...")
        scheduler = self.session.make_scheduler()
        self.session.generate(scheduler)
        if scheduler.failed_tasks:
            raise RuntimeError(
                "Failed to generate session execution scripts before deployment"
            )

    def _compute_dist_config_hash(
        self,
        dist_id: str,
        dist_cfg: DistTargetConfig,
        cheribuild_json: dict[str, Any],
        manifest: DeployManifest,
    ) -> str:
        base_cfg = DistTargetConfig(
            strategy=dist_cfg.strategy,
            bundle_session=dist_cfg.bundle_session,
            enable_autorun=dist_cfg.enable_autorun,
            command_hooks=dist_cfg.command_hooks,
        )
        fetch_fingerprints = {}
        for key, rec in sorted(manifest.fetch.items()):
            fetch_fingerprints[key] = {
                "url": rec.url,
                "revision": rec.revision,
                "digest": rec.digest,
            }

        system_configs = []
        for sys_entry in manifest.system:
            if sys_entry.run_on == dist_id:
                system_configs.append(sys_entry.schema().dump(sys_entry))

        payload = {
            "dist_cfg": base_cfg.schema().dump(base_cfg),
            "autorun": manifest.autorun.schema().dump(manifest.autorun),
            "remote_session_path": str(self.session.config.remote_session_path),
            "cheribuild_json": cheribuild_json,
            "fetch": fetch_fingerprints,
            "systems": system_configs,
        }
        return _canonical_sha256(payload)

    def _guest_session_relpath(self) -> PurePosixPath:
        raw = self.session.config.remote_session_path.as_posix().lstrip("/")
        rel_path = PurePosixPath(raw)
        if not raw or ".." in rel_path.parts:
            raise ConfigurationError(
                f"Invalid remote_session_path '{self.session.config.remote_session_path}'"
            )
        return rel_path

    def _prepare_extra_files_dir(
        self,
        extra_files_dir: Path,
        manifest: DeployManifest,
        bundle_session: bool,
        enable_autorun: bool,
    ) -> None:
        guest_rel = self._guest_session_relpath()
        if extra_files_dir.exists():
            shutil.rmtree(extra_files_dir)
        extra_files_dir.mkdir(parents=True, exist_ok=True)
        if bundle_session:
            _stage_session_bundle(self.session, extra_files_dir, guest_rel)
        if enable_autorun:
            _stage_autorun_files(
                self.session, extra_files_dir, guest_rel, manifest, self.logger
            )

    def _assemble_dist_artefact(
        self,
        dist_id: str,
        dist_cfg: DistTargetConfig,
        extra_files_dir: Path,
        output_path: Path,
        manifest: DeployManifest,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        strategy = dist_cfg.strategy
        if strategy.handler == DistHandler.CHERIBUILD:
            cfg_path = self.context.ensure_cheribuild_config(manifest)
            cb_target = strategy.target
            args: list[str | Path] = [
                "--force",
                "--skip-update",
                f"--{cb_target}/extra-files",
                extra_files_dir,
                f"--{cb_target}/path",
                output_path,
                cb_target,
            ]
            self.context.run_cheribuild(cfg_path, args)
        elif strategy.handler == DistHandler.BSD_ROOTFS_ARCHIVE:
            _assemble_bsd_rootfs_archive(
                self.context,
                dist_id,
                strategy.target,
                extra_files_dir,
                output_path,
                manifest,
                self.logger,
            )
        else:
            raise ConfigurationError(f"Unsupported dist handler '{strategy.handler}'")


class SessionDeployment:
    """
    Coordinate source fetching, cheribuild target builds, and deployable image/archive
    assembly for a benchmark session.
    """

    def __init__(self, session: Session, force: bool = False):
        self.session = session
        self.force = force
        self.logger = new_logger("deploy", parent=session.logger)
        self.context = DeploymentContext.from_session(session)
        self.fetcher = DeploySourceFetcher(session, self.context, force)
        self.builder = DeployBuilder(session, self.context, force)
        self.assembler = DeployDistAssembler(session, self.context, force)

    def deploy(
        self,
        skip_fetch: bool = False,
        skip_build: bool = False,
        skip_dist: bool = False,
        dist_targets: list[str] | None = None,
    ) -> DeployManifest | None:
        manifest = self.session.config.deploy
        self._validate_dist_targets(manifest, dist_targets)
        if not self._has_deployment_actions(manifest):
            self.logger.info(
                "Session %s has no deployment actions configured, nothing to do",
                self.session.name,
            )
            return None

        if isinstance(manifest, DeployConfig):
            manifest = DeployManifest.from_deploy_config(manifest)
            self.session.config.deploy = manifest

        self.context.ensure_dirs()
        if not skip_fetch:
            self.fetcher.fetch_all(manifest)
            self._persist_session_config()
        if not skip_build:
            self.builder.build(manifest)
            self._persist_session_config()
        if not skip_dist:
            self.assembler.assemble(manifest, dist_targets)
            self._persist_session_config()
        return manifest

    def _persist_session_config(self) -> None:
        session_run_path = self.session.session_root_path / SESSION_RUN_FILE
        tmp_path = session_run_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(self.session.config.emit_json() + "\n")
            tmp_path.replace(session_run_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _has_deployment_actions(
        self, deploy_cfg: DeployConfig | DeployManifest | None
    ) -> bool:
        if deploy_cfg is None:
            return False
        return bool(
            deploy_cfg.fetch
            or deploy_cfg.cheribuild is not None
            or deploy_cfg.dist
            or any(deploy_cfg.command_hooks.values())
        )

    def _validate_dist_targets(
        self,
        deploy_cfg: DeployConfig | DeployManifest | None,
        dist_targets: list[str] | None,
    ) -> None:
        if dist_targets is None:
            return
        configured = deploy_cfg.dist if deploy_cfg is not None else {}
        if unknown := [target for target in dist_targets if target not in configured]:
            raise ConfigurationError(
                f"Unknown deployment dist target(s): {', '.join(unknown)}"
            )
