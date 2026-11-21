"""
Flag-wiring tests for the `session deploy`, `session clean --deploy`, and
`session push --image` CLI commands.

These tests invoke the SessionSubCommand handlers directly with mocked
backends: they only verify that CLI flags drive the backend interfaces
correctly and do not exercise any real deployment, archive, or scp side
effects.
"""

from unittest.mock import sentinel

import pytest

from pycheribenchplot.benchplot_cli import SessionSubCommand
from pycheribenchplot.core.config import BenchplotUserConfig
from pycheribenchplot.core.error import ConfigurationError
from .util.cli import make_subcommand, parse_args
from .util.session import benchplot_user_config


@pytest.fixture
def session_cmd():
    return make_subcommand(SessionSubCommand())


def test_deploy_flags(session_cmd, mocker, benchplot_user_config):
    """
    `session deploy` with default flags constructs SessionDeployment without
    force and runs all deployment phases for all dist targets.
    """
    get_session = mocker.patch.object(
        SessionSubCommand, "_get_session", return_value=sentinel.session
    )
    deployment_cls = mocker.patch("pycheribenchplot.benchplot_cli.SessionDeployment")

    args = parse_args(session_cmd, ["deploy", "target-session"])
    session_cmd.handle(benchplot_user_config, args)

    get_session.assert_called_once_with(benchplot_user_config, args)
    deployment_cls.assert_called_once_with(sentinel.session, force=False)
    deployment_cls.return_value.deploy.assert_called_once_with(
        skip_fetch=False, skip_build=False, skip_dist=False, dist_targets=None
    )


def test_deploy_all_flags(session_cmd, mocker, benchplot_user_config):
    """
    `session deploy -f --skip-* --dist-target` forwards every flag to the
    SessionDeployment backend.
    """
    mocker.patch.object(
        SessionSubCommand, "_get_session", return_value=sentinel.session
    )
    deployment_cls = mocker.patch("pycheribenchplot.benchplot_cli.SessionDeployment")

    args = parse_args(
        session_cmd,
        [
            "deploy",
            "-f",
            "--skip-fetch",
            "--skip-build",
            "--skip-dist",
            "--dist-target",
            "img-a",
            "--dist-target",
            "img-b",
            "target-session",
        ],
    )
    session_cmd.handle(benchplot_user_config, args)

    deployment_cls.assert_called_once_with(sentinel.session, force=True)
    deployment_cls.return_value.deploy.assert_called_once_with(
        skip_fetch=True,
        skip_build=True,
        skip_dist=True,
        dist_targets=["img-a", "img-b"],
    )


def test_deploy_backend_error_propagates(
    session_cmd, mocker, benchplot_user_config: BenchplotUserConfig
):
    """
    Backend failures such as unknown dist targets propagate out of the
    `session deploy` handler.
    """
    mocker.patch.object(
        SessionSubCommand, "_get_session", return_value=sentinel.session
    )
    deployment_cls = mocker.patch("pycheribenchplot.benchplot_cli.SessionDeployment")
    deployment_cls.return_value.deploy.side_effect = ConfigurationError(
        "Unknown deployment dist target(s): bogus"
    )

    args = parse_args(
        session_cmd, ["deploy", "--dist-target", "bogus", "target-session"]
    )
    with pytest.raises(ConfigurationError, match="Unknown deployment dist target"):
        session_cmd.handle(benchplot_user_config, args)


def test_clean_flags(session_cmd, mocker, benchplot_user_config):
    """
    `session clean` preserves the deploy tree by default; `--deploy` also
    removes it.
    """
    session = mocker.Mock()
    mocker.patch.object(SessionSubCommand, "_get_session", return_value=session)

    args = parse_args(session_cmd, ["clean", "target-session"])
    session_cmd.handle(benchplot_user_config, args)
    session.clean_all.assert_called_once_with(clean_deploy=False)

    session.clean_all.reset_mock()
    args = parse_args(session_cmd, ["clean", "--deploy", "target-session"])
    session_cmd.handle(benchplot_user_config, args)
    session.clean_all.assert_called_once_with(clean_deploy=True)


def test_push_flags_bundle_by_default(session_cmd, mocker, benchplot_user_config):
    """
    `session push` without `--image` bundles the session and pushes the
    bundle; dist artefacts are not pushed.
    """
    session = mocker.Mock()
    mocker.patch.object(SessionSubCommand, "_get_session", return_value=session)

    args = parse_args(session_cmd, ["push", "target-session", "fakehost:"])
    session_cmd.handle(benchplot_user_config, args)

    session.bundle.assert_called_once_with(path=None)
    session.push.assert_called_once_with("fakehost:", session.bundle.return_value)
    session.push_images.assert_not_called()

    session.bundle.reset_mock()
    session.push.reset_mock()
    args = parse_args(
        session_cmd, ["push", "-o", "bundles", "target-session", "fakehost:"]
    )
    session_cmd.handle(benchplot_user_config, args)
    session.bundle.assert_called_once_with(path=args.output)
    session.push.assert_called_once_with("fakehost:", session.bundle.return_value)


def test_push_flags_image(session_cmd, mocker, benchplot_user_config):
    """
    `session push --image` pushes assembled dist artefacts instead of the
    bundle: all artefacts when no DIST_ID is given, or the selected ones.
    """
    session = mocker.Mock()
    mocker.patch.object(SessionSubCommand, "_get_session", return_value=session)

    # Bare --image pushes every assembled artefact
    args = parse_args(session_cmd, ["push", "target-session", "fakehost:", "--image"])
    session_cmd.handle(benchplot_user_config, args)
    session.push_images.assert_called_once_with("fakehost:", images=None)
    session.bundle.assert_not_called()

    # --image <ids>... pushes only the selected artefacts; with nargs="*" the
    # ids must follow the positionals.
    session.push_images.reset_mock()
    args = parse_args(
        session_cmd,
        ["push", "target-session", "fakehost:", "--image", "img-one", "img-two"],
    )
    session_cmd.handle(benchplot_user_config, args)
    session.push_images.assert_called_once_with(
        "fakehost:", images=["img-one", "img-two"]
    )
