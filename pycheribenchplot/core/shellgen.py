from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from threading import Lock
from typing import IO

from jinja2 import Environment, PackageLoader, TemplateNotFound, select_autoescape

from .config import PlatformArch, PlatformABI

type Benchmark = "Benchmark"


@dataclass
class ScriptHook:
    #: Hook human-readable name
    name: str
    #: Hook template
    template: Path | None = None
    #: Hook command, if the template is None
    commands: list[str] | None = None


ENV = Environment(
    loader=PackageLoader("pycheribenchplot", "templates"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def script_filter(fn):
    ENV.filters[fn.__name__] = fn
    return fn


@script_filter
def path_parent(value: Path | list[Path]) -> Path | list[Path]:
    """Helper to find the parent path within the template"""
    if isinstance(value, Path):
        return value.parent
    else:
        return [p.parent for p in value]


@script_filter
def path_name(value: Path | list[Path]) -> str | list[str]:
    """Helper to find the path name portion within the template"""
    if isinstance(value, Path):
        return value.name
    else:
        return [p.name for p in value]


@script_filter
def path_with_suffix(path: Path, suffix: str) -> Path:
    """Helper to replace the suffix of a path"""
    return path.with_suffix(suffix)


class TemplateContextBase:
    """
    Generates a script based on a Jinja2 template and parameters.

    The context maintains the set of parameters passed to the template.
    """

    DEFAULT_TEMPLATE = "must-be-set-explicitly"

    def __init__(self, logger: Logger):
        self.logger = logger
        #: Script context lock
        self._lock = Lock()
        #: Template file to use
        self._template = self.DEFAULT_TEMPLATE
        #: Template context
        self._context = {}

    def set_template(self, new_template: str):
        with self._lock:
            if self._template != self.DEFAULT_TEMPLATE:
                self.logger.warning(
                    "Multiple changes to the default script template "
                    "script, this may result in unexpected behaviour."
                )
            self._template = new_template

    def register_global(self, name, obj):
        with self._lock:
            if name in self._context:
                self.logger.warning("Overriding global %s with %s", name, obj)
            self._context[name] = obj

    def extend_context(self, ctxdict: dict[str, any]):
        with self._lock:
            for name, value in ctxdict.items():
                existing = self._context.get(name)
                if existing:
                    self.logger.warning(
                        "Preventing override of %s with value %s", name, existing
                    )
                    continue
                self._context[name] = value

    @property
    def context(self) -> dict[str, any]:
        return self._context

    def render(self, fd: IO[str]):
        try:
            tmpl = ENV.get_template(self._template)
        except TemplateNotFound:
            self.logger.error(
                "Can not find file template %s, target setup is wrong", self._template
            )
            raise RuntimeError("Target error")

        script = tmpl.render(**self._context)

        # def lstrip_spaces(line):
        #     # Keep anything that starts with a tab
        #     if line.startswith(" "):
        #         line = line.lstrip(" ")
        #     return line + "\n"

        # fd.writelines(map(lstrip_spaces, script.splitlines()))
        fd.write(script)


class ScriptContext(TemplateContextBase):
    """
    Generates a benchmark runner script from a template.

    The context maintains the data that is passed to the template upon rendering.
    These include the key/value parameters for the benchmark parameterization,
    as well as the sets of benchmark-specific commands to populate the hook functions.

    The main template is designed to be extended by benchmark-specific templates that
    determine how to run the benchmark.
    """

    DEFAULT_TEMPLATE = "runner-script.sh.jinja"

    def __init__(self, benchmark: Benchmark):
        super().__init__(benchmark.logger)
        #: Template context
        self._context = {
            "dataset_id": benchmark.uuid,
            "iterations": benchmark.config.iterations,
            "instance": benchmark.config.instance,
            "parameters": benchmark.parameters,
            "setup_hooks": [],
            "teardown_hooks": [],
            "iter_setup_hooks": [],
            "iter_teardown_hooks": [],
            "iter_exec_hooks": [],
        }

        # Add config enum types to the context
        self.register_global("PlatformABI", PlatformABI)
        self.register_global("PlatformArch", PlatformArch)

    def add_hook(self, phase: str, hook: ScriptHook):
        with self._lock:
            self._context[f"{phase}_hooks"].append(hook)

    def setup(self, name, *, commands: str = None, template: Path = None):
        assert commands is None or template is None
        self.add_hook("setup", ScriptHook(name, template=template, commands=commands))

    def teardown(self, name, *, commands: str = None, template: Path = None):
        assert commands is None or template is None
        self.add_hook(
            "teardown", ScriptHook(name, template=template, commands=commands)
        )

    def setup_iteration(self, name, *, commands: str = None, template: Path = None):
        assert commands is None or template is None
        self.add_hook(
            "iter_setup", ScriptHook(name, template=template, commands=commands)
        )

    def teardown_iteration(self, name, *, commands: str = None, template: Path = None):
        assert commands is None or template is None
        self.add_hook(
            "iter_teardown", ScriptHook(name, template=template, commands=commands)
        )

    def exec_iteration(self, name, *, commands: str = None, template: Path = None):
        assert commands is None or template is None
        self.add_hook(
            "iter_exec", ScriptHook(name, template=template, commands=commands)
        )
