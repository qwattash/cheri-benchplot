from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import IO

from jinja2 import (Environment, PackageLoader, TemplateNotFound, select_autoescape)

from .artefact import RemoteTarget, Target
from .util import new_logger


@dataclass
class ScriptHook:
    #: Hook human-readable name
    name: str
    #: Hook template
    template: Path | None = None
    #: Hook command, if the template is None
    commands: list[str] | None = None


class ScriptContext:
    """
    Generates a script based on a Jinja2 template and parameters.

    The context maintains the set of parameters passed to the template.
    These include the key/value parameters for the benchmark parameterization,
    as well as the sets of benchmark-specific commands to populate the hook functions.

    The main template
    """

    DEFAULT_TEMPLATE = "runner-script.sh.jinja"
    ENV = Environment(loader=PackageLoader("pycheribenchplot", "templates"),
                      autoescape=select_autoescape(),
                      trim_blocks=True,
                      lstrip_blocks=True)

    def __init__(self, benchmark: "Benchmark"):
        self.logger = benchmark.logger
        #: Script context lock
        self._lock = Lock()
        #: Template file to use
        self._template = self.DEFAULT_TEMPLATE
        #: Template context
        self._context = {
            "dataset_id": benchmark.uuid,
            "dataset_gid": benchmark.g_uuid,
            "iterations": benchmark.config.iterations,
            "instance": benchmark.config.instance,
            "parameters": benchmark.parameters,
            "setup_hooks": [],
            "teardown_hooks": [],
            "iter_setup_hooks": [],
            "iter_teardown_hooks": [],
        }

    def set_template(self, new_template: str):
        with self._lock:
            if self._template != self.DEFAULT_TEMPLATE:
                self.logger.warning("Multiple changes to the default benchmark template "
                                    "script, this may result in unexpected behaviour.")
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
                    self.logger.warning("Preventing override of %s with value %s", name, existing)
                    continue
                self._context[name] = value

    def render(self, fd: IO[str]):
        try:
            tmpl = self.ENV.get_template(self._template)
        except TemplateNotFound:
            self.logger.error("Can not find file template %s, target setup is wrong", self._template)
            raise RuntimeError("Target error")

        script = tmpl.render(**self._context)

        def lstrip_spaces(line):
            # Keep anything that starts with a tab
            if line.startswith(" "):
                line = line.lstrip(" ")
            return line + "\n"

        fd.writelines(map(lstrip_spaces, script.splitlines()))

    def add_hook(self, phase: str, hook: ScriptHook):
        with self._lock:
            self._context[f"{phase}_hooks"].append(hook)

    def setup(self, name, *, command: str = None, template: Path = None):
        assert command is None or template is None
        self.add_hook("setup", ScriptHook(name, template=template, command=command))

    def teardown(self, name, *, command: str = None, template: Path = None):
        assert command is None or template is None
        self.add_hook("teardown", ScriptHook(name, template=template, command=command))

    def setup_iteration(self, name, *, command: str = None, template: Path = None):
        assert command is None or template is None
        self.add_hook("iter_setup", ScriptHook(name, template=template, command=command))

    def teardown_iteration(self, name, *, command: str = None, template: Path = None):
        assert command is None or template is None
        self.add_hook("iter_teardown", ScriptHook(name, template=template, command=command))
