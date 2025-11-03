class ConfigurationError(RuntimeError):
    """
    Error during configuration parsing or manipulation.
    """
    pass


class ConfigTemplateBindError(ConfigurationError):
    """
    Failed to bind a configuration template field to a value
    """
    def __init__(self, msg):
        super().__init__(msg)

        self.location = ""


class TaskNotFound(RuntimeError):
    """
    Exception signaling that a given task could not be found.

    This may occur when looking up a task within a session or searching
    tasks in the main registry.
    """
    pass


class MissingDependency(RuntimeError):
    """
    Exception signaling that dependency resolution failed.

    This may happen if a non-optional dependency is not found.
    """
    pass


class ToolArgparseError(RuntimeError):
    """
    Exception signaling a failure in argument processing.

    This is used in the CLI/GUI tool helpers during argument handling.
    """
    pass
