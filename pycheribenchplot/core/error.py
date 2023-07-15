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
