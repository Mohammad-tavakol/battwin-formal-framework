"""Per-call output routing, isolated across independent sessions and threads."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_directory = ContextVar("battwin_plot_directory", default=None)


@contextmanager
def output_directory(directory):
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = _directory.set(root)
    try:
        yield root
    finally:
        _directory.reset(token)


def result_path(relative):
    root = _directory.get()
    if root is None:
        raise RuntimeError("Use a public plot function with an explicit directory.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Plot output must stay within its output directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_table(frame, path, index=False):
    destination = result_path(path)
    frame.to_csv(destination, index=index)
    return destination
