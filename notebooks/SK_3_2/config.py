"""Configure the project."""

import os
from joblib import Memory

root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def path(x=None):
    """Construct a relative path."""
    global root_dir
    if x is None:
        return root_dir
    return os.path.join(root_dir, x)


memory = Memory(path('cache'))


def cache():
    return memory.cache()


def clear_cache():
    memory.clear()
