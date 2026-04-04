"""ADMM - Automatic Decomposition Method by MindOpt

ADMM is a Python library for building and solving structured optimization
models. Describe objectives and constraints in a natural mathematical style,
and ADMM handles canonicalization, decomposition, and solving automatically
through a high-performance C++ backend.

Example:
    >>> import admm
    >>> import numpy as np
    >>> model = admm.Model()
    >>> x = admm.Var("x", 2)
    >>> model.setObjective(x @ x)
    >>> model.optimize()
"""

import os
import platform

if platform.system() == "Windows":
    pkgs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    admmlib_path = os.path.join(pkgs_path, "admmlib", "lib")
    if os.path.isdir(admmlib_path):
        os.add_dll_directory(admmlib_path)

from .admm import *  # noqa: F401,F403

try:
    from importlib.metadata import version, metadata
    __version__ = version("admm")
    _meta = metadata("admm")
    __author__ = _meta.get("Author", "MindOpt Team")
    __email__ = _meta.get("Author-email", "")
except Exception:
    __version__ = "dev"
    __author__ = "MindOpt Team"
    __email__ = ""
