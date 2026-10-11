"""
Lazy package exports (PEP 562).

Package ``__init__`` modules used to import every submodule eagerly, so *any*
``import dreamer...`` — including the one a spawned pool worker does to unpickle
its task function — loaded the whole package and its heavy optional
dependencies (pandas, numba, networkx, scipy.optimize, python-mip, ...):
~240 MB per worker against ~120 MB for what a trajectory computation needs.

``lazy_exports`` gives a package a module-level ``__getattr__`` that imports an
exported name on first access, so ``from dreamer import System`` /
``dreamer.search.SearcherModV1`` keep working unchanged while importing one
submodule no longer imports the rest.  (Python 3.15's ``lazy import`` statement
would do the same, but the project supports Python >= 3.11.)
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Tuple


def lazy_exports(package: str, exports: Dict[str, str]) -> Tuple[Callable, Callable, List[str]]:
    """Build ``(__getattr__, __dir__, __all__)`` for a package.

    :param package: The package's ``__name__``.
    :param exports: ``{name: "module.path"}`` for a submodule, or
        ``{name: "module.path:attr"}`` for an attribute of a module.  Relative
        paths (starting with ``.``) are resolved against *package*.
    :return: The module-level ``__getattr__``, ``__dir__`` and ``__all__``.
    """
    def __getattr__(name: str):
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        mod_path, _, attr = target.partition(":")
        module = importlib.import_module(mod_path, package)
        value = getattr(module, attr) if attr else module
        import sys
        setattr(sys.modules[package], name, value)   # cache: next access is direct
        return value

    def __dir__() -> List[str]:
        import sys
        return sorted(set(vars(sys.modules[package])) | set(exports))

    return __getattr__, __dir__, sorted(exports)
