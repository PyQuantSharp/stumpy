# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import ast
import importlib
import pathlib
import site
import warnings

CACHE_WARNING = "Caching `numba` functions is purely for experimental purposes "
CACHE_WARNING += "and should never be used or depended upon as it is not supported! "
CACHE_WARNING += "All caching capabilities are not tested and may be removed/changed "
CACHE_WARNING += "without prior notice. Please proceed with caution!"


def get_njit_funcs():
    """
    Identify all njit functions

    Parameters
    ----------
    None

    Returns
    -------
    out : list
        A list of (`module_name`, `func_name`) pairs
    """
    ignore_py_files = ["__init__", "__pycache__"]

    pkg_dir = pathlib.Path(__file__).parent
    module_names = []
    for fname in pkg_dir.iterdir():
        if fname.stem not in ignore_py_files and not fname.stem.startswith("."):
            module_names.append(fname.stem)

    njit_funcs = []
    for module_name in module_names:
        filepath = pkg_dir / f"{module_name}.py"
        file_contents = ""
        with open(filepath, encoding="utf8") as f:
            file_contents = f.read()
        module = ast.parse(file_contents)
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                for decorator in node.decorator_list:
                    decorator_name = None
                    if isinstance(decorator, ast.Name):
                        # Bare decorator
                        decorator_name = decorator.id
                    if isinstance(decorator, ast.Call) and isinstance(
                        decorator.func, ast.Name
                    ):
                        # Decorator is a function
                        decorator_name = decorator.func.id

                    if decorator_name == "njit":
                        njit_funcs.append((module_name, func_name))

    return njit_funcs


def _enable():
    """
    Enable numba caching

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    warnings.warn(CACHE_WARNING)
    njit_funcs = get_njit_funcs()
    for module_name, func_name in njit_funcs:
        module = importlib.import_module(f".{module_name}", package="stumpy")
        func = getattr(module, func_name)
        func.enable_caching()


def _clear():
    """
    Clear numba cache

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    warnings.warn(CACHE_WARNING)
    site_pkg_dir = site.getsitepackages()[0]
    numba_cache_dir = site_pkg_dir + "/stumpy/__pycache__"
    [f.unlink() for f in pathlib.Path(numba_cache_dir).glob("*nb*") if f.is_file()]


def _get_cache():
    """
    Retrieve a list of cached numba functions

    Parameters
    ----------
    None

    Returns
    -------
    out : list
        A list of cached numba functions
    """
    warnings.warn(CACHE_WARNING)
    site_pkg_dir = site.getsitepackages()[0]
    numba_cache_dir = site_pkg_dir + "/stumpy/__pycache__"
    return [f.name for f in pathlib.Path(numba_cache_dir).glob("*nb*") if f.is_file()]
