#!/usr/bin/env python

import argparse
from urllib import request
import re

import pandas as pd
from packaging.specifiers import SpecifierSet
from packaging.version import Version

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
}


def get_min_python_version():
    """
    Find the minimum version of Python supported (i.e., not end-of-life)
    """
    min_python = (
        pd.read_html(
            "https://devguide.python.org/versions/",
            storage_options=HEADERS,
        )[0]
        .iloc[-1]
        .Branch
    )
    return min_python


def get_min_numba_numpy_version(min_python):
    """
    Find the minimum versions of Numba and NumPy that supports the specified
    `min_python` version
    """
    df = (
        pd.read_html(
            "https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information",  # noqa
            storage_options=HEADERS,
        )[0]
        .dropna()
        .drop(columns=["Numba.1", "llvmlite", "LLVM", "TBB"])
        .query('`Python`.str.contains("2.7") == False')
        .query('`Numba`.str.contains(".x") == False')
        .query('`Numba`.str.contains("{") == False')
        .pipe(
            lambda df: df.assign(
                MIN_PYTHON_SPEC=(
                    df.Python.str.split().str[1].replace({"<": "="}, regex=True)
                    + df.Python.str.split().str[0].replace({".x": ""}, regex=True)
                ).apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MIN_NUMPY=(df.NumPy.str.split().str[0].replace({".x": ""}, regex=True))
            )
        )
        .assign(
            COMPATIBLE=lambda row: row.apply(
                check_python_compatibility, axis=1, args=(Version(min_python),)
            )
        )
        .query("COMPATIBLE == True")
        .pipe(lambda df: df.assign(MINOR=df.Numba.str.split(".").str[1]))
        .pipe(lambda df: df.assign(PATCH=df.Numba.str.split(".").str[2]))
        .sort_values(["MINOR", "PATCH"], ascending=[False, True])
        .iloc[-1]
    )
    return df.Numba, df.MIN_NUMPY


def check_python_compatibility(row, min_python):
    """
    Determine the Python version compatibility
    """
    python_compatible = min_python in (row.MIN_PYTHON_SPEC)
    return python_compatible


def check_scipy_compatibility(row, min_python, min_numpy):
    """
    Determine the Python and NumPy version compatibility
    """
    python_compatible = min_python in (row.MIN_PYTHON_SPEC & row.MAX_PYTHON_SPEC)
    numpy_compatible = min_numpy in (row.MIN_NUMPY_SPEC & row.MAX_NUMPY_SPEC)
    return python_compatible & numpy_compatible


def get_scipy_version_df():
    """
    Retrieve raw SciPy version table as DataFrame
    """
    colnames = pd.read_html(
        "https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy",
        storage_options=HEADERS,
    )[1].columns
    converter = {colname: str for colname in colnames}
    return (
        pd.read_html(
            "https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy",
            storage_options=HEADERS,
            converters=converter,
        )[1]
        .rename(columns=lambda x: x.replace(" ", "_"))
        .replace({".x": ""}, regex=True)
    )


def get_min_scipy_version(min_python, min_numpy):
    """
    Determine the SciPy version compatibility
    """
    df = (
        get_scipy_version_df()
        .pipe(
            lambda df: df.assign(
                SciPy_version=df.SciPy_version.str.replace(
                    r"\d\/", "", regex=True  # noqa
                )
            )
        )
        .query('`Python_versions`.str.contains("2.7") == False')
        .pipe(
            lambda df: df.assign(
                MIN_PYTHON_SPEC=df.Python_versions.str.split(",")
                .str[0]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MAX_PYTHON_SPEC=df.Python_versions.str.split(",")
                .str[1]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MIN_NUMPY_SPEC=df.NumPy_versions.str.split(",")
                .str[0]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MAX_NUMPY_SPEC=df.NumPy_versions.str.split(",")
                .str[1]
                .apply(SpecifierSet)
            )
        )
        .assign(
            COMPATIBLE=lambda row: row.apply(
                check_scipy_compatibility,
                axis=1,
                args=(Version(min_python), Version(min_numpy)),
            )
        )
        .query("COMPATIBLE == True")
        .pipe(lambda df: df.assign(MINOR=df.SciPy_version.str.split(".").str[1]))
        .pipe(lambda df: df.assign(PATCH=df.SciPy_version.str.split(".").str[2]))
        .sort_values(["MINOR", "PATCH"], ascending=[False, True])
        .iloc[-1]
    )
    return df.SciPy_version


def get_minor_versions_between(start_version_str, end_version_str):
    """
    Returns a list of all minor Python versions between two specified minor versions.
    Assumes semantic versioning (MAJOR.MINOR.PATCH) and only considers minor versions
    within the same major version.

    Args:
        start_version_str (str): The starting version string (e.g., "3.6.0").
        end_version_str (str): The ending version string (e.g., "3.9.5").

    Returns:
        list: A list of strings representing the minor versions in between,
              including the start and end minor versions if they are distinct.
              Returns an empty list if the start version is greater than or equal
              to the end version, or if major versions differ.
    """
    try:
        start_parts = [int(x) for x in start_version_str.split('.')]
        end_parts = [int(x) for x in end_version_str.split('.')]
    except ValueError:
        raise ValueError("Invalid version string format. Expected 'MAJOR.MINOR.PATCH'.")

    if len(start_parts) < 2 or len(end_parts) < 2:
        raise ValueError("Version string must include at least major and minor parts.")

    start_major, start_minor = start_parts[0], start_parts[1]
    end_major, end_minor = end_parts[0], end_parts[1]

    if start_major != end_major:
        print("Warning: Major versions differ. Returning an empty list.")
        return []

    if start_minor >= end_minor:
        print("Warning: Start minor version is not less than end minor version. Returning an empty list.")
        return []

    versions = []
    for minor in range(start_minor, end_minor + 1):
        versions.append(f"{start_major}.{minor}")

    return versions


def get_latest_numpy_version():
    url = "https://pypi.org/project/numpy/"
    req = request.Request(
            url,
            data=None,
            headers=HEADERS
        )
    html = request.urlopen(req).read().decode("utf-8")
    match = re.search(r'numpy (\d+\.\d+\.\d+)', html, re.DOTALL)
    return match.groups()[0]


def check_python_version(row):
    versions = get_minor_versions_between(row.START_PYTHON_VERSION, row.END_PYTHON_VERSION)

    compatible_version = None
    for version in versions:
        if Version(version) in row.NUMBA_PYTHON_SPEC & row.SCIPY_PYTHON_SPEC:
            compatible_version = version
    return compatible_version

def check_numpy_version(row):
    if row.NUMPY in row.NUMPY_SPEC:
        return row.NUMPY
    else:
        return None


def get_all_max_versions():
    """
    Find the maximum version of Python that is compatible with Numba and NumPy
    """
    df = (
        pd.read_html(
            "https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information",  # noqa
            storage_options=HEADERS,
        )[0]
        .dropna()
        .drop(columns=["Numba.1", "llvmlite", "LLVM", "TBB"])
        .query('`Python`.str.contains("2.7") == False')
        .query('`Numba`.str.contains(".x") == False')
        .query('`Numba`.str.contains("{") == False')
        .pipe(
            lambda df: df.assign(
                START_PYTHON_VERSION=(
                    df.Python.str.split().str[0].replace({".x": ""}, regex=True)
                )
            )
        )
        .pipe(
            lambda df: df.assign(
                END_PYTHON_VERSION=(
                    df.Python.str.split().str[4].replace({".x": ""}, regex=True)
                )
            )
        )
        .pipe(
            lambda df: df.assign(
                NUMBA_PYTHON_SPEC=(
                    df.Python.str.split().str[1].replace({"<": ">"}, regex=True)
                    + df.Python.str.split().str[0].replace({".x": ""}, regex=True)
                    + ", "
                    + df.Python.str.split().str[3]
                    + df.Python.str.split().str[4].replace({".x": ""}, regex=True)
                ).apply(SpecifierSet)
            )
        )
        .assign(
            NUMPY = get_latest_numpy_version()
        )
        .pipe(
            lambda df: df.assign(
                NUMPY_SPEC=(
                    df.NumPy
                    .str.replace(r' [;†\.]$', '', regex=True)
                    .str.split().str[-2:].replace({".x": ""}, regex=True)
                    .str.join('')
                ).apply(SpecifierSet)
            )
        )
        .assign(
            NUMPY=lambda row: row.apply(
                check_numpy_version, axis=1
            )
        )
        .assign(
            SCIPY = get_scipy_version_df().iloc[0].SciPy_version
        )
        .assign(
            SCIPY_PYTHON_SPEC = get_scipy_version_df().iloc[0].Python_versions
        )
        .pipe(
            lambda df:
                df.assign(
                    SCIPY_PYTHON_SPEC = df.SCIPY_PYTHON_SPEC.apply(SpecifierSet)
            )
        )
        .assign(
            MAX_PYTHON=lambda row: row.apply(
                check_python_version, axis=1
            )
        )
        .pipe(lambda df: df.assign(MAJOR=df.MAX_PYTHON.str.split(".").str[0]))
        .pipe(lambda df: df.assign(MINOR=df.MAX_PYTHON.str.split(".").str[1]))
        .sort_values(["MAJOR", "MINOR"], ascending=[False, False])
        .iloc[0]
    )

    print(
        f"python: {df.MAX_PYTHON}\n"
        f"numba: {df.Numba}\n"
        f"numpy: {df.NUMPY}\n"
        f"scipy: {df.SCIPY}"
    )





def match_pkg_version(line, pkg_name):
    """
    Regular expression to match package versions
    """
    matches = re.search(
        rf"""
                        {pkg_name}  # Package name
                        [\s=><:"\'\[\]]*  # Zero or more spaces or special characters
                        (\d+\.\d+[\.0-9]*)  # Capture "version" in `matches`
                        """,
        line,
        re.VERBOSE | re.IGNORECASE,  # Ignores all whitespace and case in pattern
    )

    return matches


def find_pkg_mismatches(pkg_name, pkg_version, fnames):
    """
    Determine if any package version has mismatches
    """
    pkg_mismatches = []

    for fname in fnames:
        with open(fname, "r") as file:
            for line_num, line in enumerate(file, start=1):
                l = line.strip().replace(" ", "").lower()
                matches = match_pkg_version(l, pkg_name)
                if matches is not None:
                    version = matches.groups()[0]
                    if version != pkg_version:
                        pkg_mismatches.append((pkg_name, version, fname, line_num))

    return pkg_mismatches


def test_pkg_mismatch_regex():
    """
    Validation function for the package mismatch regex
    """
    pkgs = {
        "numpy": "0.0",
        "scipy": "0.0",
        "python": "2.7",
        "python-version": "2.7",
        "numba": "0.0",
    }

    lines = [
        "Programming Language :: Python :: 3.8",
        "STUMPY supports Python 3.8",
        "python-version: ['3.8']",
        'requires-python = ">=3.8"',
        "numba>=0.55.2",
    ]

    for line in lines:
        match_found = False
        for pkg_name, pkg_version in pkgs.items():
            matches = match_pkg_version(line, pkg_name)

            if matches:
                match_found = True
                break

        if not match_found:
            raise ValueError(f'Package mismatch regex fails to cover/match "{line}"')


def get_all_min_versions(MIN_PYTHON):
    MIN_NUMBA, MIN_NUMPY = get_min_numba_numpy_version(MIN_PYTHON)
    MIN_SCIPY = get_min_scipy_version(MIN_PYTHON, MIN_NUMPY)

    print(
        f"python: {MIN_PYTHON}\n"
        f"numba: {MIN_NUMBA}\n"
        f"numpy: {MIN_NUMPY}\n"
        f"scipy: {MIN_SCIPY}"
    )

    pkgs = {
        "numpy": MIN_NUMPY,
        "scipy": MIN_SCIPY,
        "numba": MIN_NUMBA,
        "python": MIN_PYTHON,
        "python-version": MIN_PYTHON,
    }

    fnames = [
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        ".github/workflows/github-actions.yml",
        "README.rst",
    ]

    test_pkg_mismatch_regex()

    for pkg_name, pkg_version in pkgs.items():
        for name, version, fname, line_num in find_pkg_mismatches(
            pkg_name, pkg_version, fnames
        ):
            print(
                f"{pkg_name} {pkg_version} Mismatch: Version {version} "
                f"found in {fname}:{line_num}"
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, default='min', help='Options: ["min", "max"]')
    parser.add_argument("python_version", nargs="?", default=None)
    args = parser.parse_args()
    # Example
    # ./versions.py
    # ./versions.py 3.11
    # ./versions.py -mode max

    print(f'mode: {args.mode}')

    if args.mode == 'min':
        if args.python_version is not None:
            MIN_PYTHON = str(args.python_version)
        else:
            MIN_PYTHON = get_min_python_version()
        get_all_min_versions(MIN_PYTHON)
    elif args.mode == 'max':
        get_all_max_versions()
    else:
        raise ValueError(f'Unrecognized mode: "{args.mode}"')
    