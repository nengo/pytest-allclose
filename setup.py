#!/usr/bin/env python

# Automatically generated by nengo-bones, do not edit this file directly

import io
import os
import runpy

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version = runpy.run_path(os.path.join(root, "pytest_allclose", "version.py"))["version"]

install_req = [
    "numpy>=1.11",
    "pytest",
]
docs_req = [
    "nbsphinx>=0.6.0",
    "nengo_sphinx_theme>1.2.2",
    "numpydoc>=0.9.2",
    "sphinx",
]
optional_req = []
tests_req = [
    "codespell",
    "coverage>=4.3",
    "flake8",
    "gitlint",
    "pylint",
]

setup(
    name="pytest-allclose",
    version=version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    url="https://www.nengo.ai/pytest-allclose",
    include_package_data=False,
    license="MIT license",
    description="Pytest fixture extending Numpy's allclose function",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    install_requires=install_req,
    extras_require={
        "all": docs_req + optional_req + tests_req,
        "docs": docs_req,
        "optional": optional_req,
        "tests": tests_req,
    },
    python_requires=">=3.5",
    entry_points={"pytest11": ["allclose = pytest_allclose.plugin",],},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Framework :: Pytest",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
