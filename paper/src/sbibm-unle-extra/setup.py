#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = "Utilities and Tasks for benchmarking (S)UNLE using `sbibm`"


dist = setup(
    name="sbibm_unle_extra",
    version="0.0.1dev0",
    description=description,
    license="BSD 3-Clause License",
    packages=["sbibm_unle_extra"],
    install_requires=[
        # "sbibm",
        # currently, the unle dependency is not available on PyPI.
        # unle is only present locally, and thus needs to be installed manually
        # before hand. I ended up not using direct references since they
        # don't seem to handle editable install correctly:
        # https://stackoverflow.com/questions/75290271/pyproject-toml-listing-an-editable-package-as-a-dependency-for-an-editable-packa
        "unle"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.9",
)
