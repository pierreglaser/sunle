#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = "Simulation Based Inference using Energy Based Models"

dist = setup(
    name="unle",
    version="0.0.1dev0",
    description=description,
    license="BSD 3-Clause License",
    packages=["unle"],
    install_requires=[
        "numpy",
        "scipy",
        # Jax 0.4.11 removed some modules used by `numpyro`.
        # TODO: Remove this once `numpyro` is updated.
        # https://jax.readthedocs.io/en/latest/changelog.html#jax-0-4-11-may-31-2023
        "jax<0.4.11",
        "jaxlib<0.4.11",
        "flax",
        "numpyro",
        "optax",
        "blackjax",
        "typing_extensions",
        "cloudpickle",
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
