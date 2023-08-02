#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

dist = setup(
    name='smnle',
    version='0.0.0dev0',
    description='Extended pickling support for Python objects',
    license='BSD 3-Clause License',
    packages=["smnle", "smnle.src"],
    long_description='Score Matched Neural Likelihood Estimation',
    long_description_content_type="text/markdown",
    install_requires=[
        # currently, the unle dependency is not available on PyPI.
        # unle is only present locally, and thus needs to be installed manually
        # before hand. I ended up not using direct references since they
        # don't seem to handle editable install correctly:
        # https://stackoverflow.com/questions/75290271/pyproject-toml-listing-an-editable-package-as-a-dependency-for-an-editable-packa
        "unle"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Distributed Computing',
    ],
    test_suite='tests',
    python_requires='>=3.6',
)
