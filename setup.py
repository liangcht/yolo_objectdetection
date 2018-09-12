#!/usr/bin/python

from __future__ import print_function
import os
import sys
import os.path as op
from setuptools import find_packages, setup

# change directory to this module path
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)
if op.dirname(this_file):
    os.chdir(op.dirname(this_file))
script_dir = os.getcwd()


def readme(fname):
    """Read text out of a file in the same directory as setup.py.
    """
    return open(op.join(script_dir, fname)).read()


setup(
    name="Microsoft Massive Object Detection",
    version="0.0.1",
    author="ehazar",
    author_email="ehazar@microsoft.com",
    url='',
    description="Microsoft Object Detection",
    long_description=readme('README.md'),
    packages=find_packages(),
    license="BSD",
    classifiers=[
        'Intended Audience :: Developers',
        "Programming Language :: Python",
        'Topic :: Software Development',
    ]
)
