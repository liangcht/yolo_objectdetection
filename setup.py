#!/usr/bin/python

from __future__ import print_function
import os
import sys
import os.path as op
from setuptools import find_packages, setup
import numpy as np
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# change directory to this module path
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)
if op.dirname(this_file):
    os.chdir(op.dirname(this_file))
script_dir = os.getcwd()

include_dirs = [op.abspath('./mtorch/common/')]
# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def readme(fname):
    """Read text out of a file in the same directory as setup.py.
    """
    return open(op.join(script_dir, fname)).read()


setup(
    name="Microsoft Massive Object Detection",
    version="0.0.2",
    author="ehazar",
    author_email="ehazar@microsoft.com",
    url='',
    description="Microsoft Massive Object Detection (MMOD)",
    long_description=readme('README.md'),
    packages=find_packages(),
    ext_modules=[
        Extension(
            "mtorch.regionloss_utils.cython_bbox",
            ["mtorch/regionloss_utils/bbox.pyx"],
            extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
            include_dirs=[numpy_include]
        ),
        CUDAExtension('region_target_cuda', [
            'mtorch/rt/rt_cuda.cpp',
            'mtorch/rt/rt_cuda_kernel.cu',
        ], include_dirs=include_dirs),
        CppExtension('region_target_cpu', [
            'mtorch/rt/rt_cpu.cpp',
        ], include_dirs=include_dirs),
        CUDAExtension('smt_cuda', [
            'mtorch/smt/smt_cuda.cpp',
            'mtorch/smt/smt_cuda_kernel.cu',
        ], include_dirs=include_dirs),
        CUDAExtension('smtl_cuda', [
            'mtorch/smtl/smtl_cuda.cpp',
            'mtorch/smtl/smtl_cuda_kernel.cu',
        ], include_dirs=include_dirs),
        CppExtension('smtl_cpu', [
            'mtorch/smtl/smtl_cpu.cpp',
        ], include_dirs=include_dirs),
        CppExtension('smt_cpu', [
            'mtorch/smt/smt_cpu.cpp',
        ], include_dirs=include_dirs),
        CUDAExtension('nmsfilt_cuda', [
            'mtorch/nmsfilt/nmsfilt_cuda.cpp',
            'mtorch/nmsfilt/nmsfilt_cuda_kernel.cu',
        ], include_dirs=include_dirs),
        CppExtension('nmsfilt_cpu', [
            'mtorch/nmsfilt/nmsfilt_cpu.cpp',
        ], include_dirs=include_dirs),
        CUDAExtension('smtpred_cuda', [
            'mtorch/smtpred/smtpred_cuda.cpp',
            'mtorch/smtpred/smtpred_cuda_kernel.cu',
        ], include_dirs=include_dirs),
        CppExtension('smtpred_cpu', [
            'mtorch/smtpred/smtpred_cpu.cpp',
        ], include_dirs=include_dirs),
        CppExtension('darkresize', [
            'mtorch/darkresize/dark_resize.cpp',
        ], include_dirs=include_dirs),

    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
    license="BSD",
    classifiers=[
        'Intended Audience :: Developers',
        "Programming Language :: Python",
        'Topic :: Software Development',
    ]
)
