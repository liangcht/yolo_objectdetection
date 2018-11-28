import sys
import os
import os.path as op
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# change directory to this module path
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)
if op.dirname(this_file):
    os.chdir(op.dirname(this_file))

include_dirs = [op.abspath('../common/')]

setup(
    name='smtl',
    ext_modules=[
        CUDAExtension('smtl_cuda', [
            'smtl_cuda.cpp',
            'smtl_cuda_kernel.cu',
        ], include_dirs=include_dirs),
        CppExtension('smtl_cpu', [
            'smtl_cpu.cpp',
        ], include_dirs=include_dirs),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False
)
