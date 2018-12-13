# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "cython_bbox",
        ["bbox.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs=[numpy_include]
    ),
]

setup(
    name='mot_utils',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': build_ext},
)
