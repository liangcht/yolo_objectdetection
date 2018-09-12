from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='smt_cuda',
    ext_modules=[
        CUDAExtension('smt_cuda', [
            'smt_cuda.cpp',
            'smt_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False
)
