from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='smtl_cuda',
    ext_modules=[
        CUDAExtension('smtl_cuda', [
            'smtl_cuda.cpp',
            'smtl_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False
)
