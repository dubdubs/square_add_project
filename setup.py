from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='square_add',
    ext_modules=[
        CUDAExtension(
            name='square_add',
            sources=[
                'square_add/square_add_cpu.cpp',
                'square_add/square_add_cuda.cu',
                'square_add/bindings.cpp'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)