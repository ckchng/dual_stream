from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='deep_hough_v2',
    ext_modules=[
        CUDAExtension('deep_hough_v2', [
            'deep_hough_cuda.cpp',
            'deep_hough_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-arch=sm_60']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
