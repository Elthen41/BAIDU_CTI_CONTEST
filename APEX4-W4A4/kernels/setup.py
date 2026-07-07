from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name='APEX4',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='APEX4._CUDA',
            sources=[
                'csrc/pybind.cpp',
                'csrc/apex4_gemm_w4a4_group.cu',    # W4A4 per-group, half
                'csrc/apex4_gemm_w4a4_channel.cu',  # W4A4 per-channel, half
                'csrc/quantize_kernel.cu',          # fused activation quant + 4-bit pack
            ],
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(exclude=['notebook', 'scripts', 'tests']),
)
