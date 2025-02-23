from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='pointpillars',
    ext_modules=[
        CppExtension(
            name='voxel_op', 
            sources=['voxelization/voxelization.cpp',
                     'voxelization/voxelization_cpu.cpp',
                     # 'voxelization/voxelization_cuda.cu',
                    ],
            define_macros=[('WITH_CUDA', None)]    
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })