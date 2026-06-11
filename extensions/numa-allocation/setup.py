from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='numa_allocation',
    version='0.1.0',
    description='NUMA aware tensor allocation for PyTorch',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        CUDAExtension(
            name='numa_allocation.allocation',
            sources=['src/numa_allocation/allocation.cu'],
            extra_link_args=['-lnuma']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
