from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="zero_overhead_pinned_memory",
    version="0.0.0",
    author="Your Name",
    description="Zero Overhead Pinned Memory extension for PyTorch",
    packages=find_packages(where="src"),
    package_dir={"": "src"},             
    # Tells setuptools that packages are under src/
    ext_modules=[
        CUDAExtension(
            # Install the extension as a submodule: zero_overhead_pinned_memory.cuda_mem_manager
            name="zero_overhead_pinned_memory.cuda_mem_manager",
            sources=["src/zero_overhead_pinned_memory/cuda_mem_manager.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
