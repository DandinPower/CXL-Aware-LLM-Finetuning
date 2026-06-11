import platform
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


PROJECT_ROOT = Path(__file__).resolve().parent


def get_cpu_flags():
    if platform.system() != "Linux":
        raise RuntimeError(
            f"cpu_lion package build currently supports Linux only, got {platform.system()}."
        )

    with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo_file:
        cpuinfo = cpuinfo_file.read()

    for line in cpuinfo.splitlines():
        if line.startswith("flags"):
            return line.split(":", maxsplit=1)[1].strip().split()
    return []


def ensure_avx512_build_support():
    if "avx512f" not in get_cpu_flags():
        raise RuntimeError(
            "The cpu_lion extension requires AVX512 support. "
            "Current machine did not report the avx512f flag."
        )


ensure_avx512_build_support()


setup(
    name="cpu_lion",
    version="0.0.0",
    author="Joseph Liaw",
    description="Standalone CPU Lion optimizer package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CppExtension(
            name="cpu_lion.cpu_lion_interface",
            sources=[
                str(PROJECT_ROOT / "src" / "cpu_lion" / "csrc" / "lion_interface.cpp"),
                str(PROJECT_ROOT / "src" / "cpu_lion" / "csrc" / "lion_impl.cpp"),
            ],
            include_dirs=[str(PROJECT_ROOT / "src" / "cpu_lion" / "includes")],
            extra_compile_args=["-fopenmp", "-march=native", "-D__AVX512__"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
