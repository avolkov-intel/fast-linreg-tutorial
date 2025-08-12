from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import pybind11
import numpy
import sys

# OpenMP linkage
# Note: different platforms and compilers require different arguments for OMP linkage.
# For example, on windows with the MSVC compiler, it requires argument '/openmp',
# while on macOS it might require '-Xclang -fopenmp -lomp' (plus additional external
# installations of the OpenMP runtime library), and under some clang versions on
# linux, might require additional linkage to 'libomp'.
# For simplicity, this skips linkage to OpenMP on platforms other than windows and linux
# (i.e. will not use OpenMP on macOS). Note that compiling on windows with the mingw
# compiler requires changing the argument from '/openmp' to '-fopenmp'.
args_openmp = []
if sys.platform == "linux":
    args_openmp = ["-fopenmp"]
elif sys.platform == "win32":
    args_openmp = ["/openmp"]

# === 1. Pybind11 Extension ===
pybind11_ext = Extension(
    "utils_pybind",
    ["utils_pybind.cpp"],
    include_dirs=[
        pybind11.get_include(),
        numpy.get_include(),
        ".",
    ],
    extra_compile_args=args_openmp,
    extra_link_args=args_openmp,
    language="c++"
)

# === 2. Cython Extension ===
cython_exts = cythonize([
    Extension(
        "utils_cython",
        ["utils_cython.pyx"],
        include_dirs=[
            numpy.get_include()
        ],
        extra_compile_args=args_openmp,
        extra_link_args=args_openmp,
        language="c++"
    )
])

# === 3. Setup ===
setup(
    name="syrk_gemv_acceleration",
    version="0.1",
    description="Project combining pybind11 and Cython extensions",
    ext_modules=[pybind11_ext] + cython_exts,
    zip_safe=False,
)
