from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import pybind11
import numpy
import sys
import os

# Detect OpenBLAS
openblas_include_dirs = []
openblas_library_dirs = []
openblas_libs = ["openblas"]

# Compiler flags
extra_compile_args = ["-O3", "-std=c++14"]

# === 1. Pybind11 Extension ===
pybind11_ext = Extension(
    "utils_pybind",
    ["utils_pybind.cpp"],
    include_dirs=[
        pybind11.get_include(),
        numpy.get_include(),
        *openblas_include_dirs
    ],
    libraries=openblas_libs,
    library_dirs=openblas_library_dirs,
    extra_compile_args=extra_compile_args,
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
        extra_compile_args=extra_compile_args,
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
