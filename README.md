### Description

This repository contains the materials for tutorial 
proposal. Some of the provided code will be intentionally
removed, allowing tutorial attendees to implement 
missing parts and learn how to use functions written in 
Cython and C++ within their Python libraries.

- `utils.py` - contains implementation of python baselines and auxilarity functions
- `utils_cython.pyx` - contains implementation of $X^tX$ and $X^ty$ functions in Cython
- `utils_pybind.cpp` - contains implementations of $X^tX$ and $X^ty$ functions with and without OpenBLAS library usage
- `blas_helpers.h` - contains helpers to load required BLAS functions from SciPy when used outside of Cython
- `benchmarks.py` - contains benchmarks to compare performance of different methods
- `setup.py` - script used to build the modules

### Setup Instructions

1) Install conda/mamba through the [miniforge installer](https://github.com/conda-forge/miniforge):

    * Windows:
    ```shell
    start /wait "" Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniforge3
    ```

    * Linux and macOS:
    ```shell
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    ```

2) Create an environment with Python, compiler toolchain, and PyData stack:

```shell
mamba create -n linreg_env -c conda-forge python numpy scipy cython pybind11 setuptools scikit-learn blas[build=openblas] cxx-compiler

```

3) Build the extension modules for this tutorial

```
python setup.py build_ext --inplace
```

4) Runtime benchmarks

```
python benchmarks.py
```

#### Example:

![alt text](example_output.png)
