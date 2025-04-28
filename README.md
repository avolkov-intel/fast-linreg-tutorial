### Instructions

1) Install any c++ compiler (for example, g++)

```
sudo apt update
sudo apt install g++
```

2) Install OpenBLAS library

```
sudo apt update
sudo apt install libopenblas-dev
```

3) Install Python dependencies

```
pip install numpy scikit-learn Cython setuptools pybind11 
```

4) Build the library

```
python setup.py build_ext --inplace
```

5) Run benchmarks

```
python benchmarks.py
```