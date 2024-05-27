from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("loc_lin_est.pyx"),
    include_dirs=[numpy.get_include()]
    )

