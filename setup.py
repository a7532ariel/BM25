from setuptools import setup
from Cython.Build import cythonize
setup(
    name="vsmmodel",
    ext_modules = cythonize('vsmmodel.pyx'))