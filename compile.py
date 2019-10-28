from distutils.core import setup
from Cython.Build import cythonize
import numpy
from setuptools import Extension

setup(
    name='topological',
    ext_modules=cythonize(
        Extension(
            "topological",
            sources=["topological.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ),
    install_requires=["numpy"]
)
