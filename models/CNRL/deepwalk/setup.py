from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "My hello app",
    ext_modules =cythonize('word2vec_inner.pyx'),  # accepts a glob pattern
    include_dirs=[numpy.get_include()]
)