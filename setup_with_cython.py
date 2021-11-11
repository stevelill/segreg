"""
Build the ``segreg`` project.

Runs cython followed by C build.  As such, has dependency on ``cython``.  This
setup is intended to be run by the developer.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy


extensions = [
    Extension("segreg.model.regression",
              ["segreg/model/regression.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("segreg.model.one_bkpt_segreg",
              ["segreg/model/one_bkpt_segreg.pyx"],
              include_dirs=[numpy.get_include()],
              language="c++"),
    Extension("segreg.model.two_bkpt_segreg",
              ["segreg/model/two_bkpt_segreg.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    name="segreg",
    ext_modules=cythonize(extensions,
                          language_level=3,
                          compiler_directives={'embedsignature': True})
)
