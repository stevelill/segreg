"""
Build the ``segreg`` project.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from setuptools import setup
from setuptools.extension import Extension

import numpy


extensions = [
    Extension("segreg.model.regression",
              ["segreg/model/regression.c"],
              include_dirs=[numpy.get_include()]),
    Extension("segreg.model.one_bkpt_segreg",
              ["segreg/model/one_bkpt_segreg.cpp"],
              include_dirs=[numpy.get_include()],
              language="c++"),
    Extension("segreg.model.two_bkpt_segreg",
              ["segreg/model/two_bkpt_segreg.c"],
              include_dirs=[numpy.get_include()])
]

setup(
    name="segreg",
    ext_modules=extensions
)
