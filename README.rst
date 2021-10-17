segreg
======

**segreg** is a Python module for segmented regression.

Segmented regression models are defined as having breakpoints where the functional form
changes.  The code here treats the case of piecewise-linear and continuous univariate
models with at most two breakpoints.

This code fits segmented regression models to data using an exact algorithm due to Hudson.
The primary implementation is based on ``cython``.  Alternative implementations
are also provided in pure python, with or without ``numba``.

Releases
--------
This is a pre-release.  Further small changes will be made before the code
is considered available for general use.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5574166.svg
   :target: https://doi.org/10.5281/zenodo.5574166

License
-------
``segreg`` is licensed under a BSD-3-Clause License.  See `LICENSE <LICENSE>`_.

Documentation
-------------
For a technical overview of Segmented Regression and algorithms used in ``segreg``,
see `segmented_regression.pdf <doc/segmented_regression.pdf>`_.

Code documentation and a user guide shall be forthcoming.

Setup
-----
To build core modules, run this:

``python setup.py build_ext --inplace``

This builds **C** code which has already been generated using ``cython``.  As such,
there is no direct ``cython`` dependency.
