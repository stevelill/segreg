segreg
======

**segreg** is a Python module for segmented regression.

Segmented regression models are defined as having breakpoints where the functional form
changes.  This project is currently limited in scope to the case of piecewise-linear and 
continuous univariate models with at most two breakpoints.

The ``segreg`` code fits segmented regression models to data using an exact algorithm due to Hudson.
The primary implementation is based on ``cython``.  Alternative implementations (of the same
algorithm)
are also provided in pure python, with or without ``numba``.

To accomodate bootstrap and other monte carlo routines, it helps for the model fitting to
be fast and exact.  By exact, we mean that the optimal parameter values minimizing the
residual sum of squares function are found.  This is in contrast to numerically approximate
routines such as grid search or non-linear optimization.

Hudson's algorithm is exact, but can involve a fair amount of computation
for two or more breakpoints.  The implementation here strives to be fast.  


Releases
--------
This is a pre-release.  Further small changes will be made before the code
is considered available for general use.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5574166.svg
   :target: https://doi.org/10.5281/zenodo.5574166

Installation
------------
``pip install segreg``

**We strongly recommended using a virtual environment (venv), or a conda environment,
to avoid possible conflicts with other packages or other issues.**

License
-------
``segreg`` is licensed under a BSD-3-Clause License.  See `LICENSE <LICENSE>`_.

Documentation
-------------
For a technical overview of Segmented Regression and algorithms used in ``segreg``,
see `segmented_regression.pdf <doc/segmented_regression.pdf>`_.

Code documentation and tutorials are available here:
https://stevelill.github.io/segreg/

Development Setup
-----------------
To build core modules, run this:

``python setup.py build_ext --inplace``

This builds **C** code which has already been generated using ``cython``.  As such,
there is no direct ``cython`` dependency.
