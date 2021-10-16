"""
Items for ``regression.``
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


cdef struct OLSData:
    size_t num
    double sum_x
    double sum_y
    double sum_xx
    double sum_yy
    double sum_xy

## TODO: rename to OLSResult
cdef struct OlsEstTerms:
    double intercept
    double slope
    double rss

cdef OLSData add(OLSData ols_data_lhs, OLSData ols_data_rhs)

cdef OLSData subtract(OLSData ols_data_lhs, OLSData ols_data_rhs)

cdef OLSData ols_data(double[:] x_arr, double[:] y_arr)

cdef double sum(double[:] arr)

cdef double vdot(double[:] lhs_arr, double[:] rhs_arr)

cdef OlsEstTerms ols_from_formula_with_rss_cimpl(OLSData ols_data, double slope=*)
