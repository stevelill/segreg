"""
Items for ``one_bkpt_segreg``.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from segreg.model.regression cimport OLSData

cdef struct FixedBkptTerms:
    double v
    double m1
    double m2
    double rss

cdef FixedBkptTerms fixed_breakpt_ls(OLSData ols_data_1, 
                                     OLSData ols_data_2, 
                                     double u, 
                                     double m2=*)
