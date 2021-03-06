"""
Compute segreg by brute force by searching for minimum RSS on a grid.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np
from segreg.model.alt import two_bkpt_segreg_alt, one_bkpt_segreg_alt


try:
    from numba import jit
except ImportError as e:
    from segreg.mockjit import jit


@jit(nopython=True, cache=False)
def estimate_one_bkpt_segreg(indep,
                             dep,
                             num_end_to_skip=0,
                             dx=0.0001,
                             verbose=False,
                             extra_verbose=False):
    """
    Estimate one-bkpt segmented regression model using a brute-force method.

    This method is limited to univariate, continuous, linear, one-bkpt 
    segmented regression problems.  Estimates the parameters: 

        ``[u, v, m1, m2]``

    where

        ``(u,v)`` is the breakpoint (in x-y plane)

        ``m1`` is the slope of the left-hand segment

        ``m2`` is the slope of the right-hand segment

    Notes
    -----
    This method picks the parameters giving the minimum RSS by brute-force
    calculation on a grid in the space of bkpt parameter, ``u``.  That is, for
    each choice of bkpt, the method computes the RSS for the segmented
    fit conditional on that bkpt. The parameters corresponding to the overall 
    minimal RSS from these calculations are returned.

    Parameters
    ----------
    indep: numpy array of shape (num_data,)
        The independent data.  Also called predictor, explanatory variable,
        regressor, or exogenous variable.
    dep: numpy array of shape (num_data,)
        The dependent data.  Also called response, regressand, or endogenous
        variable.
    num_end_to_skip: int
        Number of data points to skip at each end of the data when solving for
        the bkpts.  As such, this determines a guaranteed minimum number of data 
        points in the left and right segments in the returned fit.
        If None, defaults to the underlying implementation.
        TODO: explain
    dx: float
        The stepsize for the grid search.
    verbose: bool
    extra_verbose: bool

    Examples
    --------
    >>> import numpy as np
    >>> from segreg.model.alt import brute_fit_one_bkpt
    >>> indep = np.array([1,2,3,4,5,6,7,8,9])
    >>> dep = np.array([1,2,3,4,5,4,3,2,1])
    >>> brute_fit_one_bkpt(indep, dep)
    (array([ 5.,  5.,  1., -1.]), 0.0)

    Returns
    -------
    params: array of shape (num_params,)
        The estimated parameters.  The returned parameters are, in order,
        [u, v, m1, m2].
    rss: float
        Residual sum of squares of the fit.
    """

    # list call gets set out of order, so we call sort here on indices
    unique_indep = list(set(indep))
    unique_indep_lhs_indices = np.searchsorted(indep, unique_indep)
    unique_indep_lhs_indices.sort()
    unique_indep = indep[unique_indep_lhs_indices]

    num_data = len(unique_indep)

    # STEP 2. check for local min in intervals between data points
    index1_begin = num_end_to_skip + 1
    index1_end = num_data - num_end_to_skip - 1

    indices_to_check = unique_indep_lhs_indices[index1_begin:index1_end]

    first_index = indices_to_check[0]
    last_index = indices_to_check[-1]

    u_start = indep[first_index]
    u_end = indep[last_index]

    u_domain = np.arange(u_start, u_end + dx, dx)

    # np.arange can be inconsistent; sometimes goes beyond endpt
    if u_domain[-1] - u_end > 0.0:
        if extra_verbose:
            print()
            print("DELETING u1 end")
            print("B4 u1 domain end: ", u_domain[-1])
        u_domain = np.delete(u_domain, -1)
        if extra_verbose:
            print("AFTA u1 domain end: ", u_domain[-1])
            print()

    min_params = None
    min_rss = np.inf

    if verbose:
        print()
        print("u_start: ", u_domain[0])
        print("u_end: ", u_domain[-1])
        print("dx: ", dx)
        print()
        print("indices to check: ", indices_to_check)
        print()
        print("u_domain")
        print(u_domain)
        print()

    # np.arange may not hit endpts exactly, so we brute force the corners
    for u in indep[indices_to_check]:

        if verbose:
            print("corner: ", u)

        [v, m1, m2, rss] = one_bkpt_segreg_alt.fixed_bkpt_ls(indep, dep, u)

        if rss < min_rss:
            min_params = np.array([u, v, m1, m2])
            min_rss = rss

    for u in u_domain:

        [v, m1, m2, rss] = one_bkpt_segreg_alt.fixed_bkpt_ls(indep, dep, u)

        if rss < min_rss:
            min_params = np.array([u, v, m1, m2])
            min_rss = rss

    # for straight-line data, the fitted rss can sometimes be negative,
    # due to noise in the computations
    if abs(min_rss) < 1.0e-12:
        min_rss = 0.0

    return min_params, min_rss


@jit(nopython=True, cache=False)
def estimate_two_bkpt_segreg(indep,
                             dep,
                             num_end_to_skip=3,
                             num_between_to_skip=4,
                             dx=0.01,
                             verbose=False,
                             extra_verbose=False):
    """
    Estimate two-bkpt segmented regression model using a brute-force method.

    This method is limited to univariate, continuous, linear, two-bkpt 
    segmented regression problems.  Estimates the parameters: 

    ``[u1, v1, u2, v2, m1, m2]``

    where

        ``(u1,v1), (u2, v2)`` are the breakpoints (in x-y plane), ordered such
        that ``u1 < u2``

        ``m1`` is the slope of the left-most segment

        ``m2`` is the slope of the right-most segment

    Notes
    -----
    This method picks the parameters giving the minimum RSS by brute-force
    calculation on a grid in the space of bkpt parameters, ``(u1, u2)``.  That 
    is, for each choice of bkpt pairs, the method computes the RSS for the 
    segmented fit conditional on the bkpt pair. The parameters corresponding to 
    the overall minimal RSS from these calculations are returned.

    Parameters
    ----------
    indep: numpy array of shape (num_data,)
        The independent data.  Also called predictor, explanatory variable,
        regressor, or exogenous variable.
    dep: numpy array of shape (num_data,)
        The dependent data.  Also called response, regressand, or endogenous
        variable.
    num_end_to_skip: int
        Number of data points to skip at each end of the data when solving for
        the bkpts.  As such, this determines a guaranteed minimum number of data 
        points in the left and right segments in the returned fit.
        If None, defaults to the underlying implementation.
        TODO: explain
    num_between_to_skip: int
        Number of data points to skip between the two bkpts (ie: the middle
        segment) when solving for the bkpts.  Specifically, for each choice of
        left bkpt ``u1``, will skip this many data points between ``u1`` and
        ``u2``.  As such, this determines a guaranteed minimum number of data 
        points between the bkpts in the returned fit.
    dx: float
        The stepsize for the grid search.
    verbose: bool
    extra_verbose: bool

    Examples
    --------
    >>> import numpy as np
    >>> from segreg.model.alt import brute_fit_two_bkpt
    >>> indep = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    >>> dep = np.array([1,2,3,4,5,4,3,2,1,0,1,2,3,4])
    >>> brute_fit_two_bkpt(indep, dep)
    (array([ 5.,  5., 10., -0.,  1.,  1.]), 0.0)


    Returns
    -------
    params: array of shape (num_params,)
        The estimated parameters.  The returned parameters are, in order,
        [u, v, m1, m2].
    rss: float
        Residual sum of squares of the fit.
    """
    # list call gets set out of order, so we call sort here on indices
    unique_indep = list(set(indep))
    unique_indep_lhs_indices = np.searchsorted(indep, unique_indep)
    unique_indep_lhs_indices.sort()
    unique_indep = indep[unique_indep_lhs_indices]

    # STEP 2. check for local min in intervals between data points
    # NOTE: we use array mask with minus, like: x[0:-3]
    # these are indices of LHS of allowable intervals to check
    num_uniq = len(unique_indep)
    index1_begin = num_end_to_skip + 1
    index2_end = num_uniq - num_end_to_skip - 3
    index1_end = index2_end - num_between_to_skip

    if verbose:
        print()
        print("indep")
        print(indep)
        print()
        print("unique_indep_lhs_indices")
        print(unique_indep_lhs_indices)
        print()
        print()
        print("num_end_to_skip: ", num_end_to_skip)
        print("num_between_to_skip: ", num_between_to_skip)
        print("unique range: ", np.arange(num_uniq))
        print("index1_begin: ", index1_begin)
        print("index1_end: ", index1_end)
        print("index2_end: ", index2_end)
        print()

    index1_range = np.arange(index1_begin, index1_end + 1)

    #print("index1_range: ", index1_range)

    min_params = None
    min_rss = np.inf

    for index1 in index1_range:
        index2_begin = index1 + num_between_to_skip
        index2_range = np.arange(index2_begin, index2_end + 1)

        for index2 in index2_range:

            ind1 = unique_indep_lhs_indices[index1 + 1]
            ind2 = unique_indep_lhs_indices[index2 + 1]

            indep1 = indep[0:ind1]
            indep2 = indep[ind1:ind2]
            indep3 = indep[ind2:]

            # TODO: put in check that mins actually hit
            if len(indep1) < num_end_to_skip:
                raise Exception("region one has too few data")
            if len(indep2) < num_between_to_skip:
                raise Exception("region two has too few data")
            if len(indep3) < num_end_to_skip:
                raise Exception("region three has too few data")

            u1_data = indep1[-1]
            u1_data_next = indep2[0]

            u2_data = indep2[-1]
            u2_data_next = indep3[0]

            if verbose:
                print()
                print("--------------------------------------------------------")
                print("INDICES: ", index1, ",", index2)
                print()
                print("u1 interval: ", u1_data, ", ", u1_data_next)
                print("u2 interval: ", u2_data, ", ", u2_data_next)
                print()
                print("indep1: len: ", len(indep1))
                print(indep1)
                print("indep2: len: ", len(indep2))
                print(indep2)
                print("indep3: len: ", len(indep3))
                print(indep3)
                print()

            u1_domain = np.arange(u1_data, u1_data_next + dx, dx)

            # np.arange can be inconsistent; sometimes goes beyond endpt
            if u1_domain[-1] - u1_data_next > 1.0e-10:
                if extra_verbose:
                    print()
                    print("DELETING u1 end")
                    print("B4 u1 domain end: ", u1_domain[-1])
                u1_domain = np.delete(u1_domain, -1)
                if extra_verbose:
                    print("AFTA u1 domain end: ", u1_domain[-1])
                    print()

            u2_domain = np.arange(u2_data, u2_data_next + dx, dx)

            # np.arange can be inconsistent; sometimes goes beyond endpt
            if u2_domain[-1] - u2_data_next > 1.0e-10:
                if extra_verbose:
                    print()
                    print("DELETING u2 end")
                    print("B4 u2 domain end: ", u2_domain[-1])
                u2_domain = np.delete(u2_domain, -1)
                if extra_verbose:
                    print("AFTA u1 domain end: ", u2_domain[-1])
                    print()

            if verbose:
                print()
                print("u1_start: ", u1_domain[0])
                print("u1_end: ", u1_domain[-1])
                print("u2_start: ", u2_domain[0])
                print("u2_end: ", u2_domain[-1])
                print("dx: ", dx)
                print()

            # np.arange may not hit endpts exactly, so we brute force the
            # corners
            for u1 in [u1_data, u1_data_next]:

                for u2 in [u2_data, u2_data_next]:

                    if verbose:
                        print()
                        print("checking corners: [u1, u2]: ", np.array([u1, u2]))

                    [v1, v2, m1, m2, rss] = two_bkpt_segreg_alt.fixed_bkpt_ls_from_data(indep,
                                                                                        dep,
                                                                                        u1,
                                                                                        u2)

                    if rss < min_rss:
                        min_params = np.array([u1, v1, u2, v2, m1, m2])
                        min_rss = rss
                        if verbose:
                            print()
                            print("NEW LOW; RSS: ", min_rss)
                            print("PARAMS: ", min_params)
                            print()

            # main loop
            for u1 in u1_domain:

                for u2 in u2_domain:

                    if extra_verbose:
                        print()
                        print("checking: [u1, u2]: ", np.array([u1, u2]))

                    [v1, v2, m1, m2, rss] = two_bkpt_segreg_alt.fixed_bkpt_ls_from_data(indep,
                                                                                        dep,
                                                                                        u1,
                                                                                        u2)

                    if rss < min_rss:
                        min_params = np.array([u1, v1, u2, v2, m1, m2])
                        min_rss = rss
                        if verbose:
                            print()
                            print("NEW LOW; RSS: ", min_rss)
                            print("PARAMS: ", min_params)
                            print()

    # for straight-line data, the fitted rss can sometimes be negative,
    # due to noise in the computations
    if abs(min_rss) < 1.0e-13:
        min_rss = 0.0

    return min_params, min_rss
