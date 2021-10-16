"""
Create fake @jit decorator so we can turn off jit if numba not available.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

__all__ = ['jit']


def jit(**kwargs):
    """
    Creates a fake ``@jit`` decorator so we can turn off jit if numba not
    available.

    Example usage:

    >>> try:
    >>>     from numba import jit
    >>> except ImportError as e:
    >>>     from segreg.mockjit import jit
    """

    def real_decorator(function):

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        return wrapper

    return real_decorator
