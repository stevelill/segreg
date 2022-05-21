"""
The ``segreg.analysis`` module provides tools for analyzing the results
of segmented regression modelling.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from segreg.analysis.stats_plotting import (plot_models,
                                            plot_one_bkpt_segreg_rss,
                                            plot_two_bkpt_segreg_rss)

__all__ = ["plot_models",
           "plot_one_bkpt_segreg_rss",
           "plot_two_bkpt_segreg_rss"]
