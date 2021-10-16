"""
The ``segreg.analysis`` module provides tools for analyzing the results
of segmented regression modelling.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from segreg.analysis.stats_plotting import (plot_segmented_fit,
                                            plot_boot_sample,
                                            plot_fitted_model,
                                            plot_model)

__all__ = ["plot_segmented_fit",
           "plot_boot_sample",
           "plot_fitted_model",
           "plot_model"]
