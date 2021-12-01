API Reference
=============

Models
******
.. autosummary::
   :toctree: _autosummary

   ~segreg.model.OLSRegressionEstimator
   ~segreg.model.OneBkptSegRegEstimator
   ~segreg.model.TwoBkptSegRegEstimator
   ~segreg.model.one_bkpt_segreg.one_bkpt_rss_func

Tools for Analyzing Results
***************************
.. autosummary::
   :toctree: _autosummary

   ~segreg.analysis.plot_model

Bootstrap
*********
.. autosummary::
   :toctree: _autosummary

   ~segreg.bootstrap.boot_conf_intervals
   ~segreg.bootstrap.bca_acceleration
   ~segreg.bootstrap.model_bca
   ~segreg.bootstrap.bca
   ~segreg.bootstrap.boot_basic_conf_interval
   ~segreg.bootstrap.boot_percentile_conf_interval
   ~segreg.bootstrap.boot_param_dist
   ~segreg.bootstrap.boot_resample
   ~segreg.bootstrap.random_selection_with_replacement
   ~segreg.bootstrap.random_selection_with_replacement_two_series


Datasets
********
.. autosummary::
   :toctree: _autosummary

   ~segreg.data.hinkley
   ~segreg.data.test1
   ~segreg.data.test2

Alternative Implementations
***************************

Pure Python Segmented Regression
--------------------------------
.. autosummary::
   :toctree: _autosummary

   ~segreg.model.alt.estimate_one_bkpt_segreg
   ~segreg.model.alt.estimate_two_bkpt_segreg

Brute Force Grid Search
-----------------------
.. autosummary::
   :toctree: _autosummary

   ~segreg.model.alt.brute_fit_one_bkpt
   ~segreg.model.alt.brute_fit_two_bkpt
