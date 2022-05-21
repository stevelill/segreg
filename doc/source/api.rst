API
===

``segreg.model``: Models
************************
.. autosummary::
   :toctree: _autosummary

   ~segreg.model.OLSRegressionEstimator
   ~segreg.model.OneBkptSegRegEstimator
   ~segreg.model.TwoBkptSegRegEstimator
   ~segreg.model.one_bkpt_rss_func
   ~segreg.model.two_bkpt_rss_func

``segreg.analysis``: Tools for Analyzing Results
************************************************
.. autosummary::
   :toctree: _autosummary

   ~segreg.analysis.plot_models
   ~segreg.analysis.plot_one_bkpt_segreg_rss
   ~segreg.analysis.plot_two_bkpt_segreg_rss

``segreg.bootstrap``: Bootstrap
*******************************
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


``segreg.data``: Datasets
*************************
.. autosummary::
   :toctree: _autosummary

   ~segreg.data.hinkley
   ~segreg.data.test1
   ~segreg.data.test2

``segreg.model.alt``: Alternative Implementations
*************************************************

Pure Python Segmented Regression
--------------------------------
.. autosummary::
   :toctree: _autosummary

   ~segreg.model.alt.fit_one_bkpt
   ~segreg.model.alt.fit_two_bkpt

Brute Force Grid Search
-----------------------
.. autosummary::
   :toctree: _autosummary

   ~segreg.model.alt.brute_fit_one_bkpt
   ~segreg.model.alt.brute_fit_two_bkpt
