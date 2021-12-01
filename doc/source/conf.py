# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# Author: Steven Lillywhite
# License: BSD 3 clause


import os
import sys
sys.path.insert(0, os.path.abspath('../../segreg/'))

# -- Project information -----------------------------------------------------

project = 'segreg'
copyright = '2021, Steven Lillywhite'
author = 'Steven Lillywhite'

# The full version, including alpha/beta/rc tags
release = '1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'nbsphinx']
#'sphinx.ext.mathjax']

#mathjax_path = 'MathJax.js'
#mathjax_path = 'file:///home/steven/github/segreg/doc/source/_static/MathJax.js'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', 'autodoc']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
###html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
#html_theme = 'bizstyle'
#html_theme = 'agogo'
#html_theme = 'sphinxdoc'
html_theme = 'nature'
#html_theme = 'pyramid'

# enable overrides of theme; uses stylesheet in source/_style


html_css_files = ['theme_overrides.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False

# orig -- works for apidoc
# issue: sphinx won't show doc for @jit-decorated functions
# this workaround appears to fix
# https://github.com/sphinx-doc/sphinx/issues/3783
def setup(app):
    import inspect
    isfunction = inspect.isfunction
    inspect.isfunction = lambda f: isfunction(f)
    return {}

# don't use this since jit is a function
#    inspect.isfunction = lambda f: isfunction(f) or isinstance(f, jit)

