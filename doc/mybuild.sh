#!/usr/bin/env bash

export PYTHONPATH=..

AUTO_GEN_RST_DIR=source/segreg_automated_docs

AUTOSUMMARY_DIR=source/_autosummary

make clean
rm ${AUTO_GEN_RST_DIR}/*
rm ${AUTOSUMMARY_DIR}/*

## noindex turns off index everywhere
export SPHINX_APIDOC_OPTIONS=members,show-inheritance,inherited-members,undoc-members
#,noindex

# orig had this
#sphinx-apidoc -e -M -f -o ${AUTO_GEN_RST_DIR} ../segreg/

make html

# not auto-generated
# conf.py  index.rst
