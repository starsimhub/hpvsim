=====
Tests
=====

This folder contains the tests for HPVsim.

Installation
------------

To install test dependencies, use ``pip install -r requirements.txt``.

Usage
-----

Recommended usage is ``./check_coverage`` or ``./run_tests``. You can also use ``pytest`` to run all the tests in the folder.

If you want to test a specific version of HPVsim, you can use the included ``conda`` environments, e.g.::

    conda env create -f hpvsim_v1.2.2.yml