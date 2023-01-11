Human papillomavirus simulator (HPVsim)
=======================================

.. image:: https://github.com/amath-idm/hpvsim/actions/workflows/tests.yaml/badge.svg
    :target: https://github.com/amath-idm/hpvsim/actions/workflows/tests.yaml
    :alt: pipeline status

.. image:: https://github.com/amath-idm/hpvsim/actions/workflows/docsbuild.yaml/badge.svg
    :target: https://github.com/amath-idm/hpvsim/actions/workflows/docsbuild.yaml
    :alt: pipeline status

This repository contains the code for IDM's human papillomavirus simulator, HPVsim. 

**HPVsim is currently under development**.

The structure is as follows:

- HPVsim, in the folder ``hpvsim``, is a standalone Python library for performing HPV analyses.
- Data is in the ``data`` folder.
- Docs are in the ``docs`` folder.
- Tests are in the ``tests`` folder.


Installation
------------

The easiest way to install is simply via pip: ``pip install hpvsim``. Alternatively, you can clone this repository, then run ``pip install -e .`` (don't forget the dot!) in this folder to install ``hpvsim`` and its dependencies. This will make ``hpvsim`` available on the Python path. The first time HPVsim is imported, it will automatically download the required data files (~30 MB).


Usage
-----

See the tests folder for usage examples.


Documentation
-------------

Documentation is available at https://docs.idmod.org/projects/hpvsim/en/latest/.


Contributing
------------

**Style guide**

Please follow the starsim style guide at: https://github.com/amath-idm/styleguide


Disclaimer
----------

The code in this repository was developed by IDM, the Burnet Institute, and other collaborators to support our joint research on HPV. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. Note that HPVsim depends on a number of user-installed Python packages that can be installed automatically via ``pip install``. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the MIT License. See the contributing and code of conduct READMEs for more information.


