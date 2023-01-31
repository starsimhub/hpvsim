Human papillomavirus simulator (HPVsim)
=======================================

.. image:: https://github.com/amath-idm/hpvsim/actions/workflows/tests.yaml/badge.svg
    :target: https://github.com/amath-idm/hpvsim/actions/workflows/tests.yaml
    :alt: pipeline status

.. image:: https://github.com/amath-idm/hpvsim/actions/workflows/docsbuild.yaml/badge.svg
    :target: https://github.com/amath-idm/hpvsim/actions/workflows/docsbuild.yaml
    :alt: pipeline status

This repository contains the code for the Starsim suite's human papillomavirus simulator, HPVsim. HPVsim is a flexible agent-based model that can be parameterized with country-specific vital dynamics, structured sexual networks, co-transmitting HPV genotypes, B- and T-cell mediated immunity, and high-resolution disease natural history. HPVsim is designed with a user-first lens: it is implemented in pure Python, has built-in tools for simulating commonly-used interventions, has been extensively tested and documented, and runs in a matter of seconds to minutes on a laptop. Useful complexity was not sacrificed: the platform is flexible, allowing bespoke scenario modeling.

HPVsim is currently under active development.


Installation
------------

The easiest way to install is simply via pip: ``pip install hpvsim``. Alternatively, you can clone this repository, then run ``pip install -e .`` (don't forget the dot!) in this folder to install ``hpvsim`` and its dependencies. This will make ``hpvsim`` available on the Python path. The first time HPVsim is imported, it will automatically download the required data files (~30 MB).


Usage and documentation
-----------------------

Documentation is available at https://docs.hpvsim.org. Additional usage examples are available in the ``tests`` folder.


Contributing
------------

If you wish to contribute, please follow the Starsim style guide at: https://github.com/amath-idm/styleguide. See the code of conduct readme for more information.


Disclaimer
----------

The code in this repository was developed by IDM, the Burnet Institute, and other collaborators to support our joint research on HPV. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. Note that HPVsim depends on a number of user-installed Python packages that can be installed automatically via ``pip install``. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the MIT License. 


