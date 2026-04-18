# Human papillomavirus simulator (HPVsim)

[![PyPI version](https://badgen.net/pypi/v/hpvsim/?color=blue)](https://pypi.org/project/hpvsim)
[![Tests](https://github.com/starsimhub/hpvsim/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/starsimhub/hpvsim/actions/workflows/tests.yaml)

## About HPVsim

HPVsim is a flexible agent-based model for simulation HPV transmission
and progression through cervical disease to cancer. The model can be
parameterized with country-specific vital dynamics, structured sexual
networks, co-transmitting HPV genotypes, B- and T-cell mediated
immunity, and high-resolution disease natural history. HPVsim is
designed with a user-first lens: it is implemented in pure Python, has
built-in tools for simulating commonly-used interventions, has been
extensively tested and documented, and runs in a matter of seconds to
minutes on a laptop.

In mid-2025, version 3.0 of HPVsim will be released, built on the
Starsim [Starsim](https://starsim.org/) modeling architecture. Since
this represents a significant shift, version 3.0 and above will live in
a [separate repository](https://github.com/starsimhub/hpvsim). The code
in this repository will be frozen after version 2.2, aside from any
critical bugfixes.

The scientific paper describing HPVsim is available at
<https://doi.org/10.1371/journal.pcbi.1012181>. The recommended citation
is:

> **HPVsim: An agent-based model of HPV transmission and cervical
> disease** (2024). Stuart RM, Cohen JA, Kerr CC, Mathur P , NDMC India,
> Zimmermann M, Rao DW, Boudreau MC, Lee S, Yang L, Klein DJ. *PLOS
> Computational Biology* 20(7): e1012181.
> <https://doi.org/10.1371/journal.pcbi.1012181>

## Background

HPVsim has been used for analyses in several countries. Academic papers
that have been written using HPVsim include:

1.  **Inferring the natural history of HPV from global cancer
    registries: insights from a multi-country calibration** (2024).
    Stuart RM, Cohen JA, Abeysuriya RG, Sanz-Leon P, Kerr CC, Rao DW,
    Klein DJ. *Sci Rep* 14, 15875.
    <https://doi.org/10.1038/s41598-024-65842-3>
2.  **HPV DNA Screening and Vaccination Strategies in Tunisia** (2025).
    Lahdhiri A, Benzina B, Ennaifer E, Tounsi H, Gzara A, Rammeh-Rommani
    S, Laraj O, Arfaoui H, Stuart RM Kebir, A, BenMiled S (2025). *Sci
    Rep*, forthcoming.
3.  **Can pruning improve agent-based models’ calibration? An
    application to HPVsim** (2025). Sturman F, Swallow B, Kerr CC,
    Stuart RM, Panovska-Griffiths J. *Journal of Theoretical Biology*,
    forthcoming.

## Installation

The easiest way to install is simply via pip: `pip install hpvsim`.
Alternatively, you can clone this repository, then run
`pip install -e .` (don't forget the dot!) in this folder to install
`hpvsim` and its dependencies. This will make `hpvsim` available on the
Python path. The first time HPVsim is imported, it will automatically
download the required data files (~30 MB). You can also download the
data manually by running `hpvsim-download-data` from the terminal.

## Usage and documentation

Documentation is available at <https://docs.hpvsim.org>. Additional
usage examples are available in the `tests` folder.

## Contributing

If you wish to contribute, please follow the Starsim style guide at:
<https://github.com/starsimhub/styleguide>. See the code of conduct
readme for more information.

## Disclaimer

The code in this repository was developed by IDM, the Burnet Institute,
and other collaborators to support our joint research on HPV. We've made
it publicly available under the MIT License to provide others with a
better understanding of our research and an opportunity to build upon it
for their own work. Note that HPVsim depends on a number of
user-installed Python packages that can be installed automatically via
`pip install`. We make no representations that the code works as
intended or that we will provide support, address issues that are found,
or accept pull requests. You are welcome to create your own fork and
modify the code to suit your own modeling needs as contemplated under
the MIT License.
