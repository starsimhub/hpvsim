"""hpvsim — HPV simulation tools (Starsim-based).

This is the v3 package, in-place replacement of the legacy v2 package.
Public API expands as milestones land:

    M01: Sim, HPV, SexualNetwork, data            (this milestone)
    M02: + natural history (CIN, cancer)
    M03: + multi-genotype, cross-immunity
    M04: + calibration
    M05: + interventions
    ...

v2 modules awaiting port live in hpvsim/_v2_legacy/. Active code MUST NOT
import from there.
"""

import sciris as sc

# Stable utility imports — these modules stayed active through the migration:
from .version import __version__, __versiondate__, __license__
from .settings import options
from .defaults import datadir, default_int, default_float, get_default_plots
from . import data
from . import migration_utils
from . import parameters
from .parameters import SimPars, GenotypePars, get_genotype_pars
from . import misc
from . import utils

from .hpv import HPV
from .migration_utils import Poisson1
from .network import SexualNetwork
from .sim import Sim

rootdir = sc.thispath(__file__).parent

__all__ = [
    'HPV', 'Poisson1', 'SexualNetwork', 'Sim', 'data', 'migration_utils',
    'options', 'datadir', '__version__',
    'SimPars', 'GenotypePars', 'get_genotype_pars',
]

del sc