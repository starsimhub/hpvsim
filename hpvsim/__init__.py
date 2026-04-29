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
from . import parameters
from . import misc
from . import utils

# M01 public API — populated as components land in Tasks 3-7. Imports are
# commented out for now and uncommented in Task 8 (final hpvsim/__init__.py).
# from .hpv import HPV
# from .network import SexualNetwork
# from .sim import Sim

del sc