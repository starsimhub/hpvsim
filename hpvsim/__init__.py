"""hpvsim — HPV simulation tools (Starsim-based).

Public API: ``Sim``, ``HPV``, ``SexualNetwork``, ``AgeMigration``,
``SimPars``, ``GenotypePars``, ``get_genotype_pars``, ``data``.

Quarantined modules awaiting port live in ``hpvsim/_v2_legacy/``;
active code must not import from there.
"""

import sciris as sc

from .version import __version__, __versiondate__, __license__
from .settings import options
from .defaults import datadir
from . import data
from . import migration_utils
from .parameters import SimPars, GenotypePars, get_genotype_pars, get_cross_immunity, GENOTYPE_KEYS
from . import misc
from . import utils

from .hpv import HPV
from .network import SexualNetwork
from .sim import Sim
from .demographics import AgeMigration, AnnualBirths
from .cross_genotype import CrossImmunity, HPVTotal
from .analyzers import AgeResults
from .calibration import Calibration
from . import calibration
from .products import vx
from .interventions import BaseVaccination, routine_vx, campaign_vx

rootdir = sc.thispath(__file__).parent

__all__ = [
    'HPV', 'SexualNetwork', 'Sim', 'AgeMigration', 'AnnualBirths', 'CrossImmunity', 'HPVTotal',
    'AgeResults', 'Calibration', 'calibration',
    'data', 'migration_utils', 'options', 'datadir', '__version__',
    'SimPars', 'GenotypePars', 'get_genotype_pars', 'get_cross_immunity',
    'GENOTYPE_KEYS',
    'vx', 'BaseVaccination', 'routine_vx', 'campaign_vx',
]

del sc