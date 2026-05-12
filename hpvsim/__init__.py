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
from .demographics import AgeMigration
from .connectors import CrossImmunity

rootdir = sc.thispath(__file__).parent

__all__ = [
    'HPV', 'SexualNetwork', 'Sim', 'AgeMigration', 'CrossImmunity',
    'data', 'migration_utils', 'options', 'datadir', '__version__',
    'SimPars', 'GenotypePars', 'get_genotype_pars', 'get_cross_immunity',
    'GENOTYPE_KEYS',
]

del sc