"""hpvsim — HPV simulation tools (Starsim-based).

Public API: ``Sim``, ``HPV``, ``SexualNetwork``, ``AgeMigration``,
``SimPars``, ``GenotypePars``, ``get_genotype_pars``, ``data``.
"""

import sciris as sc

from .version import __version__, __versiondate__, __license__
from .settings import options
from .defaults import datadir
from . import data
from .parameters import SimPars, GenotypePars, get_genotype_pars, GENOTYPE_KEYS, route_pars, expanddict, par_registry
from . import misc
from . import utils

from .data.country import load_country
from .hpv import HPV, _normalize_genotype
from .network import SexualNetwork
from .seeding import _ExclusiveSeeder
from .sim import Sim
from .demographics import AgeMigration, AnnualBirths, Births
from .cross_genotype import CrossImmunity, HPVTotal
from .analyzers import by_age, snapshot, age_pyramid, age_causal_infection, dalys, results_by_genotype
from .calibration import Calibration, make_calib_sims
from . import calibration
from .plotting import plot_by_age, plot_by_genotype, plot_type_distribution, plot_sim, plot_intervention_impact, plot_calibration
from .products import vx, dx, tx, txvx, radiation
from .interventions import (
    BaseVaccination, routine_vx, campaign_vx,
    BaseTest, BaseScreening, BaseTriage,
    routine_screening, campaign_screening,
    routine_triage, campaign_triage,
    BaseTreatment, treat_num, treat_delay,
    BaseTxVx, routine_txvx, campaign_txvx, linked_txvx,
    dynamic_pars,
)
from .examples import demo

rootdir = sc.thispath(__file__).parent

__all__ = [
    'HPV', 'SexualNetwork', 'Sim', 'AgeMigration', 'AnnualBirths', 'CrossImmunity', 'HPVTotal',
    'by_age', 'snapshot', 'age_pyramid', 'age_causal_infection', 'dalys', 'results_by_genotype',
    'Calibration', 'make_calib_sims', 'calibration',
    'plot_by_age', 'plot_by_genotype', 'plot_type_distribution', 'plot_sim', 'plot_intervention_impact', 'plot_calibration',
    'data', 'options', 'datadir', '__version__',
    'SimPars', 'GenotypePars', 'get_genotype_pars', 'route_pars', 'expanddict', 'par_registry',
    'GENOTYPE_KEYS',
    'vx', 'dx', 'tx', 'txvx', 'radiation',
    'BaseVaccination', 'routine_vx', 'campaign_vx',
    'BaseTest', 'BaseScreening', 'BaseTriage',
    'routine_screening', 'campaign_screening',
    'routine_triage', 'campaign_triage',
    'BaseTreatment', 'treat_num', 'treat_delay',
    'BaseTxVx', 'routine_txvx', 'campaign_txvx', 'linked_txvx',
    'dynamic_pars',
    'demo',
]

del sc


_HIV_NAMES = ('HIV', 'HIV_transmit', 'HIV_incidence')


def __getattr__(name):
    """Expose the HIV classes lazily; stisim is an optional dependency."""
    if name in _HIV_NAMES:
        misc.require_stisim()
        from . import hiv
        return getattr(hiv, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(_HIV_NAMES))
