"""hpvsim — HPV simulation tools, on Starsim 4.

Public API: ``Sim``, ``HPV``, ``CrossImmunity``, ``HPVTotal``, ``HIV``, ``SexualNetwork``,
the products (``vx``, ``dx``, ``tx``, ``txvx``, ``radiation``) and interventions
(``routine_vx``, ``routine_screening``, ``treat_num``, ...), ``data``.
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
from .seeding import init_prevs
from .hiv import HIV, HIV_transmit, HIV_incidence, ART
from .cross_genotype import CrossImmunity, HPVTotal
from .sim import Sim
from .demographics import AgeMigration, AnnualBirths, Births
from .analyzers import by_age, snapshot, age_pyramid, age_causal_infection, dalys, results_by_genotype
from .calibration import Calibration, make_calib_sims
from . import calibration
from .plotting import plot_by_age, plot_by_genotype, plot_type_distribution, plot_sim, plot_intervention_impact, plot_calibration
from .products import vx, dx, tx, txvx, radiation
from .interventions import (
    routine_vx, campaign_vx,
    routine_screening, campaign_screening,
    routine_triage, campaign_triage,
    treat_num, treat_delay,
    routine_txvx, campaign_txvx, linked_txvx,
    dynamic_pars, any_cancer,
)
from .examples import demo

rootdir = sc.thispath(__file__).parent

__all__ = [
    'HPV', 'SexualNetwork', 'Sim', 'AgeMigration', 'AnnualBirths', 'CrossImmunity', 'HPVTotal',
    'HIV', 'HIV_transmit', 'HIV_incidence', 'ART',
    'by_age', 'snapshot', 'age_pyramid', 'age_causal_infection', 'dalys', 'results_by_genotype',
    'Calibration', 'make_calib_sims', 'calibration',
    'plot_by_age', 'plot_by_genotype', 'plot_type_distribution', 'plot_sim', 'plot_intervention_impact', 'plot_calibration',
    'data', 'options', 'datadir', '__version__',
    'SimPars', 'GenotypePars', 'get_genotype_pars', 'route_pars', 'expanddict', 'par_registry',
    'GENOTYPE_KEYS',
    'vx', 'dx', 'tx', 'txvx', 'radiation',
    'routine_vx', 'campaign_vx',
    'routine_screening', 'campaign_screening',
    'routine_triage', 'campaign_triage',
    'treat_num', 'treat_delay',
    'routine_txvx', 'campaign_txvx', 'linked_txvx',
    'dynamic_pars', 'any_cancer',
    'demo',
]

del sc
