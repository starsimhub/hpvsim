'''
Set the defaults across each of the different files.

TODO: review/merge this across the different *sims

'''

import numpy as np
import numba as nb
import sciris as sc
import pylab as pl
from .settings import options as hpo # To set options

# Specify all externally visible functions this file defines -- other things are available as e.g. hp.defaults.default_int
__all__ = ['default_float', 'default_int', 'get_default_colors', 'get_default_plots']


#%% Specify what data types to use

result_float = np.float64 # Always use float64 for results, for simplicity
if hpo.precision == 32:
    default_float = np.float32
    default_int   = np.int32
    nbfloat       = nb.float32
    nbint         = nb.int32
elif hpo.precision == 64: # pragma: no cover
    default_float = np.float64
    default_int   = np.int64
    nbfloat       = nb.float64
    nbint         = nb.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {hpo.precision}')


#%% Define all properties of people

class PeopleMeta(sc.prettyobj):
    ''' For storing all the keys relating to a person and people '''

    def __init__(self):

        # Set the properties of a person
        self.person = [
            'uid',              # Int
            'age',              # Float
            'sex',              # Float
            'debut',            # Float
            'partners',         # Int by relationship type
            'current_partners', # Int by relationship type
        ]

        # Set the states that a person can be in, all booleans per person and per genotype except other_dead
        self.states = [
            'susceptible',
            'infectious',
            'cin1',
            'cin2',
            'cin3',
            'cancerous',
            'alive', # Save this as a state so we can record population sizes
            'dead_cancer',
            'dead_other',  # Dead from all other causes
        ]

        # Immune states, by genotype
        self.imm_states = [
            'sus_imm',  # Float, by genotype
        ]

        # Immunity states, by genotype/vaccine
        self.imm_by_source_states = [
            'peak_imm', # Float, peak level of immunity
            'imm',  # Float, current immunity level
            't_imm_event',  # Int, time since immunity event
        ]

        self.dates = [f'date_{state}' for state in self.states if state != 'alive'] # Convert each state into a date
        self.dates += ['date_clearance']

        # Duration of different states: these are floats per person -- used in people.py
        self.durs = [
            'dur_hpv', # Length of time that a person has HPV DNA present. This is EITHER the period until the virus clears OR the period until viral integration
            'dur_none2cin1', # Length of time to go from no dysplasia to CIN1
            'dur_cin12cin2', # Length of time to go from CIN1 to CIN2
            'dur_cin22cin3', # Length of time to go from CIN2 to CIN3
            'dur_cin2cancer',# Length of time to go from CIN3 to cancer
            'dur_cancer',  # Duration of cancer
        ]

        self.all_states = self.person + self.states + self.imm_states + \
                          self.imm_by_source_states + self.dates + self.durs

        # Validate
        self.state_types = ['person', 'states', 'imm_states',
                            'imm_by_source_states', 'dates', 'durs', 'all_states']
        for state_type in self.state_types:
            states = getattr(self, state_type)
            n_states        = len(states)
            n_unique_states = len(set(states))
            if n_states != n_unique_states: # pragma: no cover
                errormsg = f'In {state_type}, only {n_unique_states} of {n_states} state names are unique'
                raise ValueError(errormsg)

        return


#%% Default result settings

# Flows: we count new and cumulative totals for each
# All are stored (1) by genotype and (2) as the total across genotypes
# the by_age vector tells the sim which results should be stored by age - should have entries in [None, 'total', 'genotype', 'both']
flow_keys   = ['infections',    'cin1s',        'cin2s',        'cin3s',        'cins',         'cancers',  'cancer_deaths',    'reinfections']
flow_names  = ['infections',    'CIN1s',        'CIN2s',        'CIN3s',        'CINs',         'cancers',  'cancer deaths',    'reinfections']
flow_colors = [pl.cm.GnBu,      pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds, pl.cm.Purples,      pl.cm.GnBu]
flow_by_age = ['both',          None,           None,           None,           'total',        'total',    'total',            None]

# Stocks: the number in each of the following states
# All are stored (1) by genotype and (2) as the total across genotypes
# the by_age vector tells the sim which results should be stored by age - should have entries in [None, 'total', 'genotype', 'both']
stock_keys   = ['susceptible',  'infectious',   'cin1',         'cin2',         'cin3',         'cin',          'cancerous']
stock_names  = ['susceptible',  'infectious',   'with CIN1',    'with CIN2',    'with CIN3',    'with CIN',     'with cancer']
stock_colors = [pl.cm.Greens,   pl.cm.GnBu,     pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds]
stock_by_age = ['total',        'both',         None,           None,           None,           'total',        'total']

# Incidence and prevalence. Strong overlap with stocks, but with slightly different naming conventions
# All are stored (1) by genotype and (2) as the total across genotypes
inci_keys   = ['hpv',       'cin1',         'cin2',         'cin3',         'cin',          'cancer']
inci_names  = ['HPV',       'CIN1',         'CIN2',         'CIN3',         'CIN',          'cancer']
inci_colors = [pl.cm.GnBu,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds]
inci_by_age = ['both',      None,           None,           None,           'total',        'total']

# Results by age
age_brackets    = np.array([15, 25, 45, 65, 150])  # TODO: consider how this will change once vaccination status is there
age_labels      = ['0-14', '15-24', '25-44', '45-64', '65+']
n_age_brackets  = len(age_brackets)
by_age_colors   = sc.gridcolors(n_age_brackets)

# Demographics
dem_keys    = ['births',    'other_deaths']
dem_names   = ['births',    'other deaths']
dem_colors  = ['#000000',   '#000000']

# Results by sex
by_sex_keys    = ['total_infections_by_sex',    'other_deaths_by_sex']
by_sex_names   = ['total infections by sex',    'deaths from other causes by sex']
by_sex_colors  = ['#000000',                    '#000000']


#%%
# Parameters that can vary by genotype (WIP)
genotype_pars = [
    'rel_beta',
    'rel_cin1_prob',
    'rel_cin2_prob',
    'rel_cin3_prob',
    'rel_cancer_prob',
    'rel_death_prob'
]

#%% Default data (age, death rates, birth dates, initial prevalence)

# Default age data, based on Seattle 2018 census data -- used in population.py
default_age_data = np.array([
    [ 0,  4, 0.0605],
    [ 5,  9, 0.0607],
    [10, 14, 0.0566],
    [15, 19, 0.0557],
    [20, 24, 0.0612],
    [25, 29, 0.0843],
    [30, 34, 0.0848],
    [35, 39, 0.0764],
    [40, 44, 0.0697],
    [45, 49, 0.0701],
    [50, 54, 0.0681],
    [55, 59, 0.0653],
    [60, 64, 0.0591],
    [65, 69, 0.0453],
    [70, 74, 0.0312],
    [75, 79, 0.02016], # Calculated based on 0.0504 total for >=75
    [80, 84, 0.01344],
    [85, 89, 0.01008],
    [90, 99, 0.00672],
])


default_death_rates = {
    'm': np.array([
        [0, 1, 5.99966600e-03],
        [1, 4, 2.51593000e-04],
        [5, 9, 1.35127000e-04],
        [10, 14, 1.78153000e-04],
        [15, 19, 6.61341000e-04],
        [20, 24, 1.30016800e-03],
        [25, 29, 1.63925500e-03],
        [30, 34, 1.96618300e-03],
        [35, 39, 2.28799200e-03],
        [40, 44, 2.63302300e-03],
        [45, 49, 3.66449800e-03],
        [50, 54, 5.70753600e-03],
        [55, 59, 9.46976600e-03],
        [60, 64, 1.34425950e-02],
        [65, 69, 1.83650650e-02],
        [70, 74, 2.89760800e-02],
        [75, 79, 4.17993600e-02],
        [80, 84, 6.58443370e-02],
        [85, 99, 1.47244865e-01]]),
    'f': np.array([
        [0, 1, 5.01953300e-03],
        [1, 4, 2.01505000e-04],
        [5, 9, 1.08226000e-04],
        [10, 14, 1.25870000e-04],
        [15, 19, 2.85938000e-04],
        [20, 24, 4.81500000e-04],
        [25, 29, 6.72314000e-04],
        [30, 34, 9.84953000e-04],
        [35, 39, 1.27814400e-03],
        [40, 44, 1.61936000e-03],
        [45, 49, 2.42485500e-03],
        [50, 54, 3.86320600e-03],
        [55, 59, 6.15726500e-03],
        [60, 64, 8.21110500e-03],
        [65, 69, 1.17604260e-02],
        [70, 74, 1.86539200e-02],
        [75, 79, 3.04550980e-02],
        [80, 84, 5.16382510e-02],
        [85, 99, 1.33729522e-01]])
    }

default_birth_rates = np.array([
    [2015, 2016, 2017, 2018, 2019],
    [12.4, 12.2, 11.8, 11.6, 11.4],
])

default_init_prev = {
    'age_brackets'  : np.array([  12,   17,   24,   34,   44,   64,   150]),
    'm'             : np.array([ 0.0, 0.05, 0.12, 0.25, 0.15, 0.05, 0.005]),
    'f'             : np.array([ 0.0, 0.05, 0.12, 0.25, 0.15, 0.05, 0.005]),
}

#%% Default plotting settings

# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
    'cum_total_infections',
    'cum_total_cins',
    'cum_total_cancers',
]


def get_default_plots(which='default', kind='sim', sim=None):
    '''
    Specify which quantities to plot; used in sim.py.

    Args:
        which (str): either 'default' or 'overview'
    '''
    which = str(which).lower() # To make comparisons easier

    # Check that kind makes sense
    sim_kind   = 'sim'
    scens_kind = 'scens'
    kindmap = {
        None:      sim_kind,
        'sim':     sim_kind,
        'default': sim_kind,
        'msim':    scens_kind,
        'scen':    scens_kind,
        'scens':   scens_kind,
    }
    if kind not in kindmap.keys():
        errormsg = f'Expecting "sim" or "scens", not "{kind}"'
        raise ValueError(errormsg)
    else:
        is_sim = kindmap[kind] == sim_kind

    # Default plots -- different for sims and scenarios
    if which in ['none', 'default']:

        if is_sim:
            plots = sc.odict({
                'HPV prevalence': [
                    'total_hpv_prevalence',
                    'hpv_prevalence',
                ],
                'HPV incidence': [
                    'total_hpv_incidence_by_age',
                    # 'new_infections',
                ],
                'CINs and cancers per 100,000 women': [
                    'total_cin_incidence',
                    'cin_incidence',
                    'total_cancer_incidence',
                    ],
            })

        else: # pragma: no cover
            plots = sc.odict({
                'Cumulative infections': [
                    'cum_infections',
                ],
                'New infections per day': [
                    'new_infections',
                ],
            })

    # Show an overview
    elif which == 'overview': # pragma: no cover
        plots = sc.dcp(overview_plots)

    # Plot absolutely everything
    elif which == 'all': # pragma: no cover
        plots = sim.result_keys('all')

    # Show an overview 
    elif 'overview' in which: # pragma: no cover
        plots = sc.dcp(overview_plots)

    else: # pragma: no cover
        errormsg = f'The choice which="{which}" is not supported'
        raise ValueError(errormsg)

    return plots
