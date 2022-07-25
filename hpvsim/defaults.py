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
__all__ = ['default_float', 'default_int', 'get_default_plots']


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
            'death_age',        # Float
            'debut',            # Float
            'partners',         # Int by relationship type
            'current_partners', # Int by relationship type
        ]

        # Set the states that a person can be in, all booleans per person and per genotype except other_dead, screened, vaccinated and treated
        self.states = [
            'susceptible',
            'infectious',
            'hpv', # hpv in absence of any CIN
            'cin1',
            'cin2',
            'cin3',
            'cin',
            'cancerous',
            'latent',
            'alive', # Save this as a state so we can record population sizes
            'dead_cancer',
            'dead_other',  # Dead from all other causes
            'vaccinated',
            'screened',
            'treated',
        ]

        # Immune states, by genotype/vaccine
        self.imm_states = [
            'sus_imm',  # Float, by genotype
            'peak_imm',  # Float, peak level of immunity
            'imm',  # Float, current immunity level
            't_imm_event',  # Int, time since immunity event
        ]

        # Additional intervention states
        self.intv_states = [
            'doses',  # Number of doses given per person
            'vaccine_source',  # index of vaccine that individual received
            'screens', # Number of screens given per person
        ]

        # Relationship states
        self.rship_states = [
            'rship_start_dates',
            'rship_end_dates',
            'n_rships'
        ]

        self.dates = [f'date_{state}' for state in self.states if state != 'alive'] # Convert each state into a date
        self.dates += ['date_clearance', 'date_next_screen']

        # Duration of different states: these are floats per person -- used in people.py
        self.durs = [
            'dur_hpv', # Length of time that a person has HPV before progressing to CIN
            'dur_disease', # Length of time that a person has >= HPV present
            'dur_none2cin1', # Length of time to go from no dysplasia to CIN1
            'dur_cin12cin2', # Length of time to go from CIN1 to CIN2
            'dur_cin22cin3', # Length of time to go from CIN2 to CIN3
            'dur_cin2cancer',# Length of time to go from CIN3 to cancer
            'dur_cancer',  # Duration of cancer
        ]

        self.all_states = self.person + self.states + self.imm_states + self.intv_states + \
                          self.dates + self.durs + self.rship_states

        # Validate
        self.state_types = ['person', 'states', 'imm_states', 'intv_states', 'dates', 'durs', 'all_states']
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
flow_keys   = ['infections',    'cin1s',        'cin2s',        'cin3s',        'cins',         'cancers',  'detected_cancers', 'cancer_deaths',    'reinfections',     'reactivations',     'screens',  'screened']
flow_names  = ['infections',    'CIN1s',        'CIN2s',        'CIN3s',        'CINs',         'cancers',  'detected cancers', 'cancer deaths',    'reinfections',     'reactivations',     'screens',  'screened']
flow_colors = [pl.cm.GnBu,      pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds, pl.cm.Purples, pl.cm.Purples,      pl.cm.GnBu,          pl.cm.Purples, pl.cm.Purples, pl.cm.Purples,]

# Stocks: the number in each of the following states
# All are stored (1) by genotype and (2) as the total across genotypes
# the by_age vector tells the sim which results should be stored by age - should have entries in [None, 'total', 'genotype', 'both']
stock_keys   = ['susceptible',  'infectious',   'cin1',         'cin2',         'cin3',         'cin',          'cancerous']
stock_names  = ['susceptible',  'infectious',   'with CIN1',    'with CIN2',    'with CIN3',    'with CIN',     'with cancer']
stock_colors = [pl.cm.Greens,   pl.cm.GnBu,     pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds]

# Incidence and prevalence. Strong overlap with stocks, but with slightly different naming conventions
# All are stored (1) by genotype and (2) as the total across genotypes
inci_keys   = ['hpv',       'cin1',         'cin2',         'cin3',         'cin',          'cancer']
inci_names  = ['HPV',       'CIN1',         'CIN2',         'CIN3',         'CIN',          'cancer']
inci_colors = [pl.cm.GnBu,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds]

# Demographics
dem_keys    = ['births',    'other_deaths']
dem_names   = ['births',    'other deaths']
dem_colors  = ['#fcba03',   '#000000']

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
    [ 0,  4.9, 0.0605],
    [ 5,  9.9, 0.0607],
    [10, 14.9, 0.0566],
    [15, 19.9, 0.0557],
    [20, 24.9, 0.0612],
    [25, 29.9, 0.0843],
    [30, 34.9, 0.0848],
    [35, 39.9, 0.0764],
    [40, 44.9, 0.0697],
    [45, 49.9, 0.0701],
    [50, 54.9, 0.0681],
    [55, 59.9, 0.0653],
    [60, 64.9, 0.0591],
    [65, 69.9, 0.0453],
    [70, 74.9, 0.0312],
    [75, 79.9, 0.02016], # Calculated based on 0.0504 total for >=75
    [80, 84.9, 0.01344],
    [85, 89.9, 0.01008],
    [90, 99.9, 0.00672],
])


default_death_rates = {2020:{
    'm': np.array([
        [0, 5.99966600e-03],
        [1, 2.51593000e-04],
        [5, 1.35127000e-04],
        [10, 1.78153000e-04],
        [15, 6.61341000e-04],
        [20, 1.30016800e-03],
        [25, 1.63925500e-03],
        [30, 1.96618300e-03],
        [35, 2.28799200e-03],
        [40, 2.63302300e-03],
        [45, 3.66449800e-03],
        [50, 5.70753600e-03],
        [55, 9.46976600e-03],
        [60, 1.34425950e-02],
        [65, 1.83650650e-02],
        [70, 2.89760800e-02],
        [75, 4.17993600e-02],
        [80, 6.58443370e-02],
        [85, 1.47244865e-01]]),
    'f': np.array([
        [0, 5.01953300e-03],
        [1, 2.01505000e-04],
        [5, 1.08226000e-04],
        [10, 1.25870000e-04],
        [15, 2.85938000e-04],
        [20, 4.81500000e-04],
        [25, 6.72314000e-04],
        [30, 9.84953000e-04],
        [35, 1.27814400e-03],
        [40, 1.61936000e-03],
        [45, 2.42485500e-03],
        [50, 3.86320600e-03],
        [55, 6.15726500e-03],
        [60, 8.21110500e-03],
        [65, 1.17604260e-02],
        [70, 1.86539200e-02],
        [75, 3.04550980e-02],
        [80, 5.16382510e-02],
        [85, 1.33729522e-01]])
    }}

default_birth_rates = np.array([
    [2015, 2016, 2017, 2018, 2019],
    [12.4, 12.2, 11.8, 11.6, 11.4],
])

default_init_prev = {
    'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
    'm'             : np.array([ 0.0, 0.05, 0.07, 0.05, 0.02, 0.01, 0.0005, 0]),
    'f'             : np.array([ 0.0, 0.05, 0.07, 0.05, 0.02, 0.01, 0.0005, 0]),
}


#%% Default plotting settings

# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
    'total_infections',
    'total_cins',
    'total_cancers',
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
    if which in ['none', 'default', 'epi']:

        if is_sim:
            plots = sc.odict({
                'HPV prevalence': [
                    'total_hpv_prevalence',
                    'hpv_prevalence',
                ],
                'HPV incidence': [
                    'total_hpv_incidence',
                    'hpv_incidence',
                ],
                'CINs and cancers per 100,000 women': [
                    'total_cin_incidence',
                    'cin_incidence',
                    'cancer_incidence',
                    ],
            })

        else: # pragma: no cover
            plots = sc.odict({
                'HPV incidence': [
                    'total_hpv_incidence',
                ],
                'Cancers per 100,000 women': [
                    'total_cancer_incidence',
                    ],
            })

    # Demographic plots
    elif which in ['demographic', 'demographics', 'dem', 'demography']:
        if is_sim:
            plots = sc.odict({
                'Birth and death rates': [
                    'cdr',
                    'cbr',
                ],
                'Population size': [
                    'n_alive',
                    'n_alive_by_sex',
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
