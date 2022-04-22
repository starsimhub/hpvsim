'''
Set the defaults across each of the different files.

TODO: review/merge this across the different *sims

'''

import numpy as np
import numba as nb
import sciris as sc
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
            'rel_trans',        # Float
        ]

        # Set the states that a person can be in: these are all booleans per person -- used in people.py
        self.states = [
            'susceptible',
            'naive',
            'infectious',
            'precancer',
            'cancer',
            'recovered',
            'dead_cancer', # Dead from cancer
            'other_dead',  # Dead from all other causes
        ]

        # Genotype states -- these are ints
        self.genotype_states = [
            'infectious_genotype',
            'precancer_genotype',
            'cancer_genotype',
            'recovered_genotype',
        ]

        # Genotype states -- these are ints, by genotype
        self.by_genotype_states = [
            'infectious_by_genotype',
            'precancer_by_genotype',
            'cancer_by_genotype'
        ]

        # Immune states, by genotype
        self.imm_states = [
            'sus_imm',  # Float, by genotype
        ]

        # Immunity states, by genotype/vaccine
        self.imm_by_source_states = [
            'peak_imm', # Float, peak level of immunity
            'imm',  # Float, current immunity level
            't_imm_event',  # Float, time since immunity event
        ]

        self.dates = [f'date_{state}' for state in self.states] # Convert each state into a date

        # Duration of different states: these are floats per person -- used in people.py
        self.durs = [
            'dur_inf2rec',
            'dur_disease',
        ]

        self.all_states = self.person + self.states + self.genotype_states + self.by_genotype_states + self.imm_states + \
                          self.imm_by_source_states + self.dates + self.durs

        # Validate
        self.state_types = ['person', 'states', 'genotype_states', 'by_genotype_states', 'imm_states',
                            'imm_by_source_states', 'dates', 'durs', 'all_states']
        for state_type in self.state_types:
            states = getattr(self, state_type)
            n_states        = len(states)
            n_unique_states = len(set(states))
            if n_states != n_unique_states: # pragma: no cover
                errormsg = f'In {state_type}, only {n_unique_states} of {n_states} state names are unique'
                raise ValueError(errormsg)

        return



#%% Define other defaults

# A subset of the above states are used for results
result_stocks = {
    'susceptible': 'Number susceptible',
    'infectious':  'Number infectious',
    'precancer':   'Number pre-cancerous',
    'cancer':      'Number cervical cancer',
    'recovered':   'Number recovered',
    'dead_cancer': 'Number dead from cervical cancer',
    'other_dead':  'Number dead from other causes',
}

result_stocks_by_genotype = {
    'infectious_by_genotype': 'Number infectious by genotype',
    'precancer_by_genotype' : 'Number precancerous by genotype',
    'cancer_by_genotype'    : 'Number cervical cancer by genotype'
}

# The types of result that are counted as flows -- used in sim.py; value is the label suffix
result_flows = {
    'infections':   'infections',
    'precancers':   'pre-cancers',
    'cancers':      'cancers',
    'recoveries':   'recoveries',
    'cancer_deaths': 'deaths from cervical cancer',
    'other_deaths': 'deaths from other causes',
    'births':       'births'
}

result_flows_by_genotype = {
    'infections_by_genotype':  'infections by genotype',
    'precancers_by_genotype' :  'precancers by genotype',
    'cancers_by_genotype'    :  'cervical cancers by genotype'
}


# Define new and cumulative flows
new_result_flows = [f'new_{key}' for key in result_flows.keys()]
cum_result_flows = [f'cum_{key}' for key in result_flows.keys()]

new_result_flows_by_genotype = [f'new_{key}' for key in result_flows_by_genotype.keys()]
cum_result_flows_by_genotype = [f'cum_{key}' for key in result_flows_by_genotype.keys()]

# Parameters that can vary by genotype (WIP)
genotype_pars = [
    'rel_beta',

]

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



def get_default_colors():
    '''
    Specify plot colors -- used in sim.py.

    NB, includes duplicates since stocks and flows are named differently.
    '''
    c = sc.objdict()
    c.default               = '#000000'
    c.susceptible           = '#4d771e'
    c.infectious            = '#e45226'
    c.infections            = '#b62413'
    c.precancer             = c.default
    c.cancers               = c.default
    c.precancers            = c.default
    c.cancer                = c.default
    c.infectious_by_genotype = c.infectious
    c.infections_by_genotype = '#b62413'
    c.precancer_by_genotype  = c.default
    c.cancer_by_genotype    = c.default
    c.precancers_by_genotype = c.default
    c.cancers_by_genotype = c.default
    c.reinfections          = '#732e26'
    c.recoveries            = '#9e1149'
    c.recovered             = c.recoveries
    c.other_deaths          = '#000000'
    c.cancer_deaths         = c.default
    c.dead_cancer           = c.default
    c.other_dead            = c.other_deaths
    c.births                = '#797ef6'
    return c


# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
    'cum_infections',
    'new_infections',
    'n_infectious',
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
                'Total counts': [
                    'cum_infections',
                    'n_infectious',
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
