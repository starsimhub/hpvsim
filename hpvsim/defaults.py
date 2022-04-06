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
        ]

        # Set the states that a person can be in: these are all booleans per person -- used in people.py
        self.states = [
            'susceptible',
            'naive',
            'infectious',
            'recovered',
        ]

        # Genotype states -- these are ints
        self.genotype_states = [
            'infectious_genotype',
            'recovered_genotype',
        ]

        # Genotype states -- these are ints, by genotype
        self.by_genotype_states = [
            'infectious_by_genotype',
        ]

        # Immune states, by variant
        self.imm_states = [
            'sus_imm',  # Float, by genotype
        ]

        # Immunity states, by genotype/vaccine
        self.imm_by_source_states = [
            'imm',  # Float, current immunity level
            't_imm_event',  # Float, time since immunity event
        ]

        self.dates = [f'date_{state}' for state in self.states] # Convert each state into a date

        # Duration of different states: these are floats per person -- used in people.py
        self.durs = [
            'dur_exp2inf',
            'dur_disease',
        ]

        self.all_states = self.person + self.states + self.genotype_states + self.by_genotype_states + self.imm_states + self.imm_by_source_states + self.dates + self.durs

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
    'recovered':   'Number recovered',
}

# The types of result that are counted as flows -- used in sim.py; value is the label suffix
result_flows = {
    'infections':   'infections',
    'reinfections': 'reinfections',
    'infectious':   'infectious',
    'recoveries':   'recoveries',
}

# Parameters that can vary by variant
genotype_pars = [
    'rel_beta',
    'rel_symp_prob',
    'rel_severe_prob',
    'rel_crit_prob',
    'rel_death_prob',
]


# Define new and cumulative flows
new_result_flows = [f'new_{key}' for key in result_flows.keys()]
cum_result_flows = [f'cum_{key}' for key in result_flows.keys()]


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


def get_default_colors():
    '''
    Specify plot colors -- used in sim.py.

    NB, includes duplicates since stocks and flows are named differently.
    '''
    c = sc.objdict()
    c.susceptible           = '#4d771e'
    c.exposed               = '#c78f65'
    c.infectious            = '#e45226'
    c.infections            = '#b62413'
    c.reinfections          = '#732e26'
    c.recoveries            = '#9e1149'
    c.recovered             = c.recoveries
    c.default               = '#000000'
    return c


# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
    'cum_infections',
    'new_infections',
    'n_infectious',
    'r_eff',
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
