'''
Set the defaults across each of the different files.

TODO: review/merge this across the different starsims
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

class State():
    def __init__(self, name, dtype, fill_value=None, shape=None):
        """

        :param name:
        :param dtype:
        :param fill_value:
        :param shape: If not none, set to match a string in `pars` containing the dimensionality e.g., `n_genotypes`)

        """
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value
        self.shape = shape

    def new(self, pars, n):
        array_shape = n if self.shape is None else (pars[self.shape], n)

        if self.fill_value is None:
            return np.empty(array_shape, dtype=self.dtype)
        elif self.fill_value == 0:
            return np.zeros(array_shape, dtype=self.dtype)
        else:
            return np.full(array_shape, dtype=self.dtype, fill_value=self.fill_value)


class PeopleMeta(sc.prettyobj):
    ''' For storing all the keys relating to a person and people '''

    # (attribute, nrows, dtype, default value)
    # If the default value is None, then the array will not be initialized - this is faster and can
    # be used for variables where the People object explicitly supplies the values e.g. age

    # Set the properties of a person
    person = [
        State('uid',default_int),              # Int
        State('age',default_float, np.nan),            # Float
        State('sex',default_float, np.nan),              # Float
        State('debut',default_float, np.nan),         # Float
        State('partners', default_float, np.nan, shape='n_partner_types'),  # Int by relationship type
        State('current_partners', default_float, 0, 'n_partner_types'),  # Int by relationship type
    ]

    # Set the states that a person can be in, all booleans per person and per genotype except cancerous, detected_cancer, cancer_genotype, dead_cancer, other_dead, screened, vaccinated, treated
    states = [
        State('susceptible', bool, True, 'n_genotypes'),
        State('infectious', bool, False, 'n_genotypes'),
        State('none', bool, False, 'n_genotypes'), # HPV without dysplasia
        State('cin1', bool, False, 'n_genotypes'),
        State('cin2', bool, False, 'n_genotypes'),
        State('cin3', bool, False, 'n_genotypes'),
        State('cin', bool, False, 'n_genotypes'),
        State('cancerous', bool, False),
        State('detected_cancer', bool, False),
        State('cancer_genotype', default_int, -2147483648),
        State('latent', bool, False,'n_genotypes'),
        State('alive', bool, True), # Save this as a state so we can record population sizes
        State('dead_cancer', bool, False),
        State('dead_other', bool, False),  # Dead from all other causes
        State('vaccinated', bool, False),
        State('screened', bool, False),
        State('treated', bool, False)
    ]

    # Set genotype states, which store info about which genotype a person is exposed to

    # Immune states, by genotype/vaccine
    imm_states = [
        State('sus_imm', default_float, 0,'n_imm_sources'),  # Float, by genotype
        State('peak_imm', default_float, 0,'n_imm_sources'),  # Float, peak level of immunity
        State('imm', default_float, 0,'n_imm_sources'),  # Float, current immunity level
        State('t_imm_event', default_int, 0,'n_imm_sources'),  # Int, time since immunity event
    ]

    # Additional intervention states
    intv_states = [
        State('doses',default_int, 0),  # Number of doses given per person
        State('vaccine_source',default_int, 0),  # index of vaccine that individual received
        State('screens',default_int, 0),  # Number of screens given per person
    ]

    # Relationship states
    rship_states = [
        State('rship_start_dates', default_float, np.nan, shape='n_partner_types'),
        State('rship_end_dates', default_float, np.nan, shape='n_partner_types'),
        State('n_rships', default_int, 0, shape='n_partner_types'),
    ]

    dates = [State(f'date_{state.name}', default_float, np.nan, shape=state.shape) for state in states if state != 'alive']  # Convert each state into a date

    dates += [
        State('date_clearance', default_float, np.nan, shape='n_genotypes'),
        State('date_next_screen', default_float, np.nan),
    ]

    # Duration of different states: these are floats per person -- used in people.py
    durs = [
        State('dur_none', default_float, np.nan, shape='n_genotypes'), # Length of time that a person has HPV without dysplasia
        State('dur_disease', default_float, np.nan, shape='n_genotypes'), # Length of time that a person has >= HPV present
        State('dur_none2cin1', default_float, np.nan, shape='n_genotypes'), # Length of time to go from no dysplasia to CIN1
        State('dur_cin12cin2', default_float, np.nan, shape='n_genotypes'), # Length of time to go from CIN1 to CIN2
        State('dur_cin22cin3', default_float, np.nan, shape='n_genotypes'), # Length of time to go from CIN2 to CIN3
        State('dur_cin2cancer', default_float, np.nan, shape='n_genotypes'),# Length of time to go from CIN3 to cancer
        State('dur_cancer', default_float, np.nan, shape='n_genotypes'),  # Duration of cancer
    ]

    all_states = person + states + imm_states + intv_states + dates + durs + rship_states

    @classmethod
    def validate(cls):
        """
        Check that states are valid

        This check should be performed when PeopleMeta is consumed (i.e., typically in the People() constructor)
        so that any run-time modifications to the states by the end user get accounted for in validation

        Presently, the only validation check is that the state names are unique, but in principle other
        aspects of the states could be checked too

        :return: None if states are valid
        :raises: ValueError if states are not valid

        """
        # Validate
        state_types = ['person', 'states', 'imm_states', 'intv_states', 'dates', 'durs', 'all_states']
        for state_type in state_types:
            states = getattr(cls, state_type)
            n_states        = len(states)
            n_unique_states = len(set(states))
            if n_states != n_unique_states: # pragma: no cover
                errormsg = f'In {state_type}, only {n_unique_states} of {n_states} state names are unique'
                raise ValueError(errormsg)

        return


#%% Default result settings

# Flows: we count new and cumulative totals for each
# All are stored (1) by genotype and (2) as the total across genotypes
flow_keys   = ['infections',    'cin1s',        'cin2s',        'cin3s',        'cins',         'reinfections', 'reactivations']
flow_names  = ['infections',    'CIN1s',        'CIN2s',        'CIN3s',        'CINs',         'reinfections', 'reactivations']
flow_colors = [pl.cm.GnBu,      pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.GnBu, pl.cm.Purples]

# Stocks: the number in each of the following states
# All are stored (1) by genotype and (2) as the total across genotypes
stock_keys   = ['susceptible',  'infectious',   'none',                 'cin1',         'cin2',         'cin3',         'cin']
stock_names  = ['susceptible',  'infectious',   'without dysplasia',    'with CIN1',    'with CIN2',    'with CIN3',    'with CIN']
stock_colors = [pl.cm.Greens,   pl.cm.GnBu,     pl.cm.GnBu,             pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges]

# Cancer specific flows (not by genotype)
cancer_flow_keys   = ['cancers',  'cancer_deaths', 'detected_cancers', 'detected_cancer_deaths']
cancer_flow_names  = ['cancers',  'cancer deaths', 'detected cancers', 'detected cancer deaths']
cancer_flow_colors = [pl.cm.GnBu, pl.cm.Oranges,    pl.cm.Reds, pl.cm.Greens]

# Incidence and prevalence. Strong overlap with stocks, but with slightly different naming conventions
# All are stored (1) by genotype and (2) as the total across genotypes
inci_keys   = ['hpv',       'cin1',         'cin2',         'cin3',         'cin']
inci_names  = ['HPV',       'CIN1',         'CIN2',         'CIN3',         'CIN']
inci_colors = [pl.cm.GnBu,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges]

# Demographics
dem_keys    = ['births',    'other_deaths']
dem_names   = ['births',    'other deaths']
dem_colors  = ['#fcba03',   '#000000']

# Results by sex
by_sex_keys    = ['total_infections_by_sex',    'other_deaths_by_sex']
by_sex_names   = ['total infections by sex',    'deaths from other causes by sex']
by_sex_colors  = ['#000000',                    '#000000']

# Intervention-related flows (total across genotypes)
intv_flow_keys   = ['screens',  'screened',         'vaccinations', 'vaccinated', ]
intv_flow_names  = ['screens',  'women screened',   'vaccinations', 'women vaccinated']
intv_flow_colors = [pl.cm.GnBu, pl.cm.Oranges,      pl.cm.Oranges,  pl.cm.Oranges]

# Type distributions by cytology
type_keys  = ['none_types', 'cin1_types', 'cin2_types', 'cin3_types', 'cancer_types']
type_names = ['HPV type distribution, normal cytology', 'HPV type distribution, CIN1 lesions', 'HPV type distribution, CIN2 lesions', 'HPV type distribution, CIN3 lesions', 'HPV type distribution, cervical cancer']
type_colors = [pl.cm.GnBu, pl.cm.Oranges, pl.cm.Oranges,  pl.cm.Oranges, pl.cm.Reds]


#%% Default data (age, death rates, birth dates, initial prevalence)

# Default age data, based on population distribution of Kenya in 1990 -- used in population.py
default_age_data = np.array([
    [ 0,  4.9, 0.1900],
    [ 5,  9.9, 0.1645],
    [10, 14.9, 0.1366],
    [15, 19.9, 0.1114],
    [20, 24.9, 0.0886],
    [25, 29.9, 0.0714],
    [30, 34.9, 0.0575],
    [35, 39.9, 0.0459],
    [40, 44.9, 0.0333],
    [45, 49.9, 0.0230],
    [50, 54.9, 0.0205],
    [55, 59.9, 0.0184],
    [60, 64.9, 0.0142],
    [65, 69.9, 0.0104],
    [70, 74.9, 0.0072],
    [75, 79.9, 0.0044],
    [80, 84.9, 0.0021],
    [85, 89.9, 0.0006],
    [90, 99.9, 0.0001],
])


default_death_rates = {1990:{
    'm': np.array([
        [0, 7.2104400e-02],
        [1, 1.0654040e-02],
        [5, 2.8295600e-03],
        [10, 1.9216100e-03],
        [15, 2.7335400e-03],
        [20, 4.0810100e-03],
        [25, 4.8902400e-03],
        [30, 5.9253300e-03],
        [35, 7.4720500e-03],
        [40, 9.3652300e-03],
        [45, 1.1931680e-02],
        [50, 1.5847690e-02],
        [55, 2.0939170e-02],
        [60, 3.0100500e-02],
        [65, 4.2748730e-02],
        [70, 6.1530140e-02],
        [75, 8.9883930e-02],
        [80, 1.3384614e-01],
        [85, 1.9983915e-01],
        [90, 2.8229192e-01],
        [95, 3.8419482e-01],
        [100, 4.9952545e-01]]),
     'f': np.array([
         [0, 6.5018870e-02],
         [1, 9.0851100e-03],
         [5, 2.4186200e-03],
         [10, 1.7122100e-03],
         [15, 2.3409200e-03],
         [20, 3.2310800e-03],
         [25, 4.0792700e-03],
         [30, 4.9329000e-03],
         [35, 6.0179400e-03],
         [40, 7.2600100e-03],
         [45, 8.8378400e-03],
         [50, 1.1686220e-02],
         [55, 1.5708330e-02],
         [60, 2.3382130e-02],
         [65, 3.4809540e-02],
         [70, 5.2215630e-02],
         [75, 7.7168190e-02],
         [80, 1.1523265e-01],
         [85, 1.7457906e-01],
         [90, 2.5035197e-01],
         [95, 3.4646801e-01],
         [100, 4.6195778e-01]
     ])}
}

default_birth_rates = np.array([[
    1960., 1961., 1962., 1963., 1964., 1965., 1966., 1967., 1968., 1969.,
    1970., 1971., 1972., 1973., 1974., 1975., 1976., 1977., 1978., 1979.,
    1980., 1981., 1982., 1983., 1984., 1985., 1986., 1987., 1988., 1989.,
    1990., 1991., 1992., 1993., 1994., 1995., 1996., 1997., 1998., 1999.,
    2000., 2001., 2002., 2003., 2004., 2005., 2006., 2007., 2008., 2009.,
    2010., 2011., 2012., 2013., 2014., 2015., 2016., 2017., 2018., 2019.],
    [51.156, 51.068, 50.976, 50.887, 50.807, 50.748, 50.723, 50.731, 50.768, 50.825,
     50.887, 50.938, 50.958, 50.935, 50.859, 50.732, 50.560, 50.356, 50.125, 49.863,
     49.564, 49.219, 48.817, 48.349, 47.808, 47.171, 46.409, 45.529, 44.560, 43.544,
     42.560, 41.698, 41.015, 40.542, 40.280, 40.196, 40.226, 40.282, 40.289, 40.212,
     40.037, 39.777, 39.468, 39.135, 38.773, 38.366, 37.890, 37.330, 36.678, 35.942,
     35.128, 34.249, 33.333, 32.415, 31.522, 30.688, 29.943, 29.296, 28.748, 28.298]
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
                    'cancer_incidence',
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
