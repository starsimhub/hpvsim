'''
Set the defaults across each of the different files.
'''

import numpy as np
import sciris as sc
import pylab as pl
from .settings import options as hpo # To set options

# Specify all externally visible functions this file defines -- other things are available as e.g. hpv.defaults.default_int
__all__ = ['datadir', 'default_float', 'default_int', 'get_default_plots']

# Define paths
datadir = sc.path(sc.thisdir(__file__)) / 'data'

#%% Specify what data types to use

result_float = np.float64 # Always use float64 for results, for simplicity
if hpo.precision == 32:
    default_float = np.float32
    default_int   = np.int32
elif hpo.precision == 64: # pragma: no cover
    default_float = np.float64
    default_int   = np.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {hpo.precision}')


#%% Define all properties of people

class State(sc.prettyobj):
    def __init__(self, name, dtype, fill_value=0, shape=None, label=None, color=None,
                 totalprefix=None):
        '''
        Args:
            name: name of the result as used in the model (e.g. cin1)
            dtype: datatype
            fill_value: default value for this state upon model initialization
            shape: If not none, set to match a string in `pars` containing the dimensionality e.g., `n_genotypes`)
            label: text used to construct labels for the result for displaying on plots and other outputs
            color: color (used for plotting stocks)
            totalprefix: the prefix used for differentiating by-genotype results from total results. Set to None if the result only appears as a total
        '''
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value
        self.shape = shape
        self.label = label or name
        self.color = color
        self.totalprefix = totalprefix or ('total_' if shape else '')
        return
    
    @property
    def ndim(self):
        return len(sc.tolist(self.shape))+1 # None -> 1, 'n_genotypes' -> 2, etc.
    

    def new(self, pars, n):
        shape = sc.tolist(self.shape) # e.g. convert 'n_genotypes' to ['n_genotypes']
        shape = [pars[s] for s in shape] # e.g. convert ['n_genotypes'] to [2]
        shape.append(n) # We always want to have shape n
        return np.full(shape, dtype=self.dtype, fill_value=self.fill_value)


class PeopleMeta(sc.prettyobj):
    ''' For storing all the keys relating to a person and people '''

    # (attribute, nrows, dtype, default value)
    # If the default value is None, then the array will not be initialized - this is faster and can
    # be used for variables where the People object explicitly supplies the values e.g. age

    # Set the properties of a person
    person = [
        State('uid',            default_int),           # Int
        State('scale',          default_float,  1.0), # Float
        State('level0',         bool,  True), # "Normal" people
        State('level1',         bool,  False), # "High-resolution" people: e.g. cancer agents
        State('age',            default_float,  np.nan), # Float
        State('sex',            default_float,  np.nan), # Float
        State('debut',          default_float,  np.nan), # Float
        State('doses',          default_int,    0),  # Number of doses of the prophylactic vaccine given per person
        State('txvx_doses',     default_int,    0),  # Number of doses of the therapeutic vaccine given per person
        State('vaccine_source', default_int,    -1), # Index of the prophylactic vaccine that individual received
        State('screens',        default_int,    0),  # Number of screens given per person
        State('cin_treatments', default_int,    0),  # Number of CIN treatments given per person
        State('cancer_treatments', default_int,    0),  # Number of cancer treatments given per person
        State('art_adherence',  default_float, 0, label='adherence on ART', color='#aaa8ff')
    ]

    # Set the states that a person can be in
    # The following three groupings are all mutually exclusive and collectively exhaustive.
    alive_states = [
        # States related to whether or not the person is alive or dead
        State('alive',          bool,   True,   label='Population'),    # Save this as a state so we can record population sizes
        State('dead_cancer',    bool,   False,  label='Cumulative cancer deaths'),   # Dead from cancer
        State('dead_other',     bool,   False,  label='Cumulative deaths from other causes'),   # Dead from all other causes
        State('emigrated',      bool,   False,  label='Emigrated'),  # Emigrated
    ]

    viral_states = [
        # States related to whether virus is present
        # From these, we calculate the following additional derived states:
        #       1. 'infected' (union of infectious and inactive)
        State('susceptible',    bool, True,     'n_genotypes', label='Number susceptible', color='#4d771e'),               # Allowable dysp states: no_dysp
        State('infectious',     bool, False,    'n_genotypes', label='Number infectious',  color='#c78f65'),               # Allowable dysp states: no_dysp, cin1, cin2, cin3
        State('inactive',       bool, False,    'n_genotypes', label='Number with inactive infection', color='#9e1149'),   # Allowable dysp states: no_dysp, cancer in at least one genotype
    ]

    dysp_states = [
        # States related to whether or not cervical dysplasia is present.
        # From these and the viral_states, we derive the following additional states:
        #       1. 'cin' (union of cin1, cin2, cin3)
        #       2. 'precin' (intersection of infectious and no_dysp agents - agents with infection but not dysplasia)
        #       3. 'latent' (intersection of inactive and no_dysp - agents with latent infection)
        State('no_dysp',        bool, True,  'n_genotypes', label='Number without dyplasia', color='#9e1149'), # Allowable viral states: susceptible, infectious, and inactive
        State('cin1',           bool, False, 'n_genotypes', label='Number with CIN1', color='#c1ad71'),        # Allowable viral states: infectious
        State('cin2',           bool, False, 'n_genotypes', label='Number with CIN2', color='#c1981d'),        # Allowable viral states: infectious
        State('cin3',           bool, False, 'n_genotypes', label='Number with CIN3', color='#b86113'),        # Allowable viral states: infectious
        State('cancerous',      bool, False, 'n_genotypes', label='Number with cancer', color='#5f5cd2'),      # Allowable viral states: inactive
    ]

    derived_states = [
        State('infected',   bool, False, 'n_genotypes', label='Number infected', color='#c78f65'), # union of infectious and inactive. Includes people with cancer, people with latent infections, and people with active infections
        State('cin',        bool, False, 'n_genotypes', label='Number with dysplasia', color='#c1ad71'), # union of cin1, cin2, cin3
        State('precin',     bool, False, 'n_genotypes', label='Number with active infection and no dysplasia', color='#9e1149'), # intersection of no_dysp and infectious. Includes people with transient infections that will clear on their own plus those where dysplasia isn't established yet
        State('latent',     bool, False, 'n_genotypes', label='Number with latent infection', color='#9e1149'), # intersection of no_dysp and inactive.
    ]

    hiv_states = [
        State('hiv',        bool, False, label='Number infected with HIV', color='#5c399c'),
    ]

    # Additional intervention states
    intv_states = [
        State('detected_cancer',    bool,   False, label='Number with detected cancer'), # Whether the person's cancer has been detected
        State('screened',           bool,   False, label='Number screend'), # Whether the person has been screened (how does this change over time?)
        State('cin_treated',        bool,   False, label='Number treated for precancerous lesions'), # Whether the person has been treated for CINs
        State('cancer_treated',     bool,   False, label='Number treated for cancer'), # Whether the person has been treated for cancer
        State('vaccinated',         bool,   False, label='Number vaccinated'), # Whether the person has received the prophylactic vaccine
        State('tx_vaccinated',      bool,   False, label='Number given therapeutic vaccine'), # Whether the person has received the therapeutic vaccine
    ]

    # Collection of mutually exclusive + collectively exhaustive states
    mece_states = alive_states + viral_states + dysp_states

    # Collection of states that we store as stock results
    stock_states = viral_states + dysp_states + derived_states + intv_states + hiv_states

    # Set dates
    # Convert each MECE state and derived state into a date except for susceptible, alive, and no_dysp (which are True by default)
    dates = [State(f'date_{state.name}', default_float, np.nan, shape=state.shape) for state in mece_states+derived_states+intv_states if not state.fill_value]

    # Immune states, by genotype/vaccine
    imm_states = [
        State('sus_imm',        default_float,  0,'n_imm_sources'),  # Float, by genotype
        State('peak_imm',       default_float,  0,'n_imm_sources'),  # Float, peak level of immunity
        State('imm',            default_float,  0,'n_imm_sources'),  # Float, current immunity level
        State('t_imm_event',    default_int,    0,'n_imm_sources'),  # Int, time since immunity event
    ]

    # Relationship states
    rship_states = [
        State('rship_start_dates', default_float, np.nan, shape='n_partner_types'),
        State('rship_end_dates', default_float, np.nan, shape='n_partner_types'),
        State('n_rships', default_int, 0, shape='n_partner_types'),
        State('partners', default_float, np.nan, shape='n_partner_types'),  # Int by relationship type
        State('current_partners', default_float, 0, 'n_partner_types'),  # Int by relationship type
    ]

    dates += [
        State('date_clearance',     default_float, np.nan, shape='n_genotypes'),
        State('date_exposed',       default_float, np.nan, shape='n_genotypes'),
    ]

    # Duration of different states: these are floats per person -- used in people.py
    durs = [
        State('dur_infection', default_float, np.nan, shape='n_genotypes'), # Length of time that a person has any HPV present
        State('dur_precin', default_float, np.nan, shape='n_genotypes'), # Length of time that a person has HPV without dysplasia
        State('dur_cancer', default_float, np.nan, shape='n_genotypes'),  # Duration of cancer
    ]

    # Markers of disease severity
    sev = [
        State('dysp_rate', default_float, np.nan, shape='n_genotypes'), # Parameter in a logistic function that maps duration of initial infection to the probability of developing dysplasia
    ]


    all_states = person + mece_states + imm_states + hiv_states + intv_states + dates + durs + rship_states + sev

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
        state_types = ['person', 'mece_states', 'imm_states', 'hiv_states', 'intv_states', 'dates', 'durs', 'sev', 'all_states']
        for state_type in state_types:
            states = getattr(cls, state_type)
            n_states        = len(states)
            n_unique_states = len(set(states))
            if n_states != n_unique_states: # pragma: no cover
                errormsg = f'In {state_type}, only {n_unique_states} of {n_states} state names are unique'
                raise ValueError(errormsg)

        return


#%% Default result settings

# Flows
# All are stored (1) by genotype and (2) as the total across genotypes
class Flow():
    def __init__(self, name, label=None, color=None, by_genotype=True):
        self.name = name
        self.label = label or name
        self.color = color
        self.by_genotype = by_genotype

flows = [
    Flow('infections',              color='#c78f65',    label='Infections'),
    Flow('cin1s',                   color='#c1ad71',    label='CIN1s'),
    Flow('cin2s',                   color='#c1981d',    label='CIN2s'),
    Flow('cin3s',                   color='#b86113',    label='CIN3s'),
    Flow('cins',                    color='#c1ad71',    label='CINs'),
    Flow('cancers',                 color='#5f5cd2',    label='Cancers'),
    Flow('detected_cancers',        color='#5f5cd2',    label='Cancer detections', by_genotype=False),
    Flow('cancer_deaths',           color='#000000',    label='Cancer deaths', by_genotype=False),
    Flow('detected_cancer_deaths',  color='#000000',    label='Detected cancer deaths', by_genotype=False),
    Flow('reinfections',            color='#732e26',    label='Reinfections'),
    Flow('reactivations',           color='#732e26',    label='Reactivations'),
    Flow('hiv_infections',          color='#5c399c',    label='HIV infections', by_genotype=False)
]
flow_keys           = [flow.name for flow in flows]
genotype_flow_keys  = [flow.name for flow in flows if flow.by_genotype]

# Stocks: the number in each of the following states
# All are stored (1) by genotype and (2) as the total across genotypes
stock_keys   = [state.name for state in PeopleMeta.stock_states]
stock_names  = [state.label for state in PeopleMeta.stock_states]
stock_colors = [state.color for state in PeopleMeta.stock_states]
total_stock_keys = [state.name for state in PeopleMeta.stock_states if state.shape=='n_genotypes']
other_stock_keys = [state.name for state in PeopleMeta.intv_states+PeopleMeta.hiv_states]

# Incidence. Strong overlap with stocks, but with slightly different naming conventions
# All are stored (1) by genotype and (2) as the total across genotypes
inci_keys   = ['hpv',       'cin1',     'cin2',     'cin3',     'cin',      'cancer']
inci_names  = ['HPV',       'CIN1',     'CIN2',     'CIN3',     'CIN',      'Cancer']
inci_colors = ['#c78f65',   '#c1ad71',  '#c1981d',  '#b86113',  '#c1ad71',  '#5f5cd2']

# Demographics
dem_keys    = ['births',    'other_deaths', 'migration']
dem_names   = ['births',    'other deaths', 'migration']
dem_colors  = ['#fcba03',   '#000000',      '#000000']

# Results by sex
by_sex_keys    = ['infections_by_sex',    'other_deaths_by_sex']
by_sex_names   = ['infections by sex',    'deaths from other causes by sex']
by_sex_colors  = ['#000000',                    '#000000']

# Results for storing type distribution by dysplasia
type_dysp_keys   = ['n_precin', 'n_cin1', 'n_cin2', 'n_cin3', 'n_cancerous']
type_dysp_names  = ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']


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
    'infections',
    'cins',
    'cancers',
]

class plot_args():
    ''' Mini class for defining default plot specifications '''
    def __init__(self, keys, name=None, plot_type=None, year=None):
        self.keys = sc.tolist(keys)
        self.name = name
        self.plot_type = plot_type
        self.year = year
        return


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
            plots = sc.objdict({
                'HPV prevalence': 'hpv_prevalence',
                'CIN incidence (per 100,000 women)': 'cin_incidence',
                'Cancer incidence (per 100,000 women)': ['cancer_incidence', 'asr_cancer_incidence'],
                'Infections by age': 'infections_by_age',
                'Cancers by age': 'cancers_by_age',
                'HPV types by cytology': 'type_dysp',
            })

        else: # pragma: no cover
            plots = sc.odict({
                'HPV incidence': [
                    'hpv_prevalence',
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
