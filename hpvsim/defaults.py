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

    def __init__(self):

        # Set the properties of a person
        self.person = [
            State('uid',            default_int),           # Int
            State('scale',          default_float,  1.0), # Float
            State('level0',         bool,  True), # "Normal" people
            State('level1',         bool,  False), # "High-resolution" people: e.g. cancer agents
            State('age',            default_float,  np.nan), # Float
            State('sex',            default_float,  np.nan), # Float
            State('debut',          default_float,  np.nan), # Float
            State('sev',            default_float, np.nan, shape='n_genotypes'), # Severity of infection, taking values between 0-1
            State('rel_sev',        default_float, 1.0), # Individual relative risk for rate severe disease growth (does not vary by genotype)
            State('rel_sus',        default_float, 1.0), # Individual relative risk for acquiring infection (does not vary by genotype)
            State('rel_imm',        default_float, 1.0), # Individual relative level of immunity acquired from infection clearance/vaccination
            State('doses',          default_int,    0),  # Number of doses of the prophylactic vaccine given per person
            State('txvx_doses',     default_int,    0),  # Number of doses of the therapeutic vaccine given per person
            State('vaccine_source', default_int,    -1), # Index of the prophylactic vaccine that individual received
            State('screens',        default_int,    0),  # Number of screens given per person
            State('cin_treatments', default_int,    0),  # Number of CIN treatments given per person
            State('cancer_treatments', default_int,    0),  # Number of cancer treatments given per person
            State('art_adherence',  default_float, 0, label='adherence on ART', color='#aaa8ff')
        ]

        ###### The following section consists of all the boolean states

        # The following three groupings are all mutually exclusive and collectively exhaustive.
        self.alive_states = [
            # States related to whether or not the person is alive or dead
            State('alive',          bool,   True,   label='Population'),    # Save this as a state so we can record population sizes
            State('dead_cancer',    bool,   False,  label='Cumulative cancer deaths'),   # Dead from cancer
            State('dead_other',     bool,   False,  label='Cumulative deaths from other causes'),   # Dead from all other causes
            State('emigrated',      bool,   False,  label='Emigrated'),  # Emigrated
        ]

        self.viral_states = [
            # States related to whether virus is present
            State('susceptible',    bool, True,     'n_genotypes', label='Number susceptible', color='#4d771e'),               # Allowable dysp states: no_dysp
            State('infectious',     bool, False,    'n_genotypes', label='Number infectious',  color='#c78f65'),               # Allowable dysp states: no_dysp, cin1, cin2, cin3
            State('inactive',       bool, False,    'n_genotypes', label='Number with inactive infection', color='#9e1149'),   # Allowable dysp states: no_dysp, cancer in at least one genotype
        ]

        self.cell_states = [
            # States related to the cellular changes present in the cervix.
            State('normal',         bool, True, 'n_genotypes', label='Number with no cellular changes', color='#9e1149'), # Allowable viral states: susceptible, infectious, and inactive
            State('episomal',       bool, False, 'n_genotypes', label='Number with episomal infection', color='#9e1149'), # Allowable viral states: susceptible, infectious, and inactive
            State('transformed',    bool, False, 'n_genotypes', label='Number with transformation', color='#9e1149'), # Allowable viral states: susceptible, infectious, and inactive
            State('cancerous',      bool, False, 'n_genotypes', label='Number with cancer', color='#5f5cd2'),      # Allowable viral states: inactive
        ]

        self.derived_states = [
            # From the viral states, cell states, and severity markers, we derive the following additional states:
            State('infected',   bool, False, 'n_genotypes', label='Number infected', color='#c78f65'), # Union of infectious and inactive. Includes people with cancer, people with latent infections, and people with active infections
            State('abnormal',   bool, False, 'n_genotypes', label='Number with abnormal cells', color='#9e1149'),  # Union of episomal, transformed, and cancerous. Allowable viral states: infectious
            State('latent',     bool, False, 'n_genotypes', label='Number with latent infection', color='#5f5cd2'), # Intersection of normal and inactive.
            State('precin',     bool, False, 'n_genotypes', label='Number with precin', color='#9e1149'), # Defined as those with sev < clinical_cuttoff[0]
            State('cin1',       bool, False, 'n_genotypes', label='Number with cin1', color='#9e1149'), # Defined as those with clinical_cuttoff[0] < sev < clinical_cuttoff[1]
            State('cin2',       bool, False, 'n_genotypes', label='Number with cin2', color='#9e1149'), # Defined as those with clinical_cuttoff[1] < sev < clinical_cuttoff[2]
            State('cin3',       bool, False, 'n_genotypes', label='Number with cin3', color='#5f5cd2'), # Defined as those with clinical_cuttoff[2] < sev < clinical_cuttoff[3]
            State('carcinoma',  bool, False, 'n_genotypes', label='Number with carcinoma in situ', color='#5f5cd2'), # Defined as those with clinical_cuttoff[3] < sev < clinical_cuttoff[4]
            State('cin',        bool, False, 'n_genotypes', label='Number with detectable dysplasia', color='#5f5cd2'), # Union of CIN1, CIN3, CIN3, and carcinoma in situ
        ]

        # Additional intervention states
        self.intv_states = [
            State('detected_cancer',    bool,   False, label='Number with detected cancer'), # Whether the person's cancer has been detected
            State('screened',           bool,   False, label='Number screened'), # Whether the person has been screened (how does this change over time?)
            State('cin_treated',        bool,   False, label='Number treated for precancerous lesions'), # Whether the person has been treated for CINs
            State('cancer_treated',     bool,   False, label='Number treated for cancer'), # Whether the person has been treated for cancer
            State('vaccinated',         bool,   False, label='Number vaccinated'), # Whether the person has received the prophylactic vaccine
            State('tx_vaccinated',      bool,   False, label='Number given therapeutic vaccine'), # Whether the person has received the therapeutic vaccine
        ]

        # Any other stock states - add a placeholder here to be populated later
        self.other_stock_states = []

        # Immune states, by genotype/vaccine
        self.imm_states = [
            State('sus_imm',        default_float,  0,'n_imm_sources'),  # Float, by genotype
            State('peak_imm',       default_float,  0,'n_imm_sources'),  # Float, peak level of immunity
            State('nab_imm',        default_float,  0,'n_imm_sources'),  # Float, current immunity level
            State('t_imm_event',    default_int,    0,'n_imm_sources'),  # Int, time since immunity event
            State('cell_imm',       default_float,  0,'n_imm_sources'),
        ]

        # Relationship states
        self.rship_states = [
            State('rship_start_dates', default_float, np.nan, shape='n_partner_types'),
            State('rship_end_dates', default_float, np.nan, shape='n_partner_types'),
            State('n_rships', default_int, 0, shape='n_partner_types'),
            State('partners', default_float, np.nan, shape='n_partner_types'),  # Int by relationship type
            State('current_partners', default_float, 0, 'n_partner_types'),  # Int by relationship type
        ]

        # Duration of different states: these are floats per person -- used in people.py
        self.durs = [
            State('dur_infection',      default_float, np.nan, shape='n_genotypes'), # Length of time that a person has any HPV present. Defined for males and females. For females, dur_infection = dur_episomal + dur_transformed. For males, it's taken from a separate distribution
            State('dur_precin',         default_float, np.nan, shape='n_genotypes'), # Length of time that a person has HPV prior to precancerous changes
            State('dur_cin',            default_float, np.nan, shape='n_genotypes'), # Length of time that a person has precancerous changes
            State('dur_episomal',       default_float, np.nan, shape='n_genotypes'), # Length of time that a person has episomal HPV
            State('dur_transformed',    default_float, np.nan, shape='n_genotypes'), # Length of time that a person has transformed HPV
            State('dur_cancer',         default_float, np.nan, shape='n_genotypes'), # Duration of cancer
        ]

    # Collection of mutually exclusive + collectively exhaustive states
    @property
    def mece_states(self):
        return self.alive_states + self.viral_states + self.cell_states

    # Collection of states that we store as stock results
    @property
    def stock_states(self):
        return self.viral_states + self.cell_states + self.derived_states + self.intv_states + self.other_stock_states

    # Collection of states for which we store associated dates
    @property
    def date_states(self):
        return [state for state in self.alive_states + self.stock_states if not state.fill_value]

    # Set dates
    @property
    def dates(self):
        ''' Dates are stored for all states except susceptible, and alive, and normal (which are True by default) '''
        dates = [State(f'date_{state.name}', default_float, np.nan, shape=state.shape) for state in self.date_states]
        dates += [
            State('date_clearance', default_float, np.nan, shape='n_genotypes'),
            State('date_exposed', default_float, np.nan, shape='n_genotypes'),
        ]
        return dates

    # All states
    @property
    def all_states(self):
        return self.person + self.alive_states + self.viral_states + self.cell_states + self.derived_states + self.intv_states + self.other_stock_states + self.imm_states + self.rship_states + self.durs

    # States to set - same as above but does not include derived states
    @property
    def states_to_set(self):
        return self.person + self.alive_states + self.viral_states + self.cell_states + self.intv_states + self.other_stock_states + self.imm_states + self.rship_states + self.durs + self.dates

    @property
    def stock_keys(self):
        return [state.name for state in self.stock_states]

    @property
    def stock_names(self):
        return [state.label for state in self.stock_states]

    @property
    def stock_colors(self):
        return [state.color for state in self.stock_states]

    @property
    def genotype_stock_keys(self):
        return [state.name for state in self.stock_states if state.shape=='n_genotypes']

    @property
    def other_stock_keys(self):
        return [state.name for state in self.other_stock_states]

    @property
    def intv_stock_keys(self):
        return [state.name for state in self.intv_states]


    def validate(self):
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
        state_types = ['person', 'mece_states', 'imm_states', 'intv_states', 'dates', 'durs', 'all_states']
        for state_type in state_types:
            states = getattr(self, state_type)
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
    Flow('dysplasias',              color='#c1ad71',    label='Dysplasias'),
    Flow('precins',                 color='#c1ad71',    label='Pre-CINs'),
    Flow('cin1s',                   color='#c1ad71',    label='CIN1s'),
    Flow('cin2s',                   color='#c1981d',    label='CIN2s'),
    Flow('cin3s',                   color='#b86113',    label='CIN3s'),
    Flow('cins',                    color='#b86113',    label='CINs'),
    Flow('cancers',                 color='#5f5cd2',    label='Cancers'),
    Flow('detected_cancers',        color='#5f5cd2',    label='Cancer detections', by_genotype=False),
    Flow('cancer_deaths',           color='#000000',    label='Cancer deaths', by_genotype=False),
    Flow('detected_cancer_deaths',  color='#000000',    label='Detected cancer deaths', by_genotype=False),
    Flow('reinfections',            color='#732e26',    label='Reinfections'),
    Flow('reactivations',           color='#732e26',    label='Reactivations'),
]
flow_keys           = [flow.name for flow in flows]
genotype_flow_keys  = [flow.name for flow in flows if flow.by_genotype]


# Incidence. Strong overlap with stocks, but with slightly different naming conventions
# All are stored (1) by genotype and (2) as the total across genotypes
inci_keys   = ['hpv',       'cin1',     'cin2',     'cin3',     'dysplasia',      'cancer']
inci_names  = ['HPV',       'CIN1',     'CIN2',     'CIN3',     'Dysplasia',      'Cancer']
inci_colors = ['#c78f65',   '#c1ad71',  '#c1981d',  '#b86113',  '#c1ad71',  '#5f5cd2']

# Demographics
dem_keys    = ['births',    'other_deaths', 'migration']
dem_names   = ['births',    'other deaths', 'migration']
dem_colors  = ['#fcba03',   '#000000',      '#000000']

# Results by sex
by_sex_keys    = ['infections_by_sex',    'other_deaths_by_sex']
by_sex_names   = ['infections by sex',    'deaths from other causes by sex']
by_sex_colors  = ['#000000',              '#000000']

# Results for storing type distribution by dysplasia
type_dist_keys   = ['precin', 'cin1', 'cin2', 'cin3', 'cancerous']
type_dist_names  = ['Pre-CIN', 'CIN1', 'CIN2', 'CIN3', 'Cancer']

#%% Default initial prevalence

default_init_prev = {
    'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
    'm'             : np.array([ 0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
    'f'             : np.array([ 0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
}


#%% Default plotting settings

# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
    'infections',
    'dysplasias',
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
                'HPV infections by age': 'infections_by_age',
                'HPV prevalence': ['hpv_prevalence_by_genotype'],
                'Pre-cancer prevalence by age': ['precin_prevalence_by_age', 'cin1_prevalence_by_age', 'cin2_prevalence_by_age', 'cin3_prevalence_by_age'],
                'Cancer incidence (per 100,000 women)': ['cancer_incidence', 'asr_cancer_incidence'],
                'Cancers by age': 'cancers_by_age',
                'HPV type distribution': 'type_dist',
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
