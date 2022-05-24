'''
Set the parameters for hpvsim.
'''

import numpy as np
import sciris as sc
from .settings import options as hpo # For setting global options
from . import misc as hpm
from . import defaults as hpd
from .data import loaders as hpdata

__all__ = ['make_pars', 'reset_layer_pars', 'get_prognoses']


def make_pars(version=None, nonactive_by_age=False, set_prognoses=False, **kwargs):
    '''
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = hp.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        version       (str):  if supplied, use parameters from this version
        kwargs        (dict): any additional kwargs are interpreted as parameter names

    Returns:
        pars (dict): the parameters of the simulation
    '''
    pars = {}

    # Population parameters
    pars['pop_size']        = 20e3      # Number of agents
    pars['network']         = 'random'  # What type of sexual network to use -- 'random', 'basic', other options TBC
    pars['location']        = None      # What location to load data from -- default Seattle
    pars['death_rates']     = None      # Deaths from all other causes, loaded below
    pars['birth_rates']     = None      # Birth rates, loaded below

    # Initialization parameters
    pars['init_hpv_prev']       = hpd.default_init_prev # Initial prevalence

    # Simulation parameters
    pars['start']           = 2015.         # Start of the simulation
    pars['end']             = None          # End of the simulation
    pars['n_years']         = 10.           # Number of years to run, if end isn't specified
    pars['dt']              = 0.2           # Timestep (in years)
    pars['rand_seed']       = 1             # Random seed, if None, don't reset
    pars['verbose']         = hpo.verbose   # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)

    # Network parameters, generally initialized after the population has been constructed
    pars['debut']           = dict(f=dict(dist='normal', par1=18.6, par2=2.1), # Location-specific data should be used here if possible
                                   m=dict(dist='normal', par1=19.6, par2=1.8))
    pars['partners']        = None  # The number of concurrent sexual partners for each partnership type
    pars['acts']            = None  # The number of sexual acts for each partnership type per year
    pars['condoms']         = None  # The proportion of acts in which condoms are used for each partnership type
    pars['layer_probs']     = None  # Proportion of the population in each partnership type
    pars['dur_pship']       = None  # Duration of partnerships in each partnership type
    pars['mixing']          = None  # Mixing matrices for storing age differences in partnerships - TODO
    pars['n_partner_types'] = 1  # Number of partnership types - reset below
    # pars['nonactive_by_age']= nonactive_by_age
    # pars['nonactive']       = None 

    # Basic disease transmission parameters
    pars['beta_dist']       = dict(dist='neg_binomial', par1=1.0, par2=1.0, step=0.01) # Distribution to draw individual level transmissibility TODO does this get used? if not remove.
    pars['beta']            = 0.35  # Per-act transmission probability; absolute value, calibrated

    # Probabilities of disease progression
    pars['rel_cin1_prob'] = 1.0  # Scale factor for proportion of CIN cases
    pars['rel_cin2_prob'] = 1.0  # Scale factor for proportion of CIN cases
    pars['rel_cin3_prob'] = 1.0  # Scale factor for proportion of CIN cases
    pars['rel_cancer_prob'] = 1.0  # Scale factor for proportion of CIN that develop into cancer
    pars['rel_death_prob'] = 1.0  # Scale factor for proportion of cancer cases that result in death
    pars['prognoses'] = None # Arrays of prognoses by duration; this is populated later

    # Parameters used to calculate immunity
    pars['imm_init'] = dict(dist='beta', par1=20, par2=1)  # beta distribution for initial level of immunity following infection clearance
    pars['imm_decay'] = dict(infection=dict(form='exp_decay', init_val=1, half_life=10), # decay rate, with half life in YEARS
                             vaccine=dict(form='exp_decay', init_val=1, half_life=20)) # decay rate, with half life in YEARS
    pars['imm_kin'] = None  # Constructed during sim initialization using the nab_decay parameters
    pars['imm_boost'] = 1.5  # Multiplicative factor applied to a person's immunity levels if they get reinfected. No data on this, assumption.
    pars['immunity'] = None  # Matrix of immunity and cross-immunity factors, set by init_immunity() in immunity.py
    pars['immunity_map'] = None  # dictionary mapping the index of immune source to the type of immunity (vaccine vs natural)

    # all genotype properties get populated by user in init_genotypes()
    pars['genotypes'] = []  # Genotypes of the virus; populated by the user below
    pars['genotype_map'] = dict()  # Reverse mapping from number to genotype key
    pars['genotype_pars'] = dict()  # Populated just below

    # Genotype parameters
    pars['n_genotypes'] = 1 # The number of genotypes circulating in the population

    # Parameters determining duration of dysplasia stages
    pars['dur'] = {}
    pars['dur']['none']     = dict(dist='lognormal', par1=2.0, par2=1.0)  # Length of time that HPV is present without dysplasia
    pars['dur']['cin1']     = dict(dist='lognormal', par1=2.0, par2=1.0)  # Duration of CIN1 (mild/very mild dysplasia)
    pars['dur']['cin2']     = dict(dist='lognormal', par1=3.0, par2=1.0)  # Duration of CIN2 (moderate dysplasia)
    pars['dur']['cin3']     = dict(dist='lognormal', par1=4.0, par2=1.0)  # Duration of CIN3 (severe dysplasia/in situ carcinoma)
    pars['dur']['cancer']   = dict(dist='lognormal', par1=6.0, par2=3.0)  # Duration of untreated cancer

    # Efficacy of protection
    pars['eff_condoms']     = 0.8  # The efficacy of condoms; assumption; TODO replace with data

    # Events and interventions
    pars['interventions'] = []   # The interventions present in this simulation; populated by the user
    pars['analyzers']     = []   # Custom analysis functions; populated by the user
    pars['timelimit']     = None # Time limit for the simulation (seconds)
    pars['stopping_func'] = None # A function to call to stop the sim partway through

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    reset_layer_pars(pars)
    if set_prognoses: # If not set here, gets set when the population is initialized
        pars['prognoses'] = get_prognoses() # Default to duration-specific prognoses

    return pars


# Define which parameters need to be specified as a dictionary by layer -- define here so it's available at the module level for sim.py
layer_pars = ['partners', 'acts', 'layer_probs', 'dur_pship', 'condoms']


def reset_layer_pars(pars, layer_keys=None, force=False):
    '''
    Helper function to set layer-specific parameters. If layer keys are not provided,
    then set them based on the population type. This function is not usually called
    directly by the user, although it can sometimes be used to fix layer key mismatches
    (i.e. if the contact layers in the population do not match the parameters). More
    commonly, however, mismatches need to be fixed explicitly.

    Args:
        pars (dict): the parameters dictionary
        layer_keys (list): the layer keys of the population, if available
        force (bool): reset the parameters even if they already exist
    '''

    layer_defaults = {}
    # Specify defaults for random -- layer 'a' for 'all'
    layer_defaults['random'] = dict(
        partners    = dict(a=1),    # Default number of concurrent sexual partners; TODO make this a distribution and incorporate zero inflation
        acts        = dict(a=dict(dist='neg_binomial', par1=100,par2=50)),  # Default number of sexual acts per year
        layer_probs = dict(a=1.0),  # Default proportion of the population in each layer
        dur_pship   = dict(a=dict(dist='normal_pos', par1=5,par2=3)),    # Default duration of partnerships
        condoms     = dict(a=0.25),  # Default proportion of acts in which condoms are used
    )

    # Specify defaults for basic sexual network with regular and casual partners
    layer_defaults['basic'] = dict(
        partners    = dict(r=1, c=2),       # Default number of concurrent sexual partners; TODO make this a distribution and incorporate zero inflation
        acts        = dict(r=dict(dist='neg_binomial', par1=80, par2=40),
                           c=dict(dist='neg_binomial', par1=10, par2=5)),
        layer_probs = dict(r=0.7, c=0.4),   # Default proportion of the population in each layer
        dur_pship   = dict(r=dict(dist='normal_pos', par1=10,par2=3),
                           c=dict(dist='normal_pos', par1=2, par2=1)),
        condoms     = dict(r=0.01, c=0.8),  # Default proportion of acts in which condoms are used
    )

    # Choose the parameter defaults based on the population type, and get the layer keys
    try:
        defaults = layer_defaults[pars['network']]
    except Exception as E:
        errormsg = f'Cannot load defaults for population type "{pars["network"]}"'
        raise ValueError(errormsg) from E
    default_layer_keys = list(defaults['acts'].keys()) # All layers should be the same, but use beta_layer for convenience

    # Actually set the parameters
    for pkey in layer_pars:
        par = {} # Initialize this parameter
        default_val = layer_defaults['random'][pkey]['a'] # Get the default value for this parameter

        # If forcing, we overwrite any existing parameter values
        if force:
            par_dict = defaults[pkey] # Just use defaults
        else:
            par_dict = sc.mergedicts(defaults[pkey], pars.get(pkey, None)) # Use user-supplied parameters if available, else default

        # Figure out what the layer keys for this parameter are (may be different between parameters)
        if layer_keys:
            par_layer_keys = layer_keys # Use supplied layer keys
        else:
            par_layer_keys = list(sc.odict.fromkeys(default_layer_keys + list(par_dict.keys())))  # If not supplied, use the defaults, plus any extra from the par_dict; adapted from https://www.askpython.com/python/remove-duplicate-elements-from-list-python

        # Construct this parameter, layer by layer
        for lkey in par_layer_keys: # Loop over layers
            par[lkey] = par_dict.get(lkey, default_val) # Get the value for this layer if available, else use the default for random
        pars[pkey] = par # Save this parameter to the dictionary

    # Finally, update the number of partnership types
    pars['n_partner_types'] = len(par_layer_keys)

    return


def get_prognoses():
    '''
    Return the default parameter values for prognoses

    The prognosis probabilities are conditional given the previous disease state.

    Returns:
        prog_pars (dict): the dictionary of prognosis probabilities
    '''

    prognoses = dict(
        duration_cutoffs  = np.array([0,       1,          2,          5,          10]),     # Duration cutoffs (lower limits)
        cin1_probs        = np.array([0.015,  0.05655,    0.10800,    0.50655,    0.70]),   # Conditional probability of developing CIN1 given HPV infection
        cin2_probs        = np.array([0.015,  0.0655,    0.1080,    0.60655,    0.90]),   # Conditional probability of developing CIN2 given CIN1
        cin3_probs        = np.array([0.15,  0.655,    0.80,    0.855,    0.90]),   # Conditional probability of developing CIN3 given CIN2
        cancer_probs      = np.array([0.0055,  0.0655,    0.2080,    0.50655,    0.90]),   # Conditional probability of developing cancer given CIN3
        death_probs       = np.array([0.0015,  0.00655,    0.02080,    0.20655,    0.70]),   # Conditional probability of dying from cancer given cancer
        )

    # Check that lengths match
    expected_len = len(prognoses['duration_cutoffs'])
    for key,val in prognoses.items():
        this_len = len(prognoses[key])
        if this_len != expected_len: # pragma: no cover
            errormsg = f'Lengths mismatch in prognoses: {expected_len} duration bins specified, but key "{key}" has {this_len} entries'
            raise ValueError(errormsg)

    return prognoses


def get_births_deaths(location=None, verbose=1, by_sex=True, overall=False, die=None):
    '''
    Get mortality and fertility data by location if provided, or use default

    Args:
        location (str):  location; if none specified, use default value for XXX
        verbose (bool):  whether to print progress
        by_sex   (bool): whether to get sex-specific death rates (default true)
        overall  (bool): whether to get overall values ie not disaggregated by sex (default false)

    Returns:
        death_rates (dict): nested dictionary of death rates by sex (first level) and age (second level)
        birth_rates (arr): array of crude birth rates by year
    '''

    birth_rates = hpd.default_birth_rates 
    death_rates = hpd.default_death_rates
    if location is not None:
        if verbose:
            print(f'Loading location-specific demographic data for "{location}"')
        try:
            death_rates = hpdata.get_death_rates(location=location, by_sex=by_sex, overall=overall)
            birth_rates = hpdata.get_birth_rates(location=location)
        except ValueError as E:
            warnmsg = f'Could not load demographic data for requested location "{location}" ({str(E)}), using default'
            hpm.warn(warnmsg, die=die)
    
    return birth_rates, death_rates

#%% Genotype/immunity parameters and functions

def get_hpv_prevalence():
    '''
    Get HPV prevalence data by age and genotype for initializing the sim

    Args:
        filename (str):  filename; if none specified, use default value for XXX

    Returns:
        hpv_prevalence (dict): nested dictionary of hpv prevalence by sex (first level),  age (second level), and genotype (third level)
    '''

    hpv_prevalence = hpd.default_hpv_prevalence

    return hpv_prevalence

def get_genotype_choices():
    '''
    Define valid genotype names
    '''
    # List of choices available
    choices = {
        'hpv16':  ['hpv16', '16'],
        'hpv18': ['hpv18', '18'],
        'hpv6':  ['hpv6', '6'],
        'hpv11': ['hpv11', '11'],
        'hpv31': ['hpv31', '31'],
        'hpv33': ['hpv33', '33'],
        'hpv45': ['hpv45', '45'],
        'hpv52': ['hpv52', '52'],
        'hpv58': ['hpv58', '58'],
        'hpvlo': ['hpvlo', 'low', 'low-risk'],
        'hpvhi': ['hpvhi', 'high', 'high-risk'],
        'hpvhi5': ['hpvhi5', 'high5'],
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping


def _get_from_pars(pars, default=False, key=None, defaultkey='default'):
    ''' Helper function to get the right output from genotype functions '''

    # If a string was provided, interpret it as a key and swap
    if isinstance(default, str):
        key, default = default, key

    # Handle output
    if key is not None:
        try:
            return pars[key]
        except Exception as E:
            errormsg = f'Key "{key}" not found; choices are: {sc.strjoin(pars.keys())}'
            raise sc.KeyNotFoundError(errormsg) from E
    elif default:
        return pars[defaultkey]
    else:
        return pars


def get_genotype_pars(default=False, genotype=None):
    '''
    Define the default parameters for the different genotypes
    '''
    pars = dict(

        hpv16 = dict(
            rel_beta        = 1.0, # Default values
            rel_cin1_prob    = 1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob = 1.0,
            rel_death_prob  = 1.0
        ),

        hpv18 = dict(
            rel_beta        = 0.8, # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob = 0.8,
            rel_death_prob  = 0.8
        ),

        hpv31=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv33=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv45=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv52=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv6=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=0,
            rel_cin2_prob=0,
            rel_cin3_prob=0,
            rel_cancer_prob=0,
            rel_death_prob=0
        ),

        hpv11=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=0,
            rel_cin2_prob=0,
            rel_cin3_prob=0,
            rel_cancer_prob=0,
            rel_death_prob=0
        ),

        hpvlo=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=0,
            rel_cin2_prob=0,
            rel_cin3_prob=0,
            rel_cancer_prob=0,
            rel_death_prob=0
        ),

        hpvhi=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpvhi5=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

    )

    return _get_from_pars(pars, default, key=genotype, defaultkey='hpv16')


def get_cross_immunity(default=False, genotype=None):
    '''
    Get the cross immunity between each genotype in a sim
    '''
    pars = dict(

        hpv16 = dict(
            hpv16  = 1.0, # Default for own-immunity
            hpv18 = 0, # Assumption
            hpv31  = 0, # Assumption
            hpv33 = 0, # Assumption
            hpv45 = 0, # Assumption
            hpv52 = 0, # Assumption
            hpv58 = 0, # Assumption
            hpv6 = 0, # Assumption
            hpv11 = 0, # Assumption
            hpvlo = 0, # Assumption
            hpvhi = 0, # Assumption
            hpvhi5 = 0, # Assumption
        ),

        hpv18 = dict(
            hpv16=0,  # Default for own-immunity
            hpv18=1.0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpv31=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=1.0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpv33=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=1.0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpv45=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=1.0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpv52=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=1.0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpv58=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=1.0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpv6=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=1.0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpv11=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=1.0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpvlo=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=1.0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpvhi=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=1.0,  # Assumption
            hpvhi5=0,  # Assumption
        ),

        hpvhi5=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
            hpvlo=0,  # Assumption
            hpvhi=0,  # Assumption
            hpvhi5=1.0,  # Assumption

        ),
    )

    return _get_from_pars(pars, default, key=genotype, defaultkey='hpv16')


