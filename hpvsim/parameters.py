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


def make_pars(version=None, nonactive_by_age=False, **kwargs):
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
    pars['pop_size']        = 20e3     # Number of agents
    pars['pop_infected']    = 20       # Number of initial infections; TODO reconsider this
    pars['network']         = 'random' # What type of sexual network to use -- 'random', 'basic', other options TBC
    pars['location']        = None     # What location to load data from -- default Seattle
    pars['death_rates']     = None     # Deaths from all other causes, loaded below 
    pars['birth_rates']     = None     # Birth rates, loaded below 

    # Simulation parameters
    pars['start']           = 2015.         # Start of the simulation
    pars['end']             = None          # End of the simulation
    pars['n_years']         = 10.           # Number of years to run, if end isn't specified
    pars['dt']              = 0.2           # Timestep (in years)
    pars['rand_seed']       = 1             # Random seed, if None, don't reset
    pars['verbose']         = hpo.verbose   # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)

    # Network parameters, generally initialized after the population has been constructed
    pars['debut']           = dict(dist='normal', par1=15.5, par2=1.5) # Age of sexual debut; TODO separate for M/F?
    pars['partners']        = None  # The number of concurrent sexual partners per layer
    pars['acts']            = None  # The number of sexual acts per layer per year
    pars['condoms']         = None  # The proportion of acts in which condoms are used
    pars['layer_probs']     = None  # Proportion of the population in each layer
    pars['dur_pship']       = None  # Duration of partnerships in each layer
    pars['mixing']          = None  # Mixing matrices for storing age differences in partnerships
    # pars['nonactive_by_age']= nonactive_by_age
    # pars['nonactive']       = None 

    # Basic disease transmission parameters
    pars['beta_dist']       = dict(dist='neg_binomial', par1=1.0, par2=1.0, step=0.01) # Distribution to draw individual level transmissibility
    pars['beta']            = 0.05  # Per-act transmission probability; absolute value, calibrated
    pars['n_genotypes'] = 1  # The number of genotypes circulating in the population. By default only HPV

    # Parameters used to calculate immunity
    pars['imm_init'] = dict(dist='normal', par1=0, par2=2)  # beta distribution for initial level of immunity following infection clearance
    pars['imm_decay'] = dict(infection=dict(form='exp_decay', init_val=1, half_life=10),
                             vaccine=dict(form='exp_decay', init_val=1, half_life=10))
    pars['imm_kin'] = None  # Constructed during sim initialization using the nab_decay parameters
    pars['imm_boost'] = 1.5  # Multiplicative factor applied to a person's immunity levels if they get reinfected. No data on this, assumption.
    pars['immunity'] = None  # Matrix of immunity and cross-immunity factors, set by init_immunity() in immunity.py
    pars['immunity_map'] = None  # dictionary mapping the index of immune source to the type of immunity (vaccine vs natural)

    pars['genotypes'] = []  # Genotypes of the virus; populated by the user below
    pars['genotype_map'] = {0: 'HPV16'}  # Reverse mapping from number to variant key
    pars['genotype_pars'] = dict(HPV16={})  # Populated just below
    for sp in hpd.genotype_pars:
        if sp in pars.keys():
            pars['genotype_pars']['HPV16'][sp] = pars[sp]

    # Genotype parameters
    pars['n_genotypes'] = 1 # The number of genotypes circulating in the population
    pars['rel_beta']    = 1.0 # Relative transmissibility varies by genotype (??)

    # Duration parameters
    pars['dur'] = {}

    pars['dur']['inf2rec']  = dict(dist='lognormal', par1=1.0, par2=1.0)  # Duration from infectious to recovered in YEARS

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
        acts        = dict(a=104),  # Default number of sexual acts per year; TODO make this a distribution
        layer_probs = dict(a=1.0),  # Default proportion of the population in each layer
        dur_pship   = dict(a=dict(dist='normal_pos', par1=5,par2=3)),    # Default duration of partnerships; TODO make this a distribution
        condoms     = dict(a=0.25),  # Default proportion of acts in which condoms are used
    )

    # Specify defaults for basic sexual network with regular and casual partners
    layer_defaults['basic'] = dict(
        partners    = dict(r=1, c=2),       # Default number of concurrent sexual partners; TODO make this a distribution and incorporate zero inflation
        acts        = dict(r=100, c=50),    # Default number of sexual acts per year; TODO make this a distribution
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

    return



def get_births_deaths(location=None, verbose=1, by_sex=True, overall=False):
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
            hpm.warn(warnmsg)
    
    return birth_rates, death_rates

#%% Genotype/immunity parameters and functions

def get_genotype_choices():
    '''
    Define valid genotype names
    '''
    # List of choices currently available
    choices = {
        'HPV16':  ['HPV16', '16'],
        'HPV18': ['HPV18', '18'],
        'HPV6':  ['HPV6', '6'],
        'HPV11': ['HPV11', '11'],
        'HPV31': ['HPV31', '31'],
        'HPV33': ['HPV33', '33'],
        'HPV45': ['HPV45', '45'],
        'HPV52': ['HPV52', '52'],
        'HPV58': ['HPV58', '58'],
        'HPVlo': ['HPVlo', 'low', 'low-risk'],
        'HPVhi': ['HPVhi', 'high', 'high-risk'],
        'HPVhi5': ['HPVhi', 'high', 'high-risk'],
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

        HPV16 = dict(
            rel_beta        = 1.0, # Default values
        ),

        HPV18 = dict(
            rel_beta        = 1.0, # Default values
        ),

        HPV31=dict(
            rel_beta=1.0,  # Default values
        ),

        HPV33=dict(
            rel_beta=1.0,  # Default values
        ),

        HPV45=dict(
            rel_beta=1.0,  # Default values
        ),

        HPV52=dict(
            rel_beta=1.0,  # Default values
        ),

        HPV6=dict(
            rel_beta=1.0,  # Default values
        ),

        HPV11=dict(
            rel_beta=1.0,  # Default values
        ),

        HPVlo=dict(
            rel_beta=1.0,  # Default values
        ),

        HPVhi=dict(
            rel_beta=1.0,  # Default values
        ),

        HPVhi5=dict(
            rel_beta=1.0,  # Default values
        ),

    )

    return _get_from_pars(pars, default, key=genotype, defaultkey='HPV16')


def get_cross_immunity(default=False, genotype=None):
    '''
    Get the cross immunity between each genotype in a sim
    '''
    pars = dict(

        HPV16 = dict(
            HPV16  = 1.0, # Default for own-immunity
            HPV18 = 1.0, # Assumption
            HPV31  = 1.0, # Assumption
            HPV33 = 1.0, # Assumption
            HPV45 = 1.0, # Assumption
            HPV52 = 1.0, # Assumption
            HPV58 = 1.0, # Assumption
            HPV6 = 1.0, # Assumption
            HPV11 = 1.0, # Assumption
            HPVlo = 1.0, # Assumption
            HPVhi = 1.0, # Assumption
            HPVhi5 = 1.0, # Assumption
        ),

        HPV18 = dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPV31=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPV33=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPV45=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPV52=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPV58=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPV6=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPV11=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPVlo=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPVhi=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),

        HPVhi5=dict(
            HPV16=1.0,  # Default for own-immunity
            HPV18=1.0,  # Assumption
            HPV31=1.0,  # Assumption
            HPV33=1.0,  # Assumption
            HPV45=1.0,  # Assumption
            HPV52=1.0,  # Assumption
            HPV58=1.0,  # Assumption
            HPV6=1.0,  # Assumption
            HPV11=1.0,  # Assumption
            HPVlo=1.0,  # Assumption
            HPVhi=1.0,  # Assumption
            HPVhi5=1.0,  # Assumption
        ),
    )

    return _get_from_pars(pars, default, key=genotype, defaultkey='HPV16')


