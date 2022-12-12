'''
Set the parameters for hpvsim.
'''

import numpy as np
import sciris as sc
import pandas as pd
from .settings import options as hpo # For setting global options
from . import misc as hpm
from . import defaults as hpd
from .data import loaders as hpdata

__all__ = ['make_pars', 'reset_layer_pars']


def make_pars(**kwargs):
    '''
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = hpv.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        version       (str):  if supplied, use parameters from this version
        kwargs        (dict): any additional kwargs are interpreted as parameter names

    Returns:
        pars (dict): the parameters of the simulation
    '''
    pars = {}

    # Population parameters
    pars['n_agents']        = 20e3      # Number of agents
    pars['total_pop']       = None      # If defined, used for calculating the scale factor
    pars['pop_scale']       = None      # How much to scale the population
    pars['use_multiscale']  = False     # Whether to use multiscale modeling
    pars['ms_agent_ratio']  = 1         # Ratio of scale factor of cancer agents to normal agents -- must be an integer
    pars['network']         = 'default' # What type of sexual network to use -- 'random', 'basic', other options TBC
    pars['location']        = None      # What location to load data from -- default Seattle
    pars['lx']              = None      # Proportion of people alive at the beginning of age interval x
    pars['birth_rates']     = None      # Birth rates, loaded below
    pars['death_rates']     = None      # Death rates, loaded below
    pars['rel_birth']       = 1.0       # Birth rate scale factor
    pars['rel_death']       = 1.0       # Death rate scale factor

    # Initialization parameters
    pars['init_hpv_prev'] = sc.dcp(hpd.default_init_prev) # Initial prevalence
    pars['init_hpv_dist'] = None  # Initial type distribution
    pars['rel_init_prev'] = 1.0 # Initial prevalence scale factor

    # Simulation parameters
    pars['start']           = 2015.         # Start of the simulation
    pars['end']             = None          # End of the simulation
    pars['n_years']         = 15            # Number of years to run, if end isn't specified. Note that this includes burn-in
    pars['burnin']          = 5             # Number of years of burnin. NB, this is doesn't affect the start and end dates of the simulation, but it is possible remove these years from plots
    pars['dt']              = 0.2           # Timestep (in years)
    pars['dt_demog']        = 1.0           # Timestep for demographic updates (in years)
    pars['rand_seed']       = 1             # Random seed, if None, don't reset
    pars['verbose']         = hpo.verbose   # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)
    pars['use_waning']      = False         # Whether or not to use waning immunity. If set to False, immunity from infection and vaccination is assumed to stay at the same level permanently
    pars['use_migration']   = True          # Whether to estimate migration rates to correct the total population size
    pars['model_hiv']       = False         # Whether or not to model HIV natural history

    # Network parameters, generally initialized after the population has been constructed
    pars['debut']           = dict(f=dict(dist='normal', par1=18.6, par2=2.1), # Location-specific data should be used here if possible
                                   m=dict(dist='normal', par1=19.6, par2=1.8))
    pars['cross_layer']     = 0.05  # Proportion of females who have crosslayer relationships
    pars['partners']        = None  # The number of concurrent sexual partners for each partnership type
    pars['acts']            = None  # The number of sexual acts for each partnership type per year
    pars['condoms']         = None  # The proportion of acts in which condoms are used for each partnership type
    pars['layer_probs']     = None  # Proportion of the population in each partnership type
    pars['dur_pship']       = None  # Duration of partnerships in each partnership type
    pars['mixing']          = None  # Mixing matrices for storing age differences in partnerships
    pars['n_partner_types'] = 1  # Number of partnership types - reset below

    # Basic disease transmission parameters
    pars['beta']                = 0.25  # Per-act transmission probability; absolute value, calibrated
    pars['transf2m']            = 1.0   # Relative transmissibility of receptive partners in penile-vaginal intercourse; baseline value
    pars['transm2f']            = 3.69  # Relative transmissibility of insertive partners in penile-vaginal intercourse; based on https://doi.org/10.1038/srep10986: "For vaccination types, the risk of male-to-female transmission was higher than that of female-to-male transmission"
    pars['rel_trans_cin1']      = 1     # Transmissibility of people with CIN1 compared to those without dysplasia
    pars['rel_trans_cin2']      = 1     # Transmissibility of people with CIN2 compared to those without dysplasia
    pars['rel_trans_cin3']      = 1     # Transmissibility of people with CIN3 compared to those without dysplasia
    pars['rel_trans_cancerous'] = 0.5   # Transmissibility of people with cancer compared to those without dysplasia
    pars['eff_condoms']         = 0.7   # The efficacy of condoms; https://www.nejm.org/doi/10.1056/NEJMoa053284?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200www.ncbi.nlm.nih.gov

    # Parameters for disease progression
    pars['clinical_cutoffs']    = {'cin1': 0.33, 'cin2':0.67} # Parameters the control the clinical cliassification of dysplasia
    pars['hpv_control_prob']    = 0.0 # Probability that HPV is controlled latently vs. cleared
    pars['hpv_reactivation']    = 0.025 # Placeholder
    pars['dur_cancer']          = dict(dist='lognormal', par1=12.0, par2=3.0)  # Duration of untreated invasive cerival cancer before death (years)

    # Parameters used to calculate immunity
    pars['imm_init']        = dict(dist='beta_mean', par1=0.625, par2=0.025)  # beta distribution for initial level of immunity following infection clearance. Parameters are mean and variance
    pars['imm_decay']       = dict(form=None)  # decay rate, with half life in years
    pars['imm_boost']       = []  # Multiplicative factor applied to a person's immunity levels if they get reinfected. No data on this, assumption.
    pars['immunity']        = None  # Matrix of immunity and cross-immunity factors, set by init_immunity() in immunity.py
    pars['cross_imm_med']   = 0.3
    pars['cross_imm_high']  = 0.5

    # Genotype parametres
    pars['genotypes']       = [16, 18, 'hrhpv']  # Genotypes to model
    pars['genotype_pars']   = sc.objdict()  # Can be directly modified by passing in arguments listed in get_genotype_pars

    # HIV parameters
    pars['hiv_pars'] = {
        'rel_sus': 2.2,
        'dysp_rate': 2,
        'prog_rate': 2,
        'reactivation_prob': 3,
    }

    # Events and interventions
    pars['interventions']   = sc.autolist() # The interventions present in this simulation; populated by the user
    pars['analyzers']       = sc.autolist() # The functions present in this simulation; populated by the user
    pars['timelimit']       = None # Time limit for the simulation (seconds)
    pars['stopping_func']   = None # A function to call to stop the sim partway through

    # Population distribution of the World Standard Population, used to calculate age-standardised rates (ASR) of incidence
    pars['age_bins']        = np.array( [  0,   5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75,    80,    85, 100])
    pars['standard_pop']    = np.array([pars['age_bins'],
                                        [.12, .10, .09, .09, .08, .08, .06, .06, .06, .06, .05, .04, .04, .03, .02, .01, 0.005, 0.005,   0]])

    # The following variables are stored within the pars dict for ease of access, but should not be directly specified.
    # Rather, they are automaticall constructed during sim initialization.
    pars['immunity_map']    = None  # dictionary mapping the index of immune source to the type of immunity (vaccine vs natural)
    pars['imm_kin']         = None  # Constructed during sim initialization using the nab_decay parameters
    pars['genotype_map']    = dict()  # Reverse mapping from number to genotype key
    pars['n_genotypes']     = 1 # The number of genotypes circulating in the population
    pars['n_imm_sources']   = 1 # The number of immunity sources circulating in the population
    pars['vaccine_pars']    = dict()  # Vaccines that are being used; populated during initialization
    pars['vaccine_map']     = dict()  # Reverse mapping from number to vaccine key

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    reset_layer_pars(pars)

    return pars


# Define which parameters need to be specified as a dictionary by layer -- define here so it's available at the module level for sim.py
layer_pars = ['partners', 'mixing', 'acts', 'age_act_pars', 'layer_probs', 'dur_pship', 'condoms']


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
        partners    = dict(a=dict(dist='poisson', par1=0.01)), # Everyone in this layer has one partner; this captures *additional* partners. If using a poisson distribution, par1 is roughly equal to the proportion of people with >1 partner
        acts        = dict(a=dict(dist='neg_binomial', par1=100,par2=50)),  # Default number of sexual acts per year for people at sexual peak
        age_act_pars = dict(a=dict(peak=35, retirement=75, debut_ratio=0.5, retirement_ratio=0.1)), # Parameters describing changes in coital frequency over agent lifespans
        layer_probs = dict(a=1.0),  # Default proportion of the population in each layer
        dur_pship   = dict(a=dict(dist='normal_pos', par1=5,par2=3)),    # Default duration of partnerships
        condoms     = dict(a=0.25),  # Default proportion of acts in which condoms are used
    )
    layer_defaults['random']['mixing'], layer_defaults['random']['layer_probs'] = get_mixing('random')

    # Specify defaults for basic sexual network with marital, casual, and one-off partners
    layer_defaults['default'] = dict(
        partners    = dict(m=dict(dist='poisson', par1=0.1), # Everyone in this layer has one marital partner; this captures *additional* marital partners. If using a poisson distribution, par1 is roughly equal to the proportion of people with >1 spouse
                           c=dict(dist='poisson', par1=0.5), # If using a poisson distribution, par1 is roughly equal to the proportion of people with >1 casual partner at a time
                           o=dict(dist='poisson', par1=0.0),), # If using a poisson distribution, par1 is roughly equal to the proportion of people with >1 one-off partner at a time. Can be set to zero since these relationships only last a single timestep
        acts         = dict(m=dict(dist='neg_binomial', par1=80, par2=40), # Default number of acts per year for people at sexual peak
                            c=dict(dist='neg_binomial', par1=10, par2=5), # Default number of acts per year for people at sexual peak
                            o=dict(dist='neg_binomial', par1=1,  par2=.01)),  # Default number of acts per year for people at sexual peak
        age_act_pars = dict(m=dict(peak=35, retirement=75, debut_ratio=0.5, retirement_ratio=0.1), # Parameters describing changes in coital frequency over agent lifespans
                            c=dict(peak=25, retirement=75, debut_ratio=0.5, retirement_ratio=0.1),
                            o=dict(peak=25, retirement=50, debut_ratio=0.5, retirement_ratio=0.1)),
        dur_pship   = dict(m=dict(dist='normal_pos', par1=8, par2=3),
                           c=dict(dist='normal_pos', par1=1, par2=1),
                           o=dict(dist='normal_pos', par1=0.1, par2=0.05)),
        condoms     = dict(m=0.01, c=0.5, o=0.6),  # Default proportion of acts in which condoms are used
    )
    layer_defaults['default']['mixing'], layer_defaults['default']['layer_probs'] = get_mixing('default')

    # Choose the parameter defaults based on the population type, and get the layer keys
    try:
        defaults = layer_defaults[pars['network']]
    except Exception as E:
        errormsg = f'Cannot load defaults for population type "{pars["network"]}"'
        raise ValueError(errormsg) from E
    default_layer_keys = list(defaults['acts'].keys()) # All layers should be the same, but use acts for convenience

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


def get_births_deaths(location=None, verbose=1, by_sex=True, overall=False, die=None):
    '''
    Get mortality and fertility data by location if provided, or use default

    Args:
        location (str):  location; if none specified, use default value for XXX
        verbose (bool):  whether to print progress
        by_sex   (bool): whether to get sex-specific death rates (default true)
        overall  (bool): whether to get overall values ie not disaggregated by sex (default false)

    Returns:
        lx (dict): dictionary keyed by sex, storing arrays of lx - the number of people who survive to age x
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
        'hpv35': ['hpv35', '35'],
        'hpv45': ['hpv45', '45'],
        'hpv51': ['hpv51', '51'],
        'hpv52': ['hpv52', '52'],
        'hpv56': ['hpv56', '56'],
        'hpv58': ['hpv58', '58'],
        'hrhpv': ['hrhpv', 'ohrhpv', 'hr', 'ohr'],
        'lrhpv': ['lrhpv', 'lr'],
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping

def get_vaccine_choices():
    '''
    Define valid pre-defined vaccine names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'default': ['default', None],
        'bivalent':  ['bivalent', 'hpv2', 'cervarix'],
        'quadrivalent': ['quadrivalent', 'hpv4', 'gardasil'],
        'nonavalent': ['nonavalent', 'hpv9', 'cervarix9'],
    }
    dose_1_options = ['1dose', '1doses', '1_dose', '1_doses', 'single_dose']
    dose_2_options = ['2dose', '2doses', '2_dose', '2_doses', 'double_dose']
    dose_3_options = ['3dose', '3doses', '3_dose', '3_doses', 'triple_dose']

    choices['bivalent_2dose'] = [f'{x}_{dose}' for x in choices['bivalent'] for dose in dose_2_options]
    choices['bivalent_3dose'] = [f'{x}_{dose}' for x in choices['bivalent'] for dose in dose_3_options]
    choices['bivalent'] = ['bivalent']+[f'{x}_{dose}' for x in choices['bivalent'] for dose in dose_1_options]
    choices['quadrivalent_2dose'] = [f'{x}_{dose}' for x in choices['quadrivalent'] for dose in dose_2_options]
    choices['quadrivalent_3dose'] = [f'{x}_{dose}' for x in choices['quadrivalent'] for dose in dose_3_options]
    choices['quadrivalent'] = ['quadrivalent']+[f'{x}_{dose}' for x in choices['quadrivalent'] for dose in dose_1_options]
    choices['nonavalent_2dose'] = [f'{x}_{dose}' for x in choices['nonavalent'] for dose in dose_2_options]
    choices['nonavalent_3dose'] = [f'{x}_{dose}' for x in choices['nonavalent'] for dose in dose_3_options]
    choices['nonavalent'] = ['nonavalent']+[f'{x}_{dose}' for x in choices['nonavalent'] for dose in dose_1_options]

    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping



def get_treatment_choices():
    '''
    Define valid pre-defined treatment names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'default': ['default', None],
        'ablative':  ['ablative', 'thermal_ablation', 'TA'],
        'excisional': ['excisional', 'leep'],
        'radiation': ['radiation']
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

    pars = sc.objdict()
    mean16 = 13.9/12 # Defined here since used repeatedly below. This is the duration of HPV16 infections truncated at the time of CIN detection: https://pubmed.ncbi.nlm.nih.gov/17416761/

    pars.hpv16 = sc.objdict()
    pars.hpv16.dur_precin   = dict(dist='lognormal', par1=mean16, par2=0.4) # Duration of HPV infections truncated at the time of CIN detection: https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv16.dur_dysp     = dict(dist='lognormal', par1=4.5, par2=4.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv16.dysp_rate    = 1.5 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv16.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv16.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv16.rel_beta     = 1  # Baseline relative transmissibility, other genotypes are relative to this
    pars.hpv16.cancer_prob  = 0.25 # Share of CIN3s that will go on to cancer
    pars.hpv16.imm_boost    = 1.0 # TODO: look for data
    pars.hpv16.sero_prob    = 0.65 # https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv18 = sc.objdict()
    pars.hpv18.dur_precin   = dict(dist='lognormal', par1=14.9/12, par2=0.4) # Duration of HPV infections truncated at the time of CIN detection: https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv18.dur_dysp     = dict(dist='lognormal', par1=3.16, par2=4.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv18.dysp_rate    = 1.2 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv18.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv18.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv18.rel_beta     = 0.72  # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv18.cancer_prob  = 0.15  # Share of CIN3s that will go on to cancer
    pars.hpv18.imm_boost    = 1.0 # TODO: look for data
    pars.hpv18.sero_prob    = 0.6 # https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv31 = sc.objdict()
    pars.hpv31.dur_precin   = dict(dist='lognormal', par1=14.4/12.4*mean16, par2=0.4) # Multiply the mean duration of HPV16 infection truncated at the time of CIN detection (https://pubmed.ncbi.nlm.nih.gov/17416761/) by a scale factor derived from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/figure/F1/
    pars.hpv31.dur_dysp     = dict(dist='lognormal', par1=1.35, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv31.dysp_rate    = 0.1 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv31.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv31.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv31.rel_beta     = 0.94 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv31.cancer_prob  = 0.05  # Share of CIN3s that will go on to cancer
    pars.hpv31.imm_boost    = 1.0 # TODO: look for data
    pars.hpv31.sero_prob    = 0.53 # https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv33 = sc.objdict()
    pars.hpv33.dur_precin   = dict(dist='lognormal', par1=12.5/12.4*mean16, par2=0.4) # Multiply the mean duration of HPV16 infection truncated at the time of CIN detection (https://pubmed.ncbi.nlm.nih.gov/17416761/) by a scale factor derived from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/figure/F1/
    pars.hpv33.dur_dysp     = dict(dist='lognormal', par1=14.12, par2=3.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv33.dysp_rate    = 0.2 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv33.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv33.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv33.rel_beta     = 0.26 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv33.cancer_prob  = 0.05  # Share of CIN3s that will go on to cancer
    pars.hpv33.imm_boost    = 1.0 # TODO: look for data
    pars.hpv33.sero_prob    = 0.6  # https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv35 = sc.objdict()
    pars.hpv35.dur_precin   = dict(dist='lognormal', par1=6.0/12.4*mean16, par2=0.4) # Multiply the mean duration of HPV16 infection truncated at the time of CIN detection (https://pubmed.ncbi.nlm.nih.gov/17416761/) by a scale factor derived from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/figure/F1/
    pars.hpv35.dur_dysp     = dict(dist='lognormal', par1=4.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv35.dysp_rate    = 0.3 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv35.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv35.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv35.rel_beta     = 0.5 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv35.cancer_prob  = 0.05  # Share of CIN3s that will go on to cancer
    pars.hpv35.imm_boost    = 1.0 # TODO: look for data
    pars.hpv35.sero_prob    = 0.6  # Assumption, not provided in https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv45 = sc.objdict()
    pars.hpv45.dur_precin   = dict(dist='lognormal', par1=8.0/12.4*mean16, par2=0.4) # Multiply the mean duration of HPV16 infection truncated at the time of CIN detection (https://pubmed.ncbi.nlm.nih.gov/17416761/) by a scale factor derived from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/figure/F1/
    pars.hpv45.dur_dysp     = dict(dist='lognormal', par1=3.776, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv45.dysp_rate    = 1.2 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv45.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv45.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv45.rel_beta     = 0.77 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv45.cancer_prob  = 0.15  # Share of CIN3s that will go on to cancer
    pars.hpv45.imm_boost    = 1.0 # TODO: look for data
    pars.hpv45.sero_prob    = 0.25  # https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv51 = sc.objdict()
    pars.hpv51.dur_precin   = dict(dist='lognormal', par1=7.4/12.4*mean16, par2=0.4) # Multiply the mean duration of HPV16 infection truncated at the time of CIN detection (https://pubmed.ncbi.nlm.nih.gov/17416761/) by a scale factor derived from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/figure/F1/
    pars.hpv51.dur_dysp     = dict(dist='lognormal', par1=1.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv51.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv51.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv51.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv51.rel_beta     = 0.5 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv51.cancer_prob  = 0.05  # Share of CIN3s that will go on to cancer
    pars.hpv51.imm_boost    = 1.0 # TODO: look for data
    pars.hpv51.sero_prob    = 0.6  # Assumption, not provided in https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv52 = sc.objdict()
    pars.hpv52.dur_precin   = dict(dist='lognormal', par1=11.7/12.4*mean16, par2=0.4) # Multiply the mean duration of HPV16 infection truncated at the time of CIN detection (https://pubmed.ncbi.nlm.nih.gov/17416761/) by a scale factor derived from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/figure/F1/
    pars.hpv52.dur_dysp     = dict(dist='lognormal', par1=1.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv52.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv52.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv52.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv52.rel_beta     = 0.623 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv52.cancer_prob  = 0.05  # Share of CIN3s that will go on to cancer
    pars.hpv52.imm_boost    = 1.0 # TODO: look for data
    pars.hpv52.sero_prob    = 0.5  # https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv56 = sc.objdict()
    pars.hpv56.dur_precin   = dict(dist='lognormal', par1=11.2/12.4*mean16, par2=0.4) # Multiply the mean duration of HPV16 infection truncated at the time of CIN detection (https://pubmed.ncbi.nlm.nih.gov/17416761/) by a scale factor derived from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/figure/F1/
    pars.hpv56.dur_dysp     = dict(dist='lognormal', par1=3.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv56.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv56.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv56.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv56.rel_beta     = 0.6 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv56.cancer_prob  = 0.15  # Share of CIN3s that will go on to cancer
    pars.hpv56.imm_boost    = 1.0 # TODO: look for data
    pars.hpv56.sero_prob    = 0.6  # Assumption, not provided in https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv58 = sc.objdict()
    pars.hpv58.dur_precin   = dict(dist='lognormal', par1=9.7/12.4*mean16, par2=0.4) # Multiply the mean duration of HPV16 infection truncated at the time of CIN detection (https://pubmed.ncbi.nlm.nih.gov/17416761/) by a scale factor derived from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/figure/F1/
    pars.hpv58.dur_dysp     = dict(dist='lognormal', par1=3.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv58.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv58.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv58.prog_rate_sd = 0.05 # Standard deviation of the progression rate
    pars.hpv58.rel_beta     = 0.6 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv58.cancer_prob  = 0.15  # Share of CIN3s that will go on to cancer
    pars.hpv58.imm_boost    = 1.0 # TODO: look for data
    pars.hpv58.sero_prob    = 0.65  # Assumption, not provided in https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv6 = sc.objdict()
    pars.hpv6.dur_precin    = dict(dist='lognormal', par1=8.4/12, par2=0.4) # Duration of HPV infections truncated at the time of CIN detection: https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv6.dur_dysp      = dict(dist='lognormal', par1=0.5, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv6.dysp_rate     = 0.01 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv6.prog_rate     = 0.05 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv6.prog_rate_sd  = 0.005 # Standard deviation of the progression rate
    pars.hpv6.rel_beta      = 0.8 # Relative transmissibility, generally calibrated to match available type distributions
    pars.hpv6.cancer_prob   = 0.0 # Share of CIN3s that will go on to cancer
    pars.hpv6.imm_boost     = 1.0 # TODO: look for data
    pars.hpv6.sero_prob     = 0.6  # Assumption, not provided in https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hpv11 = sc.objdict()
    pars.hpv11.dur_precin   = dict(dist='lognormal', par1=8.1/12, par2=0.4)  # Duration of HPV infections truncated at the time of CIN detection: https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv11.dur_dysp     = dict(dist='lognormal', par1=4.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv11.dysp_rate    = 0.01 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv11.prog_rate    = 0.05 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv11.prog_rate_sd = 0.005 # Standard deviation of the progression rate
    pars.hpv11.rel_beta     = 0.5 # Relative transmissibility, generally calibrated to match available type distributions
    pars.hpv11.cancer_prob  = 0.0  # Share of CIN3s that will go on to cancer
    pars.hpv11.imm_boost    = 1.0 # TODO: look for data
    pars.hpv11.sero_prob    = 0.6  # Assumption, not provided in https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

    pars.hrhpv = sc.objdict()
    pars.hrhpv.dur_precin   = dict(dist='lognormal', par1=14.4/12.4*mean16, par2=0.4) # placeholder, currently assumed to be the same as for 31
    pars.hrhpv.dur_dysp     = dict(dist='lognormal', par1=1.35, par2=4.0) # placeholder, currently assumed to be the same as for 31
    pars.hrhpv.dysp_rate    = 0.1 # placeholder, currently assumed to be the same as for 31
    pars.hrhpv.prog_rate    = 0.5 # same value as for all oncogenic types
    pars.hrhpv.prog_rate_sd = 0.05 # same value as for all oncogenic types
    pars.hrhpv.rel_beta     = 0.94 # placeholder, currently assumed to be the same as for 31
    pars.hrhpv.cancer_prob  = 0.25  # Share of CIN3s that will go on to cancer
    pars.hrhpv.imm_boost    = 1.0 # placeholder, currently assumed to be the same as for 31
    pars.hrhpv.sero_prob    = 0.5 # placeholder, currently assumed to be the same as for 31

    pars.lrhpv = sc.objdict()
    pars.lrhpv.dur_precin   = dict(dist='lognormal', par1=8.1/12, par2=0.4)  # placeholder, currently assumed to be the same as for 6
    pars.lrhpv.dur_dysp     = dict(dist='lognormal', par1=4.0, par2=1.0) # placeholder, currently assumed to be the same as for 6
    pars.lrhpv.dysp_rate    = 0.01 # placeholder, currently assumed to be the same as for 6
    pars.lrhpv.prog_rate    = 0.05 # placeholder, currently assumed to be the same as for 6
    pars.lrhpv.prog_rate_sd = 0.005 # placeholder, currently assumed to be the same as for 6
    pars.lrhpv.rel_beta     = 0.5 # placeholder, currently assumed to be the same as for 6
    pars.lrhpv.cancer_prob  = 0.0  # Share of CIN3s that will go on to cancer
    pars.lrhpv.imm_boost    = 1.0 # placeholder, currently assumed to be the same as for 6
    pars.lrhpv.sero_prob    = 0.6  # placeholder, currently assumed to be the same as for 6

    return _get_from_pars(pars, default, key=genotype, defaultkey='hpv16')


def get_cross_immunity(cross_imm_med=None, cross_imm_high=None, default=False, genotype=None):
    '''
    Get the cross immunity between each genotype in a sim
    '''
    pars = dict(
        # All values based roughly on https://academic.oup.com/jnci/article/112/10/1030/5753954 or assumptions
        hpv16 = dict(
            hpv16  = 1.0, # Default for own-immunity
            hpv18 = cross_imm_high,
            hpv31  = cross_imm_high,
            hpv33 = cross_imm_high,
            hpv35 = cross_imm_high,
            hpv45 = cross_imm_high,
            hpv51 = cross_imm_med,
            hpv52 = cross_imm_med,
            hpv56 = cross_imm_med,
            hpv58 = cross_imm_med,
            hpv6 = cross_imm_med,
            hpv11 = cross_imm_med,
            hrhpv = 0.1,
            lrhpv = cross_imm_med
        ),

        hpv18 = dict(
            hpv16=cross_imm_high,
            hpv18=1.0,  # Default for own-immunity
            hpv31=cross_imm_high,
            hpv33=cross_imm_high,
            hpv35=cross_imm_high,
            hpv45=cross_imm_high,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv = 0.1,
            lrhpv = cross_imm_med
        ),

        hpv31=dict(
            hpv16=cross_imm_high,
            hpv18=cross_imm_high,
            hpv31=1.0,  # Default for own-immunity
            hpv33=cross_imm_high,
            hpv35=cross_imm_high,
            hpv45=cross_imm_high,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_high,
            lrhpv=cross_imm_med
        ),

        hpv33=dict(
            hpv16=cross_imm_high,
            hpv18=cross_imm_high,
            hpv31=cross_imm_high,
            hpv33=1.0,  # Default for own-immunity
            hpv35=cross_imm_high,
            hpv45=cross_imm_high,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_high,
            lrhpv=cross_imm_med
        ),

        hpv35=dict(
            hpv16=cross_imm_high,
            hpv18=cross_imm_high,
            hpv31=cross_imm_high,
            hpv33=cross_imm_high,
            hpv35=1.0,  # Default for own-immunity
            hpv45=cross_imm_high,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_high,
            lrhpv=cross_imm_med
        ),

        hpv45=dict(
            hpv16=cross_imm_high,
            hpv18=cross_imm_high,
            hpv31=cross_imm_high,
            hpv33=cross_imm_high,
            hpv35=cross_imm_high,
            hpv45=1.0,  # Default for own-immunity
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_high,
            lrhpv=cross_imm_med
        ),

        hpv51=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hpv31=cross_imm_med,
            hpv33=cross_imm_med,
            hpv35=cross_imm_med,
            hpv45=cross_imm_med,
            hpv51=1.0,  # Default for own-immunity
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_high,
            lrhpv=cross_imm_med
        ),

        hpv52=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hpv31=cross_imm_med,
            hpv33=cross_imm_med,
            hpv35=cross_imm_med,
            hpv45=cross_imm_med,
            hpv51=cross_imm_med,
            hpv52=1.0,  # Default for own-immunity
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_high,
            lrhpv=cross_imm_med
        ),

        hpv56=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hpv31=cross_imm_med,
            hpv33=cross_imm_med,
            hpv35=cross_imm_med,
            hpv45=cross_imm_med,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=1.0,  # Default for own-immunity
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_high,
            lrhpv=cross_imm_med
        ),

        hpv58=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hpv31=cross_imm_med,
            hpv33=cross_imm_med,
            hpv35=cross_imm_med,
            hpv45=cross_imm_med,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=1.0,  # Default for own-immunity
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_high,
            lrhpv=cross_imm_med
        ),

        hpv6=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hpv31=cross_imm_med,
            hpv33=cross_imm_med,
            hpv35=cross_imm_med,
            hpv45=cross_imm_med,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=1.0,  # Default for own-immunity
            hpv11=cross_imm_med,
            hrhpv=cross_imm_med,
            lrhpv=cross_imm_high
        ),

        hpv11=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hpv31=cross_imm_med,
            hpv33=cross_imm_med,
            hpv35=cross_imm_med,
            hpv45=cross_imm_med,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=1.0,  # Default for own-immunity
            hrhpv=cross_imm_med,
            lrhpv=cross_imm_high
        ),
        hrhpv=dict(
            hpv16=0.1,
            hpv18=0.1,
            hpv31=cross_imm_med,
            hpv33=cross_imm_med,
            hpv35=cross_imm_med,
            hpv45=cross_imm_med,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_med,
            lrhpv=cross_imm_med
        ),

        lrhpv=dict(
            hpv16=cross_imm_med,
            hpv18=cross_imm_med,
            hpv31=cross_imm_med,
            hpv33=cross_imm_med,
            hpv35=cross_imm_med,
            hpv45=cross_imm_med,
            hpv51=cross_imm_med,
            hpv52=cross_imm_med,
            hpv56=cross_imm_med,
            hpv58=cross_imm_med,
            hpv6=cross_imm_med,
            hpv11=cross_imm_med,
            hrhpv=cross_imm_med,
            lrhpv=1.0# Default for own-immunity
        ),
    )

    return _get_from_pars(pars, default, key=genotype, defaultkey='hpv16')


def get_mixing(network=None):
    '''
    Define defaults for sexual mixing matrices and the proportion of people of each age group
    who have relationships of each type.

    The mixing matrices represent males in the rows and females in the columns.
    Non-zero entires mean that there are some relationships between males/females of the age
    bands in the row/column combination. Entries >1 or <1 can be used to represent relative
    likelihoods of males of a given age cohort partnering with females of that cohort.
    For example, a mixing matrix like the following would mean that males aged 15-30 were twice
    likely to partner with females of age 15-30 compared to females aged 30-50.
        mixing = np.array([
                                #15, 30,
                            [15,  2, 1],
                            [30,  1, 1]])
    Note that the first column of the mixing matrix represents the age bins. The same age bins
    must be used for males and females, i.e. the matrix must be square.

    The proportion of people of each age group who have relationships of each type is
    given by the layer_probs array. The first row represents the age bins, the second row
    represents the proportion of females of each age who have relationships of each type, and
    the third row represents the proportion of males of each age who have relationships of
    each type.
    '''

    if network == 'default':

        mixing = dict(
            m=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .1, .1, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0, .5, .1, .5 ,.1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  1, .5, .5, .5, .5, .1,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0, .5,  1,  1, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0, .1,  1,  1,  2,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0, .1, .5,  1,  1,  1,  2, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1],
        ]),
            c=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0,  1,  1, 1,  1.0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .5,  1,  1, 1,  1.0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0,  0,  1,  1,  1, 1.0,  1.0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  0,  .5,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0,  0,  .5,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0,  0,  .5,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0,  0.5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
        ]),
            o=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
        ]),
        )

        layer_probs = dict(
            m=np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,   55,   60,   65,   70,   75],
                [ 0,  0,  0.04,   0.1,  0.1,  0.5,  0.6,  0.7,  0.75,  0.65,  0.55,  0.4,  0.4,  0.4,  0.4,  0.4], # Share of females of each age who are married
                [ 0,  0,  0.01,  0.01,  0.1,  0.5,  0.6,  0.7,  0.70,  0.70,  0.70,  0.8,  0.7,  0.6,  0.5,  0.6]] # Share of males of each age who are married
            ),
            c=np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,   55,   60,   65,   70,   75],
                [ 0,  0,  0.10,   0.7,  0.8,  0.6,  0.6,  0.5,   0.2,  0.05,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # Share of females of each age having casual relationships
                [ 0,  0,  0.05,   0.7,  0.8,  0.6,  0.6,  0.5,   0.5,   0.4,   0.3,  0.1, 0.05, 0.01, 0.01, 0.01]], # Share of males of each age having casual relationships
            ),
            o=np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,   55,   60,   65,   70,   75],
                [ 0,  0,  0.01,  0.05, 0.05, 0.04, 0.03, 0.02,  0.01,  0.01,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # Share of females of each age having one-off relationships
                [ 0,  0,  0.01,  0.01, 0.01, 0.02, 0.03, 0.04,  0.05,  0.05,  0.03, 0.02, 0.01, 0.01, 0.01, 0.01]], # Share of males of each age having one-off relationships
            ),
        )

    elif network == 'random':
        mixing = dict(
            a=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
        ])
        )
        layer_probs = dict(
            a=np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,   55,   60,   65,   70,   75],
                [ 0,  0,  0.04,   0.2,  0.6,  0.8,  0.8,  0.8,  0.75,  0.65,  0.55,  0.4,  0.4,  0.4,  0.4,  0.4], # Share of females of each age who are married
                [ 0,  0,  0.01,  0.01,  0.2,  0.6,  0.8,  0.9,  0.90,  0.90,  0.90,  0.8,  0.7,  0.6,  0.5,  0.6]] # Share of males of each age who are married
            ))

    else:
        errormsg = f'Network "{network}" not found; the choices at this stage are random and default.'
        raise ValueError(errormsg)

    return mixing, layer_probs



def get_vaccine_dose_pars(default=False, vaccine=None):
    '''
    Define the parameters for each vaccine
    '''

    pars = dict(

        default = dict(
            imm_init  = dict(dist='beta', par1=30, par2=2), # Initial distribution of immunity
            doses     = 1, # Number of doses for this vaccine
            interval  = None, # Interval between doses
            imm_boost=None,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),

        bivalent = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=1,  # Number of doses for this vaccine
            interval=None,  # Interval between doses
            imm_boost=None,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),

        bivalent_2dose = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=2,  # Number of doses for this vaccine
            interval=0.5,  # Interval between doses in years
            imm_boost=1.2,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),

        bivalent_3dose = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=3,  # Number of doses for this vaccine
            interval=[0.2, 0.5],  # Interval between doses in years
            imm_boost=[1.2, 1.1],  # Factor by which each dose increases immunity
        ),

        quadrivalent = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=1,  # Number of doses for this vaccine
            interval=None,  # Interval between doses
            imm_boost=None,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),

        nonavalent = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=1,  # Number of doses for this vaccine
            interval=None,  # Interval between doses
            imm_boost=None,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),
    )

    return _get_from_pars(pars, default, key=vaccine)


def get_life_expectancy(location, verbose=False):
    '''
    Get life expectancy data by location
    life_expectancy (dict): dictionary storing life expectancy over time by age
    '''
    if location is not None:
        if verbose:
            print(f'Loading location-specific life expectancy data for "{location}" - needed for HIV runs')
        try:
            life_expectancy = hpdata.get_life_expectancy(location=location)
            return life_expectancy
        except ValueError as E:
            errormsg = f'Could not load HIV data for requested location "{location}" ({str(E)})'
            raise NotImplementedError(errormsg)
    else:
        raise NotImplementedError('Cannot load HIV data without a specified location')


def get_hiv_data(location=None, hiv_datafile=None, art_datafile=None, verbose=False):
    '''
    Load HIV incidence and art coverage data, if provided
    ART adherance calculations use life expectancy data to infer lifetime average coverage
    rates for people in different age buckets. To give an example, suppose that ART coverage
    over 2010-2020 is given by:
        art_coverage = [0.23,0.3,0.38,0.43,0.48,0.52,0.57,0.61,0.65,0.68,0.72]
    The average ART adherence in 2010 will be higher for younger cohorts than older ones.
    Someone expected to die within a year would be given an average lifetime ART adherence
    value of 0.23, whereas someone expected to survive >10 years would be given a value of 0.506.

    Args:
        location (str): must be provided if you want to run with HIV dynamics
        hiv_datafile (str):  must be provided if you want to run with HIV dynamics
        art_datafile (str):  must be provided if you want to run with HIV dynamics
        verbose (bool):  whether to print progress

    Returns:
        hiv_inc (dict): dictionary keyed by sex, storing arrays of HIV incidence over time by age
        art_cov (dict): dictionary keyed by sex, storing arrays of ART coverage over time by age
        life_expectancy (dict): dictionary storing life expectancy over time by age
    '''
    
    if hiv_datafile is None and art_datafile is None:
        hiv_incidence_rates, art_adherence = None, None
        
    else:

        # Load data
        life_exp    = get_life_expectancy(location=location, verbose=verbose) # Load the life expectancy data (needed for ART adherance calcs)
        df_inc      = pd.read_csv(hiv_datafile) # HIV incidence
        df_art      = pd.read_csv(art_datafile) # ART coverage
    
        # Process HIV and ART data
        sex_keys = ['Male', 'Female']
        sex_key_map = {'Male': 'm', 'Female': 'f'}
    
        ## Start with incidence file
        years = df_inc['Year'].unique()
        hiv_incidence_rates = dict()
    
        # Processing
        for year in years:
            hiv_incidence_rates[year] = dict()
            for sk in sex_keys:
                sk_out = sex_key_map[sk]
                hiv_incidence_rates[year][sk_out] = np.concatenate(
                    [
                    np.array(df_inc[(df_inc['Year'] == year) & (df_inc['Sex'] == sk_out)][['Age', 'Incidence']], dtype=hpd.default_float),
                    np.array([[150,0]]) # Add another entry so that all older age groups are covered
                    ]
                )
    
        # Now compute ART adherence over time/age
        art_adherence = dict()
        years = df_art['Year'].values
        for i, year in enumerate(years):
    
            # Use the incidence file to determine which age groups we want to calculate ART coverage for
            ages_inc = hiv_incidence_rates[year]['m'][:, 0] # Read in the age groups we have HIV incidence data for
            ages_ex = life_exp[year]['m'][:, 0] # Age groups available in life expectancy file
            ages = np.intersect1d(ages_inc, ages_ex) # Age groups we want to calculate ART coverage for
    
            # Initialize age-specific ART coverage dict and start filling it in
            cov = np.zeros(len(ages), dtype=hpd.default_float)
            for j, age in enumerate(ages):
                idx = np.where(life_exp[year]['f'][:, 0] == age)[0] # Finding life expectancy for this age group/year
                this_life_exp = life_exp[year]['f'][idx, 1] # Pull out value
                last_year = int(year + this_life_exp) # Figure out the year in which this age cohort is expected to die
                year_ind = sc.findnearest(years, last_year) # Get as close to the above year as possible within the data
                if year_ind > i: # Either take the mean of ART coverage from now up until the year of death
                    cov[j] = np.mean(df_art[i:year_ind]['ART Coverage'].values)
                else: # Or, just use ART overage in this year
                    cov[j] = df_art.iloc[year_ind]['ART Coverage']
    
            art_adherence[year] = np.array([ages, cov])

    return hiv_incidence_rates, art_adherence

