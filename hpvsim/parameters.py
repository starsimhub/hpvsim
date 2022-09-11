'''
Set the parameters for hpvsim.
'''

import numpy as np
import sciris as sc
from .settings import options as hpo # For setting global options
from . import misc as hpm
from . import defaults as hpd
from .data import loaders as hpdata

__all__ = ['make_pars', 'reset_layer_pars']


def make_pars(**kwargs):
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
    pars['n_agents']        = 20e3      # Number of agents
    pars['pop_scale']       = None      # How much to scale the population
    pars['network']         = 'default'  # What type of sexual network to use -- 'random', 'basic', other options TBC
    pars['location']        = None      # What location to load data from -- default Seattle
    pars['lx']              = None      # Proportion of people alive at the beginning of age interval x
    pars['birth_rates']     = None      # Birth rates, loaded below

    # Initialization parameters
    pars['init_hpv_prev']   = hpd.default_init_prev # Initial prevalence
    pars['init_hpv_dist'] = None  # Initial type distribution

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
    pars['partners']        = None  # The number of concurrent sexual partners for each partnership type
    pars['acts']            = None  # The number of sexual acts for each partnership type per year
    pars['condoms']         = None  # The proportion of acts in which condoms are used for each partnership type
    pars['layer_probs']     = None  # Proportion of the population in each partnership type
    pars['dur_pship']       = None  # Duration of partnerships in each partnership type
    pars['mixing']          = None  # Mixing matrices for storing age differences in partnerships
    pars['n_partner_types'] = 1  # Number of partnership types - reset below

    # Basic disease transmission parameters
    pars['beta_dist']       = dict(dist='neg_binomial', par1=1.0, par2=1.0, step=0.01) # Distribution to draw individual level transmissibility TODO does this get used? if not remove.
    pars['beta']            = 0.25  # Per-act transmission probability; absolute value, calibrated
    pars['transf2m']        = 1.0   # Relative transmissibility of receptive partners in penile-vaginal intercourse; baseline value
    pars['transm2f']        = 3.69  # Relative transmissibility of insertive partners in penile-vaginal intercourse; based on https://doi.org/10.1038/srep10986: "For vaccination types, the risk of male-to-female transmission was higher than that of female-to-male transmission"

    # Parameters for disease progression
    pars['sero']  = 2.5 # parameter used as the growth rate within a logistic function that maps durations to seroconversion probabilities
    pars['severity_dist'] = dict(dist='lognormal', par1=None, par2=0.1) # Distribution of individual disease severity. Par1 is set to None because the mean is determined as a function of genotype and disease duration
    pars['clinical_cutoffs']    = {'cin1': 0.33, 'cin2':0.67, 'cin3':0.99} # Parameters the control the clinical cliassification of dysplasia
    pars['cancer_treat_prob'] = 0.1 # probability of receiving cancer treatment given symptom detection
    pars['hpv_control_prob']    = 0.44 # Probability that HPV is controlled latently vs. cleared
    pars['hpv_reactivation'] = dict(
        age_cutoffs             = np.array([0,       30,          50]),      # Age cutoffs (lower limits)
        hpv_reactivation_probs  = np.array([0.0001,    0.05,        0.04]),      # made this up, need to parameterize somehow
    )

    # Parameters used to calculate immunity
    pars['imm_init']        = dict(dist='beta', par1=5, par2=3)  # beta distribution for initial level of immunity following infection clearance
    pars['imm_decay']       = dict(form=None)  # decay rate, with half life in years
    pars['imm_kin']         = None  # Constructed during sim initialization using the nab_decay parameters
    pars['imm_boost']       = []  # Multiplicative factor applied to a person's immunity levels if they get reinfected. No data on this, assumption.
    pars['immunity']        = None  # Matrix of immunity and cross-immunity factors, set by init_immunity() in immunity.py
    pars['immunity_map']    = None  # dictionary mapping the index of immune source to the type of immunity (vaccine vs natural)

    # all genotype properties get populated by user in init_genotypes()
    pars['genotypes']       = []  # Genotypes of the virus; populated by the user below
    pars['genotype_map']    = dict()  # Reverse mapping from number to genotype key
    pars['genotype_pars']   = sc.objdict()  # Populated just below

    # Genotype parameters
    pars['n_genotypes']     = 1 # The number of genotypes circulating in the population
    pars['n_imm_sources']   = 1 # The number of immunity sources circulating in the population

    # Vaccine parameters
    pars['vaccine_pars']    = dict()  # Vaccines that are being used; populated during initialization
    pars['vaccine_map']     = dict()  # Reverse mapping from number to vaccine key

    # Screening and treatment parameters
    pars['screen_pars']     = dict()  # Screening method that is being used; populated during initialization
    pars['treat_pars'] = dict()  # Treatment method that is being used; populated during initialization

    # Durations
    pars['dur_cin1_clear']  = dict(dist='lognormal', par1=0.5, par2=0.5)  # Time to clearance from CIN1
    pars['dur_cin2_clear']  = dict(dist='lognormal', par1=1.0, par2=0.5)  # Time to clearance from CIN2
    pars['dur_cin3_clear']  = dict(dist='lognormal', par1=1.5, par2=0.5)  # Time to clearance from CIN3
    pars['dur_cancer']      = dict(dist='lognormal', par1=12.0, par2=3.0)  # Duration of untreated invasive cerival cancer before death

    # Parameters determining relative transmissibility at each stage of disease
    pars['rel_trans'] = {}
    pars['rel_trans']['precin']   = 1 # Baseline value
    pars['rel_trans']['cin1']   = 1 # Baseline assumption, can be adjusted during calibration
    pars['rel_trans']['cin2']   = 1 # Baseline assumption, can be adjusted during calibration
    pars['rel_trans']['cin3']   = 1 # Baseline assumption, can be adjusted during calibration
    pars['rel_trans']['cancerous']   = 0.5 # Baseline assumption, can be adjusted during calibration

    # Efficacy of protection
    pars['eff_condoms']     = 0.7  # The efficacy of condoms; https://www.nejm.org/doi/10.1056/NEJMoa053284?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200www.ncbi.nlm.nih.gov

    # HIV parameters
    pars['hiv_infection_rates'] = None # loaded below if model_hiv == True
    pars['art_initiation_rates'] = None # loaded below if model_hiv == True

    # Events and interventions
    pars['interventions']   = []   # The interventions present in this simulation; populated by the user
    pars['analyzers']       = []   # Custom analysis functions; populated by the user
    pars['timelimit']       = None # Time limit for the simulation (seconds)
    pars['stopping_func']   = None # A function to call to stop the sim partway through

    # Population distribution of the World Standard Population, used to calculate age-standardised rates (ASR) of incidence
    pars['standard_pop']    = np.array([[  0,   5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75,    80,    85, 100],
                                        [.12, .10, .09, .09, .08, .08, .06, .06, .06, .06, .05, .04, .04, .03, .02, .01, 0.005, 0.005,   0]])

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
        # layer_probs = dict(m=0.7, c=0.4, o=0.05),   # Default proportion of the population in each layer
        dur_pship   = dict(m=dict(dist='normal_pos', par1=8,par2=3),
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

def get_screen_choices():
    '''
    Define valid pre-defined screening names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'hpv':  ['hpv', 'hpvdna'],
        'hpv1618': ['hpv1618', 'hpvgenotyping'],
        'via': ['via', 'visualinspection'],
        'via_triage': ['via_triage'],
    }
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

    pars.hpv16 = sc.objdict()
    pars.hpv16.dur_precin  = dict(dist='lognormal', par1=2.618, par2=0.5)
                                    # Made the distribution wider to accommodate varying means
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
                                    # https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv16.dur_dysp     = dict(dist='lognormal', par1=4.5, par2=4.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv16.dysp_rate    = 1.5 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv16.prog_rate    = 0.79 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv16.prog_time    = 4.4  # Point of inflection in logistic function
    pars.hpv16.rel_beta     = 1  # Baseline relative transmissibility, other genotypes are relative to this
    pars.hpv16.imm_boost    = 1.0 # TODO: look for data

    pars.hpv18 = sc.objdict()
    pars.hpv18.dur_precin  = dict(dist='lognormal', par1=2.16, par2=0.5)
                                    # Made the distribution wider to accommodate varying means
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
                                    # https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv18.dur_dysp     = dict(dist='lognormal', par1=3.16, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv18.dysp_rate    = 1.2 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv18.prog_rate    = 1.02 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv18.prog_time    = 7.82  # Point of inflection in logistic function
    pars.hpv18.rel_beta     = 0.72  # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv18.imm_boost    = 1.0 # TODO: look for data

    pars.hpv31 = sc.objdict()
    pars.hpv31.dur_precin  = dict(dist='lognormal', par1=2.5197, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv31.dur_dysp     = dict(dist='lognormal', par1=1.35, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv31.dysp_rate    = 0.5 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv31.prog_rate    = 0.179 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv31.prog_time    = 14.12  # Point of inflection in logistic function
    pars.hpv31.rel_beta     = 0.94 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv31.imm_boost    = 1.0 # TODO: look for data

    pars.hpv33 = sc.objdict()
    pars.hpv33.dur_precin  = dict(dist='lognormal', par1=2.3226, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv33.dur_dysp     = dict(dist='lognormal', par1=14.12, par2=3.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv33.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv33.prog_rate    = 0.24 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv33.prog_time    = 8.46  # Point of inflection in logistic function
    pars.hpv33.rel_beta     = 0.26 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv33.imm_boost    = 1.0 # TODO: look for data

    pars.hpv35 = sc.objdict()
    pars.hpv35.dur_precin  = dict(dist='lognormal', par1=2.5, par2=1.0)
    pars.hpv35.dur_dysp     = dict(dist='lognormal', par1=4.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv35.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv35.prog_rate    = 0.25 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv35.prog_time    = 7.5  # Point of inflection in logistic function
    pars.hpv35.rel_beta     = 0.5 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv35.imm_boost    = 1.0 # TODO: look for data

    pars.hpv45 = sc.objdict()
    pars.hpv45.dur_precin  = dict(dist='lognormal', par1=2.84, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv45.dur_dysp     = dict(dist='lognormal', par1=3.776, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv45.dysp_rate    = 1.2 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv45.prog_rate    = 0.925 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv45.prog_time    = 3.46  # Point of inflection in logistic function
    pars.hpv45.rel_beta     = 0.77 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv45.imm_boost    = 1.0 # TODO: look for data

    pars.hpv51 = sc.objdict()
    pars.hpv51.dur_precin  = dict(dist='lognormal', par1=2.0, par2=1.0)
    pars.hpv51.dur_dysp     = dict(dist='lognormal', par1=1.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv51.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv51.prog_rate    = 0.5 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv51.prog_time    = 8.  # Point of inflection in logistic function
    pars.hpv51.rel_beta     = 0.5 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv51.imm_boost    = 1.0 # TODO: look for data

    pars.hpv52 = sc.objdict()
    pars.hpv52.dur_precin  = dict(dist='lognormal', par1=2.3491, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv52.dur_dysp     = dict(dist='lognormal', par1=1.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv52.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv52.prog_rate    = 0.526 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv52.prog_time    = 8.96  # Point of inflection in logistic function
    pars.hpv52.rel_beta     = 0.623 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv52.imm_boost    = 1.0 # TODO: look for data

    pars.hpv56 = sc.objdict()
    pars.hpv56.dur_precin  = dict(dist='lognormal', par1=2.0, par2=1.0)
    pars.hpv56.dur_dysp     = dict(dist='lognormal', par1=3.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv56.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv56.prog_rate    = 0.8 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv56.prog_time    = 10  # Point of inflection in logistic function
    pars.hpv56.rel_beta     = 0.6 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv56.imm_boost    = 1.0 # TODO: look for data

    pars.hpv58 = sc.objdict()
    pars.hpv58.dur_precin  = dict(dist='lognormal', par1=2.3491, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv58.dur_dysp     = dict(dist='lognormal', par1=3.0, par2=2.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv58.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv58.prog_rate    = 0.8 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv58.prog_time    = 10  # Point of inflection in logistic function
    pars.hpv58.rel_beta     = 0.6 # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
    pars.hpv58.imm_boost    = 1.0 # TODO: look for data

    pars.hpv6 = sc.objdict()
    pars.hpv6.dur_precin   = dict(dist='lognormal', par1=1.8245, par2=1.0)
                                    # https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv6.dur_dysp      = dict(dist='lognormal', par1=0.5, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv6.dysp_rate     = 0.01 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv6.prog_rate     = 0.01 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv6.prog_time     = 30  # Point of inflection in logistic function
    pars.hpv6.rel_beta      = 0.8 # Relative transmissibility, generally calibrated to match available type distributions
    pars.hpv6.imm_boost     = 1.0 # TODO: look for data

    pars.hpv11 = sc.objdict()
    pars.hpv11.dur_precin  = dict(dist='lognormal', par1=1.8718, par2=1.0)
                                    # https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv11.dur_dysp     = dict(dist='lognormal', par1=4.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv11.dysp_rate    = 0.8 # Rate of progression to dysplasia. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv11.prog_rate    = 0.8 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
    pars.hpv11.prog_time    = 30  # Point of inflection in logistic function
    pars.hpv11.rel_beta     = 0.5 # Relative transmissibility, generally calibrated to match available type distributions
    pars.hpv11.imm_boost    = 1.0 # TODO: look for data

    return _get_from_pars(pars, default, key=genotype, defaultkey='hpv16')


def get_cross_immunity(default=False, genotype=None):
    '''
    Get the cross immunity between each genotype in a sim
    '''
    med_imm = 0.3
    high_imm = 0.5
    pars = dict(
        # All values based roughly on https://academic.oup.com/jnci/article/112/10/1030/5753954 or assumptions
        hpv16 = dict(
            hpv16  = 1.0, # Default for own-immunity
            hpv18 = high_imm,
            hpv31  = high_imm,
            hpv33 = high_imm,
            hpv35 = high_imm,
            hpv45 = high_imm,
            hpv51 = med_imm,
            hpv52 = med_imm,
            hpv56 = med_imm,
            hpv58 = med_imm,
            hpv6 = med_imm,
            hpv11 = med_imm,
        ),

        hpv18 = dict(
            hpv16=high_imm,
            hpv18=1.0,  # Default for own-immunity
            hpv31=high_imm,
            hpv33=high_imm,
            hpv35=high_imm,
            hpv45=high_imm,
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=med_imm,
        ),

        hpv31=dict(
            hpv16=high_imm,
            hpv18=high_imm,
            hpv31=1.0,  # Default for own-immunity
            hpv33=high_imm,
            hpv35=high_imm,
            hpv45=high_imm,
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=med_imm,
        ),

        hpv33=dict(
            hpv16=high_imm,
            hpv18=high_imm,
            hpv31=high_imm,
            hpv33=1.0,  # Default for own-immunity
            hpv35=high_imm,
            hpv45=high_imm,
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=med_imm,
        ),

        hpv35=dict(
            hpv16=high_imm,
            hpv18=high_imm,
            hpv31=high_imm,
            hpv33=high_imm,
            hpv35=1.0,  # Default for own-immunity
            hpv45=high_imm,
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=med_imm,
        ),

        hpv45=dict(
            hpv16=high_imm,
            hpv18=high_imm,
            hpv31=high_imm,
            hpv33=high_imm,
            hpv35=high_imm,
            hpv45=1.0,  # Default for own-immunity
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=med_imm,
        ),

        hpv51=dict(
            hpv16=med_imm,
            hpv18=med_imm,
            hpv31=med_imm,
            hpv33=med_imm,
            hpv35=med_imm,
            hpv45=med_imm,
            hpv51=1.0,  # Default for own-immunity
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=med_imm,
        ),

        hpv52=dict(
            hpv16=med_imm,
            hpv18=med_imm,
            hpv31=med_imm,
            hpv33=med_imm,
            hpv35=med_imm,
            hpv45=med_imm,
            hpv51=med_imm,
            hpv52=1.0,  # Default for own-immunity
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=med_imm,
        ),

        hpv56=dict(
            hpv16=med_imm,
            hpv18=med_imm,
            hpv31=med_imm,
            hpv33=med_imm,
            hpv35=med_imm,
            hpv45=med_imm,
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=1.0,  # Default for own-immunity
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=med_imm,
        ),

        hpv58=dict(
            hpv16=med_imm,
            hpv18=med_imm,
            hpv31=med_imm,
            hpv33=med_imm,
            hpv35=med_imm,
            hpv45=med_imm,
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=1.0,  # Default for own-immunity
            hpv6=med_imm,
            hpv11=med_imm,
        ),


        hpv6=dict(
            hpv16=med_imm,
            hpv18=med_imm,
            hpv31=med_imm,
            hpv33=med_imm,
            hpv35=med_imm,
            hpv45=med_imm,
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=1.0,  # Default for own-immunity
            hpv11=med_imm,
        ),

        hpv11=dict(
            hpv16=med_imm,
            hpv18=med_imm,
            hpv31=med_imm,
            hpv33=med_imm,
            hpv35=med_imm,
            hpv45=med_imm,
            hpv51=med_imm,
            hpv52=med_imm,
            hpv56=med_imm,
            hpv58=med_imm,
            hpv6=med_imm,
            hpv11=1.0,  # Default for own-immunity
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


def get_vaccine_genotype_pars(default=False, vaccine=None):
    '''
    Define the cross-immunity of each vaccine against each genotype
    '''
    pars = dict(

        default = dict(
            hpv16=1,
            hpv18=1,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv35=0,  # Assumption
            hpv45=0,  # Assumption
            hpv51=0,  # Assumption
            hpv52=0,  # Assumption
            hpv56=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
        ),

        bivalent = dict(
            hpv16=1,
            hpv18=1,  # Assumption
            hpv31=0.5,  # Assumption
            hpv33=0.5,  # Assumption
            hpv35=0.5,  # Assumption
            hpv45=0.5,  # Assumption
            hpv51=0.3,  # Assumption
            hpv52=0.3,  # Assumption
            hpv56=0.3,  # Assumption
            hpv58=0.3,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
        ),

        quadrivalent=dict(
            hpv16=1,
            hpv18=1,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv35=0,  # Assumption
            hpv45=0,  # Assumption
            hpv51=0,  # Assumption
            hpv52=0,  # Assumption
            hpv56=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=1,  # Assumption
            hpv11=1,  # Assumption
        ),

        nonavalent=dict(
            hpv16=1,
            hpv18=1,  # Assumption
            hpv31=1,  # Assumption
            hpv33=1,  # Assumption
            hpv35=0,  # Assumption
            hpv45=1,  # Assumption
            hpv51=0,  # Assumption
            hpv52=1,  # Assumption
            hpv56=0,  # Assumption
            hpv58=1,  # Assumption
            hpv6=1,  # Assumption
            hpv11=1,  # Assumption
        ),
    )

    pars['bivalent_2dose'] = pars['bivalent']
    pars['bivalent_3dose'] = pars['bivalent']
    pars['quadrivalent_2dose'] = pars['quadrivalent']
    pars['quadrivalent_3dose'] = pars['quadrivalent']
    pars['nonavalent_2dose'] = pars['nonavalent']
    pars['nonavalent_3dose'] = pars['nonavalent']

    return _get_from_pars(pars, default=default, key=vaccine)


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


def get_screen_pars(screen=None):
    '''
    Define the parameters for each screen method
    should extract test positivity, inadequacy
    '''

    pars = dict(
        hpv = dict(
            by_genotype=True,
            test_positivity=dict(
                precin=dict(
                    hpv16=0.55,
                    hpv18=0.55,
                    hpv31=0.55,
                    hpv33=0.55,
                    hpv35=0.55,
                    hpv45=0.55,
                    hpv51=0.55,
                    hpv52=0.55,
                    hpv56=0.55,
                    hpv58=0.55,
                    hpv6=0,
                    hpv11=0,
                ),
                cin1=dict(
                    hpv16=0.8415,
                    hpv18=0.8415,
                    hpv31=0.8415,
                    hpv33=0.8415,
                    hpv35=0.8415,
                    hpv45=0.8415,
                    hpv51=0.8415,
                    hpv52=0.8415,
                    hpv56=0.8415,
                    hpv58=0.8415,
                    hpv6=0,
                    hpv11=0,
                ),
                cin2=dict(
                    hpv16=0.93,
                    hpv18=0.93,
                    hpv31=0.93,
                    hpv33=0.93,
                    hpv35=0.93,
                    hpv45=0.93,
                    hpv51=0.93,
                    hpv52=0.93,
                    hpv56=0.93,
                    hpv58=0.93,
                    hpv6=0,
                    hpv11=0,
                ),
                cin3=dict(
                    hpv16=0.984,
                    hpv18=0.984,
                    hpv31=0.984,
                    hpv33=0.984,
                    hpv35=0.984,
                    hpv45=0.984,
                    hpv51=0.984,
                    hpv52=0.984,
                    hpv56=0.984,
                    hpv58=0.984,
                    hpv6=0,
                    hpv11=0,
                ),
                cancerous=dict(
                    hpv16=0.984,
                    hpv18=0.984,
                    hpv31=0.984,
                    hpv33=0.984,
                    hpv35=0.984,
                    hpv45=0.984,
                    hpv51=0.984,
                    hpv52=0.984,
                    hpv56=0.984,
                    hpv58=0.984,
                    hpv6=0,
                    hpv11=0,
                ),
            ),
            inadequacy=0,
        ),

        hpv1618 = dict(
            by_genotype=True,
            test_positivity=dict(
                precin=dict(
                    hpv16=1,
                    hpv18=1,
                    hpv31=0,
                    hpv33=0,
                    hpv35=0,
                    hpv45=0,
                    hpv51=0,
                    hpv52=0,
                    hpv56=0,
                    hpv58=0,
                    hpv6=0,
                    hpv11=0,
                ),
                cin1=dict(
                    hpv16=1,
                    hpv18=1,
                    hpv31=0,
                    hpv33=0,
                    hpv35=0,
                    hpv45=0,
                    hpv51=0,
                    hpv52=0,
                    hpv56=0,
                    hpv58=0,
                    hpv6=0,
                    hpv11=0,
                ),
                cin2=dict(
                    hpv16=1,
                    hpv18=1,
                    hpv31=0,
                    hpv33=0,
                    hpv35=0,
                    hpv45=0,
                    hpv51=0,
                    hpv52=0,
                    hpv56=0,
                    hpv58=0,
                    hpv6=0,
                    hpv11=0,
                ),
                cin3=dict(
                    hpv16=1,
                    hpv18=1,
                    hpv31=0,
                    hpv33=0,
                    hpv35=0,
                    hpv45=0,
                    hpv51=0,
                    hpv52=0,
                    hpv56=0,
                    hpv58=0,
                    hpv6=0,
                    hpv11=0,
                ),
                cancerous=dict(
                    hpv16=1,
                    hpv18=1,
                    hpv31=0,
                    hpv33=0,
                    hpv35=0,
                    hpv45=0,
                    hpv51=0,
                    hpv52=0,
                    hpv56=0,
                    hpv58=0,
                    hpv6=0,
                    hpv11=0,
                ),
            ),
            inadequacy=0,
        ),

        via=dict(
            by_genotype=False,
            test_positivity=dict(
                precin=0.25,
                cin1=0.3,
                cin2=0.45,
                cin3=0.41,
                cancerous=0.6,
            ),
            inadequacy=0,
        ),
        via_triage=dict(
            by_genotype=False,
            test_positivity=dict(
                precin=0.98,
                cin1=0.97,
                cin2=0.89,
                cin3=0.79,
                cancerous=0.4,
            ),
            inadequacy=0,
        ),
    )

    return _get_from_pars(pars, key=screen)


def get_hiv_rates(location=None, verbose=False, die=True):
    '''
    Get HIV incidence data by location if provided, or use default

    Args:
        location (str):  location; if none specified, use default value for XXX
        verbose (bool):  whether to print progress

    Returns:
        hiv_inc (dict): dictionary keyed by sex, storing arrays of HIV incidence over time by age
    '''

    hiv_incidence_rates = hpd.default_hiv_incidence_rates
    if location is not None:
        if verbose:
            print(f'Loading location-specific HIV incidence data for "{location}"')
        try:
            hiv_incidence_rates = hpdata.get_hiv_incidence_rates(location=location)
        except ValueError as E:
            warnmsg = f'Could not load HIV incidence data for requested location "{location}" ({str(E)}), using default'
            hpm.warn(warnmsg, die=die)

    return hiv_incidence_rates