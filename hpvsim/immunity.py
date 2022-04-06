'''
Defines classes and methods for calculating immunity
'''

import numpy as np
import sciris as sc
from collections.abc import Iterable
from . import utils as hpu
from . import defaults as hpd
from . import parameters as hppar



# %% Define variant class -- all other functions are for internal use only

__all__ = ['genotype']


class genotype(sc.prettyobj):
    '''
    Add a new genotype to the sim

    Args:
        genotype (str): name of variant

    **Example**::

        hpv16    = cv.variant('16') # Make a sim with only HPV16

    '''

    def __init__(self, genotype):
        self.index     = None # Index of the variant in the sim; set later
        self.label     = None # Variant label (used as a dict key)
        self.p         = None # This is where the parameters will be stored
        self.parse(genotype=genotype) #
        self.initialized = False
        return


    def parse(self, genotype=None):
        ''' Unpack genotype information, which may be given as a string '''

        # Option 1: variants can be chosen from a list of pre-defined variants
        if isinstance(genotype, str):

            choices, mapping = hppar.get_genotype_choices()
            known_genotype_pars = hppar.get_genotype_pars()

            label = genotype.lower()

            if label in mapping:
                label = mapping[label]
                genotype_pars = known_genotype_pars[label]
            else:
                errormsg = f'The selected genotype "{genotype}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)


        else:
            errormsg = f'Could not understand {type(genotype)}, please specify as a predefined genotype:\n{sc.pp(choices, doprint=False)}'
            raise ValueError(errormsg)

        # Set label and parameters
        self.label = label
        self.p = genotype_pars

        return


    def initialize(self, sim):
        ''' Update genotype info in sim '''
        sim['genotype_pars'][self.label] = self.p  # Store the parameters
        self.index = list(sim['genotype_pars'].keys()).index(self.label) # Find where we are in the list
        sim['genotype_map'][self.index]  = self.label # Use that to populate the reverse mapping
        self.initialized = True
        return




#%% Neutralizing antibody methods


def update_peak_nab(people, inds, nab_pars, nab_source, symp=None):
    '''
    Update peak NAb level

    This function updates the peak NAb level for individuals when a NAb event occurs.
        - individuals that already have NAbs from a previous vaccination/infection have their NAb level boosted;
        - individuals without prior NAbs are assigned an initial level drawn from a distribution. This level
            depends on whether the NAbs are from a natural infection (and if so, on the infection's severity)
            or from a vaccination (and if so, on the type of vaccine).

    Args:
        people: A people object
        inds: Array of people indices
        nab_pars: Parameters from which to draw values for quantities like ['nab_init'] - either
                    sim pars (for natural immunity) or vaccine pars
        nab_source: index of either variant or vaccine where nabs are coming from
        symp: either None (if NAbs are vaccine-derived), or a dictionary keyed by 'asymp', 'mild', and 'sev' giving the indices of people with each of those symptoms

    Returns: None
    '''

    # Extract parameters and indices
    pars = people.pars
    if symp is None: # update vaccine nab
        nab_source += pars['n_variants']

    # cross_immunity = pars['immunity'][nab_source,:]
    # boost_factor = nab_pars['nab_boost'] * cross_immunity
    # boost_factor[boost_factor < 1] = 1
    if isinstance(nab_pars['nab_boost'], Iterable):
        boost = nab_pars['nab_boost'][nab_source]
    else:
        boost = nab_pars['nab_boost']

    people.peak_nab[nab_source, inds] *= boost
    # people.peak_nab[:, inds] *= boost_factor[:,None]

    has_nabs = people.nab[nab_source, inds] > 0
    no_prior_nab_inds = inds[~has_nabs]
    prior_nab_inds = inds[has_nabs]

    if len(no_prior_nab_inds):
        progs = pars['prognoses']
        nab_inds = np.fromiter((np.nonzero(progs['age_cutoffs'] <= this_age)[0][-1] for this_age in people.age[no_prior_nab_inds]),
                               dtype=cvd.default_int,
                               count=len(no_prior_nab_inds))  # Convert ages to indices
        rel_nabs = progs['nab_level'][nab_inds]  # Relative level of nabs
        init_nab = cvu.sample(**nab_pars['nab_init'], size=len(no_prior_nab_inds))

        no_prior_nab = (2 ** init_nab) * rel_nabs

        if symp is not None: # natural infection
            prior_symp = np.full(pars['pop_size'], np.nan)
            prior_symp[symp['asymp']] = pars['rel_imm_symp']['asymp']
            prior_symp[symp['mild']] = pars['rel_imm_symp']['mild']
            prior_symp[symp['sev']] = pars['rel_imm_symp']['severe']
            prior_symp[prior_nab_inds] = np.nan
            prior_symp = prior_symp[~np.isnan(prior_symp)]
            # Applying symptom scaling and a normalization factor to the NAbs
            norm_factor = 1 + nab_pars['nab_eff']['alpha_inf_diff']
            no_prior_nab = no_prior_nab * prior_symp * norm_factor

        people.peak_nab[nab_source, no_prior_nab_inds] = no_prior_nab

    # Update time of nab event
    people.t_nab_event[nab_source, inds] = people.t

    return


def update_nab(people, inds):
    '''
    Step NAb levels forward in time
    '''
    t_since_boost = people.t-people.t_nab_event[:,inds].astype(cvd.default_int)
    # create n_nab_source x len(inds) array for nab_kin
    nab_kin = np.ones((people.pars['n_variants']+ len(people.pars['vaccine_map']), len(inds)))

    for i, nab_source in enumerate(t_since_boost):
        for j, time in enumerate(nab_source):
            nab_kin[i,j] = people.pars['nab_kin'][i, time]

    people.nab[:,inds] += nab_kin*people.peak_nab[:,inds]
    people.nab[:,inds] = np.where(people.nab[:,inds]<0, 0, people.nab[:,inds]) # Make sure nabs don't drop below 0
    people.nab[:,inds] = np.where([people.nab[:,inds] > people.peak_nab[:,inds]], people.peak_nab[:,inds], people.nab[:,inds]) # Make sure nabs don't exceed peak_nab
    return


def calc_VE(nab, ax, pars):
    '''
        Convert NAb levels to immunity protection factors, using the functional form
        given in this paper: https://doi.org/10.1101/2021.03.09.21252641

        Args:
            nab  (arr)  : an array of effective NAb levels (i.e. actual NAb levels, scaled by cross-immunity)
            ax   (str)  : axis of protection; can be 'sus', 'symp' or 'sev', corresponding to the efficacy of protection against infection, symptoms, and severe disease respectively
            pars (dict) : dictionary of parameters for the vaccine efficacy

        Returns:
            an array the same size as NAb, containing the immunity protection factors for the specified axis
         '''

    choices = ['sus', 'symp', 'sev']
    if ax not in choices:
        errormsg = f'Choice {ax} not in list of choices: {sc.strjoin(choices)}'
        raise ValueError(errormsg)

    if ax == 'sus':
        alpha = pars['alpha_inf']
        beta = pars['beta_inf']
    elif ax == 'symp':
        alpha = pars['alpha_symp_inf']
        beta = pars['beta_symp_inf']
    else:
        alpha = pars['alpha_sev_symp']
        beta = pars['beta_sev_symp']

    exp_lo = np.exp(alpha) * nab**beta
    output = exp_lo/(1+exp_lo) # Inverse logit function
    return output


def calc_VE_symp(nab, pars):
    '''
    Converts NAbs to marginal VE against symptomatic disease
    '''

    exp_lo_inf = np.exp(pars['alpha_inf']) * nab**pars['beta_inf']
    inv_lo_inf = exp_lo_inf / (1 + exp_lo_inf)

    exp_lo_symp_inf = np.exp(pars['alpha_symp_inf']) * nab**pars['beta_symp_inf']
    inv_lo_symp_inf = exp_lo_symp_inf / (1 + exp_lo_symp_inf)

    VE_symp = 1 - ((1 - inv_lo_inf)*(1 - inv_lo_symp_inf))
    return VE_symp




# %% Immunity methods

def init_immunity(sim, create=False):
    ''' Initialize immunity matrices with all genotypes are in the sim'''

    # Pull out all of the circulating genotypes for cross-immunity
    ng = sim['n_genotypes']

    # If immunity values have been provided, process them
    if sim['immunity'] is None or create:

        sim['nab_kin'] = np.ones((ng, sim.npts))
        sim['immunity_map'] = dict()
        # Firstly, initialize immunity matrix with defaults. These are then overwitten with genotype-specific values below
        # Susceptibility matrix is of size sim['n_genotypes']*sim['n_genotypes']
        immunity = np.ones((ng, ng), dtype=hpd.default_float)  # Fill with defaults

        # Next, overwrite these defaults with any known immunity values about specific variants
        default_cross_immunity = hppar.get_cross_immunity()
        for i in range(ng):
            sim['immunity_map'][i] = 'infection'
            sim['imm_kin'][i, :] = precompute_waning(length=sim.npts, pars=sim['imm_decay']['infection'])
            label_i = sim['genotype_map'][i]
            for j in range(ng):
                label_j = sim['genotype_map'][j]
                if label_i in default_cross_immunity and label_j in default_cross_immunity:
                    immunity[j][i] = default_cross_immunity[label_j][label_i]

        sim['immunity'] = immunity

    return


def check_immunity(people, variant):
    '''
    Calculate people's immunity on this timestep from prior infections + vaccination. Calculates effective NAbs by
    weighting individuals NAbs by source and then calculating efficacy.

    There are two fundamental sources of immunity:

           (1) prior exposure: degree of protection depends on variant, prior symptoms, and time since recovery
           (2) vaccination: degree of protection depends on variant, vaccine, and time since vaccination

    '''

    # Handle parameters and indices
    pars = people.pars
    immunity = pars['immunity'][variant,:] # cross-immunity/own-immunity scalars to be applied to NAb level before computing efficacy
    nab_eff = pars['nab_eff']
    current_nabs = sc.dcp(people.nab)

    current_nabs *= immunity[:, None]
    current_nabs = current_nabs.sum(axis=0)
    people.sus_imm[variant,:] = calc_VE(current_nabs, 'sus', nab_eff)
    people.symp_imm[variant,:] = calc_VE(current_nabs, 'symp', nab_eff)
    people.sev_imm[variant,:] = calc_VE(current_nabs, 'sev', nab_eff)

    return



#%% Methods for computing waning

def precompute_waning(length, pars=None):
    '''
    Process functional form and parameters into values:

        - 'exp_decay'   : exponential decay. Parameters should be init_val and half_life (half_life can be None/nan)
        - 'linear_decay': linear decay

    A custom function can also be supplied.

    Args:
        length (float): length of array to return, i.e., for how long waning is calculated
        pars (dict): passed to individual immunity functions

    Returns:
        array of length 'length' of values
    '''

    pars = sc.dcp(pars)
    form = pars.pop('form')
    choices = [
        'exp_decay',
    ]

    # Process inputs
    if form is None or form == 'exp_decay':
        if pars['half_life'] is None: pars['half_life'] = np.nan
        output = exp_decay(length, **pars)

    elif callable(form):
        output = form(length, **pars)

    else:
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
        raise NotImplementedError(errormsg)

    return output



def exp_decay(length, init_val, half_life, delay=None):
    '''
    Returns an array of length t with values for the immunity at each time step after recovery
    '''
    length = length+1
    decay_rate = np.log(2) / half_life if ~np.isnan(half_life) else 0.
    if delay is not None:
        t = np.arange(length-delay, dtype=hpd.default_int)
        growth = linear_growth(delay, init_val/delay)
        decay = init_val * np.exp(-decay_rate * t)
        result = np.concatenate([growth, decay], axis=None)
    else:
        t = np.arange(length, dtype=hpd.default_int)
        result = init_val * np.exp(-decay_rate * t)
    return np.diff(result)


def linear_decay(length, init_val, slope):
    ''' Calculate linear decay '''
    result = -slope*np.ones(length)
    result[0] = init_val
    return result


def linear_growth(length, slope):
    ''' Calculate linear growth '''
    return slope*np.ones(length)
