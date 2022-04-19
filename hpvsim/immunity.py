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

        hpv16    = hp.genotype('16') # Make a sim with only HPV16

    '''

    def __init__(self, genotype):
        self.index     = None # Index of the variant in the sim; set later
        self.label     = None # Variant label (used as a dict key)
        self.p         = None # This is where the parameters will be stored
        self.parse(genotype=genotype) #
        self.initialized = False
        return


    def parse(self, genotype=None):
        ''' Unpack genotype information, which must be given as a string '''

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




# %% Immunity methods

def init_immunity(sim, create=False):
    ''' Initialize immunity matrices with all genotypes are in the sim'''

    # Pull out all of the circulating genotypes for cross-immunity
    ng = sim['n_genotypes']

    # If immunity values have been provided, process them
    if sim['immunity'] is None or create:

        sim['imm_kin'] = np.ones((ng, sim.npts))
        sim['immunity_map'] = dict()
        # Firstly, initialize immunity matrix with defaults. These are then overwitten with genotype-specific values below
        # Susceptibility matrix is of size sim['n_genotypes']*sim['n_genotypes']
        immunity = np.ones((ng, ng), dtype=hpd.default_float)  # Fill with defaults

        # Next, overwrite these defaults with any known immunity values about specific genotypes
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


def update_peak_immunity(people, inds, imm_pars, imm_source):
    '''
        Update immunity level

        This function updates the immunity for individuals when an infection or vaccination occurs.
            - individuals that already have immunity from a previous vaccination/infection have their immunity level boosted;
            - individuals without prior immunity are assigned an initial level drawn from a distribution. This level
                depends on whether the immunity is from a natural infection or from a vaccination (and if so, on the type of vaccine).

        Args:
            people: A people object
            inds: Array of people indices
            imm_pars: Parameters from which to draw values for quantities like ['imm_init'] - either
                        sim pars (for natural immunity) or vaccine pars
            imm_source: index of either genotype or vaccine where immunity is coming from

        Returns: None
        '''

    # Extract parameters and indices
    has_imm =  people.imm[imm_source, inds] > 0
    no_prior_imm_inds = inds[~has_imm]
    prior_imm_inds = inds[has_imm]

    if isinstance(imm_pars['imm_boost'], Iterable):
        boost = imm_pars['imm_boost'][imm_source]
    else:
        boost = imm_pars['imm_boost']

    people.peak_imm[imm_source, prior_imm_inds] *= boost

    if len(no_prior_imm_inds):
        people.peak_imm[imm_source, no_prior_imm_inds] = hpu.sample(**imm_pars['imm_init'], size=len(no_prior_imm_inds))

    people.imm[imm_source, inds] = people.peak_imm[imm_source, inds]
    people.t_imm_event[imm_source, inds] = people.t
    return


def update_immunity(people, inds):
    '''
    Step immunity levels forward in time
    '''
    t_since_boost = people.t-people.t_imm_event[:,inds].astype(hpd.default_int)
    # create n_imm_source x len(inds) array for imm_kin
    imm_kin = np.ones((people.pars['n_genotypes'], len(inds)))

    for i, imm_source in enumerate(t_since_boost):
        for j, time in enumerate(imm_source):
            imm_kin[i,j] = people.pars['imm_kin'][i, time]

    people.imm[:,inds] += imm_kin*people.peak_imm[:,inds]
    people.imm[:,inds] = np.where(people.imm[:,inds]<0, 0, people.imm[:,inds]) # Make sure immunity doesn't drop below 0
    people.imm[:,inds] = np.where([people.imm[:,inds] > people.peak_imm[:,inds]], people.peak_imm[:,inds], people.imm[:,inds]) # Make sure immunity doesn't exceed peak_imm
    return


def check_immunity(people, genotype):
    '''
    Calculate people's immunity on this timestep from prior infections + vaccination. Calculates effective immunity by
    weighting individuals immunity by source.

    There are two fundamental sources of immunity:

           (1) prior exposure: degree of protection depends on genotype
           (2) vaccination: degree of protection depends on genotype, vaccine, and time since vaccination

    '''

    # Handle parameters and indices
    pars = people.pars
    immunity = pars['immunity'][genotype,:] # cross-immunity/own-immunity scalars to be applied to immunity level
    current_imm = sc.dcp(people.imm)

    current_imm *= immunity[:, None]
    current_imm = current_imm.sum(axis=0)
    people.sus_imm[genotype,:] = current_imm

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
