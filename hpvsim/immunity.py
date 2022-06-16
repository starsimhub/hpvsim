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
        self.label     = None # Genotype label (used as a dict key)
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

        # Precompute waning - same for all genotypes
        imm_decay = sc.dcp(sim['imm_decay']['infection'])
        imm_decay['half_life'] /= sim['dt']
        sim['imm_kin'] = precompute_waning(t=sim.tvec, pars=imm_decay)

        sim['immunity_map'] = dict()
        # Firstly, initialize immunity matrix with defaults. These are then overwitten with genotype-specific values below
        # Susceptibility matrix is of size sim['n_genotypes']*sim['n_genotypes']
        immunity = np.ones((ng, ng), dtype=hpd.default_float)  # Fill with defaults

        # Next, overwrite these defaults with any known immunity values about specific genotypes
        default_cross_immunity = hppar.get_cross_immunity()
        for i in range(ng):
            sim['immunity_map'][i] = 'infection'
            label_i = sim['genotype_map'][i]
            for j in range(ng):
                label_j = sim['genotype_map'][j]
                if label_i in default_cross_immunity and label_j in default_cross_immunity:
                    immunity[j][i] = default_cross_immunity[label_j][label_i]

        sim['immunity'] = immunity

    return


def update_peak_immunity(people, inds, imm_pars, imm_source, offset=None):
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

    if len(prior_imm_inds):
        if isinstance(imm_pars['imm_boost'], Iterable):
            boost = imm_pars['imm_boost'][imm_source]
        else:
            boost = imm_pars['imm_boost']
        people.peak_imm[imm_source, prior_imm_inds] *= boost

    if len(no_prior_imm_inds):
        people.peak_imm[imm_source, no_prior_imm_inds] = hpu.sample(**imm_pars['imm_init'], size=len(no_prior_imm_inds))

    # people.imm[imm_source, inds] = people.peak_imm[imm_source, inds]
    base_t = people.t + offset if offset is not None else people.t
    people.t_imm_event[imm_source, inds] = base_t
    return


def check_immunity(people):
    '''
    Calculate people's immunity on this timestep from prior infections.
    As an example, suppose HPV16 and 18 are in the sim, and the cross-immunity matrix is:
        pars['immunity'] = np.array([[1., 0.5],
                                     [0.3, 1.]])
    This indicates that people who've had HPV18 have 50% protection against getting 16, and
    people who've had 16 have 30% protection against getting 18.
    Now suppose we have 3 people, whose immunity levels are
        people.imm = np.array([[0.9, 0.0, 0.0],
                               [0.0, 0.7, 0.0]])
    This indicates that person 1 has a prior HPV16 infection, person 2 has a prior HPV18
    infection, and person 3 has no history of infection.

    In this function, we take the dot product of pars['immunity'] and people.imm to get:
        people.sus_imm = np.array([[0.9 , 0.35, 0.  ],
                                   [0.27, 0.7 , 0.  ]])
    This indicates that the person 1 has high protection against reinfection with HPV16, and
    some (30% of 90%) protection against infection with HPV18, and so on.

    '''
    immunity = people.pars['immunity'] # cross-immunity/own-immunity scalars to be applied to immunity level
    people.sus_imm = np.dot(immunity,people.imm) # Dot product gives immunity to all genotypes
    return




#%% Methods for computing waning

def precompute_waning(t, pars=None):
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
        output = exp_decay(t, **pars)

    elif callable(form):
        output = form(t, **pars)

    else:
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
        raise NotImplementedError(errormsg)

    return output


def exp_decay(t, init_val, half_life):
    '''
    Returns an array of length t with values for the immunity at each time step after recovery
    '''
    decay_rate = np.log(2) / half_life if ~np.isnan(half_life) else 0.
    result = init_val * np.exp(-decay_rate * t, dtype=hpd.default_float)
    return result


def linear_decay(length, init_val, slope):
    ''' Calculate linear decay '''
    result = -slope*np.ones(length)
    result[0] = init_val
    return result


def linear_growth(length, slope):
    ''' Calculate linear growth '''
    return slope*np.ones(length)
