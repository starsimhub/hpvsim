'''
Defines functions for making the population.
'''

#%% Imports
from re import U
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as hpu
from . import misc as hpm
# from . import base as cvb
from . import data as hpdata
from . import defaults as hpd
# from . import parameters as cvpar
from . import people as hpppl


# # Specify all externally visible functions this file defines
# __all__ = ['make_people', 'make_randpop', 'make_random_contacts']


def make_people(sim, popdict=None, reset=False, verbose=None, use_age_data=True,
                sex_ratio=0.5, dispersion=None, microstructure='random', **kwargs):
    '''
    Make the people for the simulation.

    Usually called via ``sim.initialize()``.

    Args:
        sim      (Sim)  : the simulation object; population parameters are taken from the sim object
        popdict  (any)  : if supplied, use this population dictionary instead of generating a new one; can be a dict or People object
        reset    (bool) : whether to force population creation even if self.popdict/self.people exists
        verbose  (bool) : level of detail to print
        kwargs   (dict) : passed to make_randpop()

    Returns:
        people (People): people
    '''

    # Set inputs and defaults
    pop_size = int(sim['pop_size']) # Shorten
    if verbose is None:
        verbose = sim['verbose']

    # If a people object or popdict is supplied, use it
    if sim.people and not reset:
        sim.people.initialize(sim_pars=sim.pars)
        return sim.people # If it's already there, just return
    elif sim.popdict and popdict is None:
        popdict = sim.popdict # Use stored one
        sim.popdict = None # Once loaded, remove

    if popdict is None:

        pop_size = int(sim['pop_size']) # Number of people

        # Load age data and household demographics based on 2018 Seattle demographics by default, or country if available
        age_data = hpd.default_age_data
        location = sim['location']
        if location is not None:
            if sim['verbose']:
                print(f'Loading location-specific data for "{location}"')
            if use_age_data:
                try:
                    age_data = hpdata.get_age_distribution(location)
                except ValueError as E:
                    warnmsg = f'Could not load age data for requested location "{location}" ({str(E)}), using default'
                    hpm.warn(warnmsg)

        # Set people's sexes, ages, and sexual behavior/characteristics
        uids           = np.arange(pop_size, dtype=hpd.default_int)
        sexes          = np.random.binomial(1, sex_ratio, pop_size)
        age_data_min   = age_data[:,0]
        age_data_max   = age_data[:,1] + 1 # Since actually e.g. 69.999
        age_data_range = age_data_max - age_data_min
        age_data_prob   = age_data[:,2]
        age_data_prob   /= age_data_prob.sum() # Ensure it sums to 1
        age_bins        = hpu.n_multinomial(age_data_prob, pop_size) # Choose age bins
        ages            = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(pop_size) # Uniformly distribute within this age bin
        debuts          = hpu.sample(**sim['debut'], size=pop_size)
        partners        = partner_count(pop_size=pop_size, layer_keys=sim['partners'].keys(), means=sim['partners'].values(), dispersion=dispersion)

        # Store output
        popdict = {}
        popdict['uid'] = uids
        popdict['age'] = ages
        popdict['sex'] = sexes
        popdict['debut'] = debuts
        popdict['partners'] = partners

        # Create the contacts; TODO should this be in a separate function?
        is_active = ages>debuts                 # Whether or not people have ever been sexually active
        active_inds = hpu.true(ages>debuts)     # Indices of sexually experienced people
        n_active = sum(is_active)               # Number of sexually experienced people

        if microstructure == 'random':
            contacts = dict()
            for lkey,n in sim['partners'].items():
                active_inds_layer = hpu.binomial_filter(sim['layer_probs'][lkey], active_inds)
                durations = sim['dur_pship'][lkey]
                contacts[lkey] = make_random_contacts(p_count=partners[lkey], sexes=sexes, n=n, durations=durations, mapping=active_inds_layer, **kwargs)
        else: # pragma: no cover
            errormsg = f'Microstructure type "{microstructure}" not found; choices are random or TBC'
            raise NotImplementedError(errormsg)

        popdict['contacts']   = contacts
        popdict['layer_keys'] = list(partners.keys())

    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    people = hpppl.People(sim.pars, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'], debut=popdict['debut'], partners=popdict['partners'], contacts=popdict['contacts']) # List for storing the people

    sc.printv(f'Created {pop_size} people, average age {people.age.mean():0.2f} years', 2, verbose)

    return people


def partner_count(pop_size=None, layer_keys=None, means=None, sample=True, dispersion=None):
    '''
    Assign each person a number of concurrent partners (either desired or actual)
    Args:
        pop_size    (int)   : number of people
        layer_keys  (list)  : list of layers
        means       (dict)  : dictionary keyed by layer_keys with mean number of partners per layer
        sample      (bool)  : whether or not to sample the number of partners
        dispersion  (any)   : if not None, will use negative binomial sampling

    Returns:
        p_count (dict): the number of partners per person per layer
    '''

    # Initialize output
    partners = dict()

    # If means haven't been supplied, set to zero
    if means is None:
        means = {k: np.zeros(pop_size) for k in layer_keys}
    else:
         if len(means) != len(layer_keys):
             errormsg = f'The list of means has length {len(means)}; this must be the same length as layer_keys ({len(layer_keys)}).'
             raise ValueError(errormsg)

    # Now set the number of partners
    for lkey,n in zip(layer_keys, means):
        if sample:
            if dispersion is None:
                p_count = hpu.n_poisson(n, pop_size) + 1 # Draw the number of Poisson partners for this person. TEMP: add 1 to avoid zeros
            else:
                p_count = hpu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) + 1 # Or, from a negative binomial
        else:
            p_count = np.full(pop_size, n, dtype=hpd.default_int)

#        p_count = np.array((p_count/2.0).round(), dtype=hpd.default_int)
        partners[lkey] = p_count
        
    return partners


def validate_popdict(popdict, pars, verbose=True):
    '''
    Check that the popdict is the correct type, has the correct keys, and has
    the correct length
    '''

    # Check it's the right type
    try:
        popdict.keys() # Although not used directly, this is used in the error message below, and is a good proxy for a dict-like object
    except Exception as E:
        errormsg = f'The popdict should be a dictionary or hp.People object, but instead is {type(popdict)}'
        raise TypeError(errormsg) from E

    # Check keys and lengths
    required_keys = ['uid', 'age', 'sex', 'debut']
    popdict_keys = popdict.keys()
    pop_size = pars['pop_size']
    for key in required_keys:

        if key not in popdict_keys:
            errormsg = f'Could not find required key "{key}" in popdict; available keys are: {sc.strjoin(popdict.keys())}'
            sc.KeyNotFoundError(errormsg)

        actual_size = len(popdict[key])
        if actual_size != pop_size:
            errormsg = f'Could not use supplied popdict since key {key} has length {actual_size}, but all keys must have length {pop_size}'
            raise ValueError(errormsg)

        isnan = np.isnan(popdict[key]).sum()
        if isnan:
            errormsg = f'Population not fully created: {isnan:,} NaNs found in {key}.'
            raise ValueError(errormsg)

    return


def _tidy_edgelist(m, f, mapping=None):
    ''' Helper function to convert lists to arrays and optionally map arrays '''
    m = np.array(m, dtype=hpd.default_int)
    f = np.array(f, dtype=hpd.default_int)
    if mapping is not None:
        mapping = np.array(mapping, dtype=hpd.default_int)
        m = mapping[m]
        f = mapping[f]
    output = dict(m=m, f=f)
    return output


def make_random_contacts(p_count=None, sexes=None, n=None, durations=None, mapping=None):
    '''
    Make random contacts for a single layer as an edgelist. This will select sexually
    active male partners for sexually active females with no additional age structure.

    Args:
        p_count     (arr)   : the number of contacts to add for each person
        n_new       (int)   : number of agents to create contacts between (N)
        n           (int)   : the average number of contacts per person for this layer
        overshoot   (float) : to avoid needing to take multiple Poisson draws
        dispersion  (float) : if not None, use a negative binomial distribution with this dispersion parameter instead of Poisson to make the contacts
        mapping     (array) : optionally map the generated indices onto new indices

    Returns:
        Dictionary of two arrays defining UIDs of the edgelist (sources and targets)

    '''

    # Initialize
    f_inds = hpu.false(sexes)
    f = [] # Initialize the female partners
    m = [] # Initialize the male partners

    # Define indices; TODO fix or centralize this
    all_inds = np.arange(len(sexes)) # TODO get this a better way
    f_active_inds = np.intersect1d(mapping, f_inds)
    inactive_inds = np.setdiff1d(all_inds, mapping)

    # Precalculate contacts
    n_all_contacts  = int(sum(p_count[f_active_inds])) # Sum of partners for sexually active females
    weighting       = sexes*p_count # Males are more likely to be selected if they have higher concurrency; females will not be selected
    weighting[inactive_inds] = 0 # Exclude people not active
    weighting       = weighting/sum(weighting) # Turn this into a probability
    m_contacts      = hpu.choose_w(weighting, n_all_contacts, unique=False) # Select males

    # Make contacts
    count = 0
    for p in f_active_inds:
        n_contacts = p_count[p]
        these_contacts = m_contacts[count:count+n_contacts] # Assign people
        count += n_contacts
        f.extend([p]*n_contacts)
        m.extend(these_contacts)

    # Tidy up and add durations and start dates
    output = _tidy_edgelist(m, f)
    n_partnerships = len(output['m'])
    output['dur'] = hpu.sample(**durations, size=n_partnerships)
    output['start'] = np.zeros(n_partnerships) # For now, assume commence at beginning of sim
    output['end'] = output['start'] + output['dur']

    return output


# def create_partnerships(people, n_new=None, microstructure='random', **kwargs):
#     '''
#     Create partnerships for a People object 
#     '''

#     # Deal with debuts and participation rates
#     is_active = people.ages>people.debuts               # Whether or not people have ever been sexually active
#     active_inds = hpu.true(people.ages>people.debuts)   # Indices of sexually experienced people
#     n_active = sum(is_active)                           # Number of sexually experienced people

#     # n_new gives us the number of partnerships to create. Distribute these 
#     # using a weighting function that measures the difference between each
#     # person's preferred number of partners and their current number, for 
#     # all sexually active people
#     # Do this next bit by layer. Consider how to make it tractable with the find_contacts - store separately?
#     current_partners = hpu.find_contacts()
#     desired_partners = people.partners
#     difference = max(desired_partners - current_partners,0)
#     weightings = difference/sum(difference)
#     new_p1 = hpu.choose_w(weightings,n_new) # Indices of people to assign to new partnerships

#     # Precalculate contacts
#     n_all_contacts  = int(pop_size*n*overshoot) # The overshoot is used so we won't run out of contacts if the Poisson draws happen to be higher than the expected value
#     all_contacts    = hpu.choose_r(max_n=pop_size, n=n_all_contacts) # Choose people at random


#     if microstructure == 'random':
#         contacts = dict()
#         for lkey,n in people.partners.items():
#             n_active_layer = n_active*pars['layer_probs'][lkey]
#             active_inds_layer = hpu.binomial_filter(pars['layer_probs'][lkey], mapping)
#             durations = pars['dur_pship'][lkey]
#             contacts[lkey] = make_random_contacts(n_active_layer, n, durations, mapping=active_inds_layer, **kwargs)
#     else: # pragma: no cover
#         errormsg = f'Microstructure type "{microstructure}" not found; choices are random or TBC'
#         raise NotImplementedError(errormsg)
#     return contacts
 

