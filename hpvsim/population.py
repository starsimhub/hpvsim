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
                sex_ratio=0.5, dt_round_age=True, dispersion=None, microstructure=None, **kwargs):
    '''
    Make the people for the simulation.

    Usually called via ``sim.initialize()``.

    Args:
        sim      (Sim)  : the simulation object; population parameters are taken from the sim object
        popdict  (any)  : if supplied, use this population dictionary instead of generating a new one; can be a dict or People object
        reset    (bool) : whether to force population creation even if self.popdict/self.people exists
        verbose  (bool) : level of detail to print
        use_age_data (bool):
        sex_ratio (bool):
        dt_round_age (bool): whether to round people's ages to the nearest timestep (default true)

    Returns:
        people (People): people
    '''

    # Set inputs and defaults
    pop_size = int(sim['pop_size']) # Shorten
    if verbose is None:
        verbose = sim['verbose']
    dt = sim['dt'] # Timestep

    # If a people object or popdict is supplied, use it
    if sim.people and not reset:
        sim.people.initialize(sim_pars=sim.pars)
        return sim.people # If it's already there, just return
    elif sim.popdict and popdict is None:
        popdict = sim.popdict # Use stored one
        sim.popdict = None # Once loaded, remove

    if popdict is None:

        pop_size = int(sim['pop_size']) # Number of people

        # Load age data by country if available, or use defaults.
        # Other demographic data like mortality and fertility are also available by
        # country, but these are loaded directly into the sim since they are not 
        # stored as part of the people.
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

        uids, sexes, debuts, partners = set_static(pop_size, pars=sim.pars, sex_ratio=sex_ratio, dispersion=dispersion)

        # Set ages, rounding to nearest timestep if requested
        age_data_min   = age_data[:,0]
        age_data_max   = age_data[:,1] + 1 # Since actually e.g. 69.999
        age_data_range = age_data_max - age_data_min
        age_data_prob   = age_data[:,2]
        age_data_prob   /= age_data_prob.sum() # Ensure it sums to 1
        age_bins        = hpu.n_multinomial(age_data_prob, pop_size) # Choose age bins
        if dt_round_age:
            ages = age_data_min[age_bins] + np.random.randint(age_data_range[age_bins]/dt)*dt # Uniformly distribute within this age bin
        else:
            ages            = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(pop_size) # Uniformly distribute within this age bin

        # Store output
        popdict = {}
        popdict['uid'] = uids
        popdict['age'] = ages
        popdict['sex'] = sexes
        popdict['debut'] = debuts
        popdict['partners'] = partners

        # Create the contacts
        active_inds = hpu.true(ages>debuts)     # Indices of sexually experienced people
        if microstructure in ['random', 'basic']:
            contacts = dict()
            current_partners = []
            lno = 0
            for lkey,n in sim['partners'].items():
                active_inds_layer = hpu.binomial_filter(sim['layer_probs'][lkey], active_inds)
                durations = sim['dur_pship'][lkey]
                acts = sim['acts'][lkey]
                contacts[lkey], cp = make_random_contacts(p_count=partners[lno], sexes=sexes, n=n, durations=durations, acts=acts, mapping=active_inds_layer, **kwargs)
                current_partners.append(cp)
                lno += 1
        else: 
            errormsg = f'Microstructure type "{microstructure}" not found; choices are random or TBC'
            raise NotImplementedError(errormsg)

        popdict['contacts']   = contacts
        popdict['current_partners']   = np.array(current_partners)
        popdict['layer_keys'] = list(sim['partners'].keys())

    # Ensure prognoses are set
    if sim['prognoses'] is None:
        sim['prognoses'] = hppar.get_prognoses()

    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    people = hpppl.People(sim.pars, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'], debut=popdict['debut'], partners=popdict['partners'], contacts=popdict['contacts'], current_partners=popdict['current_partners']) # List for storing the people

    sc.printv(f'Created {pop_size} people, average age {people.age.mean():0.2f} years', 2, verbose)

    return people


def partner_count(pop_size=None, layer_keys=None, means=None, sample=True, dispersion=None):
    '''
    Assign each person a preferred number of concurrent partners for each layer
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
    partners = []

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

        partners.append(p_count)
        
    return np.array(partners)


def set_static(new_n, existing_n=0, pars=None, sex_ratio=0.5, dispersion=None):
    '''
    Set static population characteristics that do not change over time.
    Can be used when adding new births, in which case the existing popsize can be given.
    '''
    uid             = np.arange(existing_n, existing_n+new_n, dtype=hpd.default_int)
    sex             = np.random.binomial(1, sex_ratio, new_n)
    debut           = np.full(new_n, np.nan, dtype=hpd.default_float)
    debut[sex==1]   = hpu.sample(**pars['debut']['m'], size=sum(sex))
    debut[sex==0]   = hpu.sample(**pars['debut']['f'], size=new_n-sum(sex))
    partners        = partner_count(pop_size=new_n, layer_keys=pars['partners'].keys(), means=pars['partners'].values(), dispersion=dispersion)
    return uid, sex, debut, partners


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


def make_random_contacts(p_count=None, sexes=None, n=None, durations=None, acts=None, mapping=None):
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
    f = [] # Initialize the female partners
    m = [] # Initialize the male partners

    # Define indices; TODO fix or centralize this
    pop_size        = len(sexes)
    f_inds          = hpu.false(sexes)
    all_inds        = np.arange(pop_size) 
    f_active_inds   = np.intersect1d(mapping, f_inds)
    inactive_inds   = np.setdiff1d(all_inds, mapping)

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

    # Count how many contacts there actually are: will be different for males, should be the same for females
    unique, count = np.unique(np.concatenate([np.array(m),np.array(f)]),return_counts=True)
    actual_p_count = np.full(pop_size, 0, dtype=hpd.default_int)
    actual_p_count[unique] = count

    # Tidy up and add durations and start dates
    output = _tidy_edgelist(m, f)
    n_partnerships = len(output['m'])
    output['dur'] = hpu.sample(**durations, size=n_partnerships)
    output['acts'] = hpu.sample(**acts, size=n_partnerships)
    output['start'] = np.zeros(n_partnerships) # For now, assume commence at beginning of sim
    output['end'] = output['start'] + output['dur']

    # TODO: count the number of acts for each person

    return output, actual_p_count


# %%
