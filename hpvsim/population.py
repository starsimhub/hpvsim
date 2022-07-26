'''
Defines functions for making the population.
'''

#%% Imports
from re import U
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as hpu
from . import misc as hpm
from . import data as hpdata
from . import defaults as hpd
from . import parameters as hppar
from . import people as hpppl


# # Specify all externally visible functions this file defines
# __all__ = ['make_people', 'make_randpop', 'make_random_contacts']


def make_people(sim, popdict=None, reset=False, verbose=None, use_age_data=True,
                sex_ratio=0.5, dt_round_age=True, microstructure=None, **kwargs):
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
    total_pop = None # Optionally created but always returned

    if verbose is None:
        verbose = sim['verbose']
    dt = sim['dt'] # Timestep

    # If a people object or popdict is supplied, use it
    if sim.people and not reset:
        sim.people.initialize(sim_pars=sim.pars)
        return sim.people, total_pop # If it's already there, just return
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
                    age_data = hpdata.get_age_distribution(location, year=sim['start'])
                except ValueError as E:
                    warnmsg = f'Could not load age data for requested location "{location}" ({str(E)}), using default'
                    hpm.warn(warnmsg)

        total_pop = sum(age_data[:,2]) # Return the total population
        uids, sexes, debuts, partners = set_static(pop_size, pars=sim.pars, sex_ratio=sex_ratio)

        # Set ages, rounding to nearest timestep if requested
        age_data_min   = age_data[:,0]
        age_data_max   = age_data[:,1]
        age_data_range = age_data_max - age_data_min
        age_data_prob   = age_data[:,2]
        age_data_prob   /= age_data_prob.sum() # Ensure it sums to 1
        age_bins        = hpu.n_multinomial(age_data_prob, pop_size) # Choose age bins
        if dt_round_age:
            ages = age_data_min[age_bins] + np.random.randint(age_data_range[age_bins]/dt)*dt # Uniformly distribute within this age bin
        else:
            ages = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(pop_size) # Uniformly distribute within this age bin

        # Store output
        popdict = {}
        popdict['uid'] = uids
        popdict['age'] = ages
        popdict['sex'] = sexes
        popdict['debut'] = debuts
        popdict['partners'] = partners

        # Create the contacts
        active_inds = hpu.true(ages>debuts)     # Indices of sexually experienced people
        if microstructure in ['random', 'default']:
            contacts = dict()
            current_partners = []
            lno = 0
            for lkey,n in sim['partners'].items():
                # active_inds_layer = hpu.binomial_filter(sim['layer_probs'][lkey], active_inds)
                durations = sim['dur_pship'][lkey]
                acts = sim['acts'][lkey]
                contacts[lkey], cp = make_contacts(p_count=partners[lno], lkey=lkey, current_partners=np.array(current_partners), mixing=sim['mixing'][lkey], sexes=sexes, ages=ages, age_act_pars=sim['age_act_pars'][lkey], layer_probs=sim['layer_probs'][lkey], debuts=debuts, n=n, durations=durations, acts=acts, **kwargs)
                contacts[lkey]['acts'] += 1 # To avoid zeros
                current_partners.append(cp)
                lno += 1
        else: 
            errormsg = f'Microstructure type "{microstructure}" not found; choices are random or TBC'
            raise NotImplementedError(errormsg)

        popdict['contacts'] = contacts
        popdict['current_partners'] = np.array(current_partners)
        popdict['layer_keys'] = list(sim['partners'].keys())

    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    people = hpppl.People(sim.pars, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'], debut=popdict['debut'], partners=popdict['partners'], contacts=popdict['contacts'], current_partners=popdict['current_partners']) # List for storing the people

    sc.printv(f'Created {pop_size} people, average age {people.age.mean():0.2f} years', 2, verbose)

    return people, total_pop


def partner_count(pop_size=None, partner_pars=None):
    '''
    Assign each person a preferred number of concurrent partners for each layer
    Args:
        pop_size    (int)   : number of people
        layer_keys  (list)  : list of layers
        means       (dict)  : dictionary keyed by layer_keys with mean number of partners per layer
        sample      (bool)  : whether or not to sample the number of partners

    Returns:
        p_count (dict): the number of partners per person per layer
    '''

    # Initialize output
    partners = []

    # Set the number of partners
    for lkey,ppars in partner_pars.items():
        p_count = hpu.sample(**ppars, size=pop_size) + 1
        partners.append(p_count)
        
    return np.array(partners)


def set_static(new_n, existing_n=0, pars=None, sex_ratio=0.5):
    '''
    Set static population characteristics that do not change over time.
    Can be used when adding new births, in which case the existing popsize can be given.
    '''
    uid             = np.arange(existing_n, existing_n+new_n, dtype=hpd.default_int)
    sex             = np.random.binomial(1, sex_ratio, new_n)
    debut           = np.full(new_n, np.nan, dtype=hpd.default_float)
    debut[sex==1]   = hpu.sample(**pars['debut']['m'], size=sum(sex))
    debut[sex==0]   = hpu.sample(**pars['debut']['f'], size=new_n-sum(sex))
    partners        = partner_count(pop_size=new_n, partner_pars=pars['partners'])
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
    if mapping is not None:
        mapping = np.array(mapping, dtype=hpd.default_int)
        m = mapping[m]
        f = mapping[f]
    output = dict(m=m, f=f)
    return output


def age_scale_acts(acts=None, age_act_pars=None, age_f=None, age_m=None, debut_f=None, debut_m=None):
    ''' Scale the number of acts for each relationship according to the age of the partners '''

    # For each couple, get the average age they are now and the average age of debut
    avg_age     = np.array([age_f, age_m]).mean(axis=0)
    avg_debut   = np.array([debut_f, debut_m]).mean(axis=0)

    # Shorten parameter names
    dr = age_act_pars['debut_ratio']
    peak = age_act_pars['peak']
    rr = age_act_pars['retirement_ratio']
    retire = age_act_pars['retirement']

    # Get indices of people at different stages
    below_peak_inds = avg_age <=  age_act_pars['peak']
    above_peak_inds = (avg_age >  age_act_pars['peak']) & (avg_age <  age_act_pars['retirement'])
    retired_inds    = avg_age >  age_act_pars['retirement']

    # Set values by linearly scaling the number of acts for each partnership according to
    # the age of the couple at the commencement of the relationship
    below_peak_vals = acts[below_peak_inds]* (dr + (1-dr)/(peak - avg_debut[below_peak_inds]) * (avg_age[below_peak_inds] - avg_debut[below_peak_inds]))
    above_peak_vals = acts[above_peak_inds]* (rr + (1-rr)/(peak - retire)                     * (avg_age[above_peak_inds] - retire))
    retired_vals = 0

    # Set values and return
    scaled_acts = np.full(len(acts), np.nan, dtype=hpd.default_float)
    scaled_acts[below_peak_inds] = below_peak_vals
    scaled_acts[above_peak_inds] = above_peak_vals
    scaled_acts[retired_inds] = retired_vals

    return scaled_acts


def make_contacts(p_count=None, lkey=None, current_partners=None, mixing=None, sexes=None, ages=None, age_act_pars=None, layer_probs=None, debuts=None, n=None, durations=None, acts=None, mapping=None):
    '''
    Make contacts for a single layer as an edgelist. This will select sexually
    active male partners for sexually active females using age structure if given.

    Args:
        p_count     (arr)   : the number of contacts to add for each person
        n_new       (int)   : number of agents to create contacts between (N)
        n           (int)   : the average number of contacts per person for this layer
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
    m_inds          = hpu.true(sexes)
    all_inds        = np.arange(pop_size)

    # Find active males and females
    f_active_inds = hpu.true((sexes == 0) * (ages > debuts))
    m_active_inds = hpu.true((sexes == 1) * (ages > debuts))
    age_order = ages[f_active_inds].argsort() # Sort the females by age
    f_active_inds = f_active_inds[age_order]
    inactive_inds = np.setdiff1d(all_inds, hpu.true((ages > debuts)))
    weighting = sexes * p_count  # Males are more likely to be selected if they have higher concurrency; females will not be selected
    weighting[inactive_inds] = 0  # Exclude people not active

    if layer_probs is not None: # If layer probabilities have been specified, use them
        bins = layer_probs[0, :] # Use the age bins from the layer probabilities
        if len(current_partners) == 0: # If current partners haven't been se tup yet, then all sexually active females are eligible for selection
            f_eligible_inds = f_active_inds
        else: # Contacts are built up by layer; here we find people who have already been assigned partners in another layer and remove them
            f_eligible_inds = hpu.true((sexes == 0) * (ages > debuts) * (current_partners.sum(axis=0) == 0))  # People who've already got a partner in another layer are not eligible for selection
        age_bins_f = np.digitize(ages[f_eligible_inds], bins=bins) - 1 # Age bins of eligible females
        bin_range_f = np.unique(age_bins_f) # Range of bins
        f_contacts = [] # Initialize female contact list
        for ab in bin_range_f: # Loop over age bins
            these_f_contacts = hpu.binomial_filter(layer_probs[1][ab], f_eligible_inds[age_bins_f==ab]) # Select females according to their participation rate in this layer
            f_contacts += these_f_contacts.tolist()
        f_contacts = np.array(f_contacts)

    else:
        age_order = ages[f_active_inds].argsort()  # We sort the contacts by age so they get matched to partners of similar age
        f_contacts = f_active_inds[age_order]

    if mixing is not None:
        bins = mixing[:, 0]
        age_bins_f = np.digitize(ages[f_contacts], bins=bins)-1 # and this
        age_bins_m = np.digitize(ages[m_active_inds], bins=bins)-1 # and this
        bin_range_f = np.unique(age_bins_f) # For each female age bin, how many females need partners?
        m_contacts = [] # Initialize the male contact list
        for ab in bin_range_f: # Loop through the age bins of females and the number of males needed for each
            nm  = int(sum(p_count[f_contacts[age_bins_f==ab]])) # How many males will be needed?
            male_dist = mixing[:, ab+1] # Get the distribution of ages of the male partners of females of this age
            this_weighting = weighting[m_active_inds] * male_dist[age_bins_m] # Weight males according to the age preferences of females of this age
            selected_males = hpu.choose_w(this_weighting, nm, unique=False)  # Select males
            m_contacts += m_active_inds[selected_males].tolist() # Extract the indices of the selected males and add them to the contact list
            weighting[np.array(m_contacts)] -= 1 # Adjust weighting for males who've been selected
            weighting[weighting<0] = 0 # Don't let these be negative - this might happen if someone has already been assigned more partners than their preferred number

    else: # If no mixing has been specified, just do rough age assortivity
        n_all_contacts = int(sum(p_count[f_contacts]))  # Sum of partners for sexually active females
        m_contacts = hpu.choose_w(weighting, n_all_contacts, unique=False)  # Select males
        m_age_order = ages[m_contacts].argsort()  # Sort the partners by age as well
        m_contacts = m_contacts[m_age_order]

    # Make contacts
    count = 0
    for p in f_contacts:
        n_contacts = p_count[p]
        these_contacts = m_contacts[count:count+n_contacts] # Assign people
        count += n_contacts
        f.extend([p]*n_contacts)
        m.extend(these_contacts)
    m = np.array(m, dtype=hpd.default_int)
    f = np.array(f, dtype=hpd.default_int)

    # Count how many contacts there actually are: will be different for males, should be the same for females
    unique, count = np.unique(np.concatenate([m, f]),return_counts=True)
    actual_p_count = np.full(pop_size, 0, dtype=hpd.default_int)
    actual_p_count[unique] = count

    # Scale number of acts by age of couple
    acts = hpu.sample(**acts, size=len(f))
    scaled_acts = age_scale_acts(acts=acts, age_act_pars=age_act_pars, age_f=ages[f], age_m=ages[m], debut_f=debuts[f], debut_m=debuts[m])
    keep_inds = scaled_acts>0 # Discard partnerships with zero acts (e.g. because they are "post-retirement")
    m = m[keep_inds]
    f = f[keep_inds]
    scaled_acts = scaled_acts[keep_inds]

    # Tidy up and add durations and start dates
    output = _tidy_edgelist(m, f)
    n_partnerships = len(output['m'])
    output['age_f'] = ages[f]
    output['age_m'] = ages[m]
    output['dur'] = hpu.sample(**durations, size=n_partnerships)
    output['acts'] = scaled_acts
    output['start'] = np.zeros(n_partnerships) # For now, assume commence at beginning of sim
    output['end'] = output['start'] + output['dur']
    output['layer'] = lkey

    return output, actual_p_count

