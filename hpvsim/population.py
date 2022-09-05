'''
Defines functions for making the population.
'''

#%% Imports
import numpy as np
import sciris as sc
from . import utils as hpu
from . import misc as hpm
from . import data as hpdata
from . import defaults as hpd
from . import people as hpppl


# Specify all externally visible functions this file defines
__all__ = ['make_people', 'make_contacts']


def make_people(sim, popdict=None, reset=False, verbose=None, use_age_data=True,
                sex_ratio=0.5, dt_round_age=True, microstructure=None, **kwargs):
    '''
    Make the people for the simulation.

    Usually called via :py:func:`hpvsim.sim.Sim.initialize`.

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
    n_agents = int(sim['n_agents']) # Shorten
    total_pop = None # Optionally created but always returned
    pop_trend = None # Populated later if location is specified

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

        n_agents = int(sim['n_agents']) # Number of people

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
                    age_data  = hpdata.get_age_distribution(location, year=sim['start'])
                    pop_trend = hpdata.get_total_pop(location)
                except ValueError as E:
                    warnmsg = f'Could not load age data for requested location "{location}" ({str(E)}), using default'
                    hpm.warn(warnmsg)

        total_pop = sum(age_data[:,2]) # Return the total population
        uids, sexes, debuts, partners = set_static(n_agents, pars=sim.pars, sex_ratio=sex_ratio)

        # Set ages, rounding to nearest timestep if requested
        age_data_min   = age_data[:,0]
        age_data_max   = age_data[:,1]
        age_data_range = age_data_max - age_data_min
        age_data_prob   = age_data[:,2]
        age_data_prob   /= age_data_prob.sum() # Ensure it sums to 1
        age_bins        = hpu.n_multinomial(age_data_prob, n_agents) # Choose age bins
        if dt_round_age:
            ages = age_data_min[age_bins] + np.random.randint(age_data_range[age_bins]/dt)*dt # Uniformly distribute within this age bin
        else:
            ages = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(n_agents) # Uniformly distribute within this age bin

        # Store output
        popdict = {}
        popdict['uid'] = uids
        popdict['age'] = ages
        popdict['sex'] = sexes
        popdict['debut'] = debuts
        popdict['partners'] = partners

        is_active = ages > debuts
        is_female = sexes == 0

        # Create the contacts
        lkeys = sim['partners'].keys() # TODO: consider a more robust way to do this
        if microstructure in ['random', 'default']:
            contacts = dict()
            current_partners = np.zeros((len(lkeys),n_agents))
            lno = 0
            for lkey,n in sim['partners'].items():
                durations = sim['dur_pship'][lkey]
                acts = sim['acts'][lkey]
                contacts[lkey], current_partners = make_contacts(
                    lno=lno, t=0, partners=partners[lno,:], current_partners=current_partners,
                    sexes=sexes, ages=ages, debuts=debuts, is_female=is_female, is_active=is_active,
                    mixing=sim['mixing'][lkey], layer_probs=sim['layer_probs'][lkey], cross_layer=sim['cross_layer'],
                    pref_weight=100, durations=durations, acts=acts, age_act_pars=sim['age_act_pars'][lkey], **kwargs
                )
                contacts[lkey]['acts'] += 1 # To avoid zeros
                lno += 1
        else:
            errormsg = f'Microstructure type "{microstructure}" not found; choices are random or TBC'
            raise NotImplementedError(errormsg)

        popdict['contacts'] = contacts
        popdict['current_partners'] = current_partners

    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    people = hpppl.People(sim.pars, pop_trend=pop_trend, **popdict) # List for storing the people

    sc.printv(f'Created {n_agents} agents, average age {people.age.mean():0.2f} years', 2, verbose)

    return people, total_pop


def partner_count(n_agents=None, partner_pars=None):
    '''
    Assign each person a preferred number of concurrent partners for each layer

    Args:
        n_agents    (int)   : number of agents
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
        p_count = hpu.sample(**ppars, size=n_agents) + 1
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
    partners        = partner_count(n_agents=new_n, partner_pars=pars['partners'])
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
    n_agents = pars['n_agents']
    for key in required_keys:

        if key not in popdict_keys:
            errormsg = f'Could not find required key "{key}" in popdict; available keys are: {sc.strjoin(popdict.keys())}'
            sc.KeyNotFoundError(errormsg)

        actual_size = len(popdict[key])
        if actual_size != n_agents:
            errormsg = f'Could not use supplied popdict since key {key} has length {actual_size}, but all keys must have length {n_agents}'
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


def make_contacts(lno=None, t=None, partners=None, current_partners=None,
                  sexes=None, ages=None, debuts=None, is_female=None, is_active=None,
                  mixing=None, layer_probs=None, cross_layer=None,
                  pref_weight=None, durations=None, acts=None, age_act_pars=None):
    '''
    Make contacts for a single layer as an edgelist. This will select sexually
    active male partners for sexually active females using age structure if given.

    Args:
        partners    (arr)   : the number of preferred partners for each person
        n_new       (int)   : number of agents to create contacts between (N)
        mapping     (array) : optionally map the generated indices onto new indices

    Returns:
        Dictionary of two arrays defining UIDs of the edgelist (sources and targets)

    '''

    f,m,current_partners = hpu.create_partnerships(
        lno, partners, current_partners, mixing, sexes, ages, is_active, is_female,
        layer_probs, pref_weight, cross_layer)

    # Scale number of acts by age of couple
    acts = hpu.sample(**acts, size=len(f))
    kwargs = dict(acts=acts,
                  age_act_pars=age_act_pars,
                  age_f=ages[f],
                  age_m=ages[m],
                  debut_f=debuts[f],
                  debut_m=debuts[m]
                  )

    scaled_acts = age_scale_acts(**kwargs)
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
    output['start'] = np.array([t] * n_partnerships, dtype=hpd.default_float)
    try:
        output['end'] = output['start'] + output['dur']
    except:
        import traceback;
        traceback.print_exc();
        import pdb;
        pdb.set_trace()

    return output, current_partners

