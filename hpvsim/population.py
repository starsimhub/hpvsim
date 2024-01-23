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
                sex_ratio=0.5, dt_round_age=True, microstructure=None,
                age_data=None, pop_trend=None, pop_age_trend=None, **kwargs):
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
    pop_age_trend = None

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
        total_pop = None

        # Handle data on population size by age & over time.
        # If a country location is supplied, this will be loaded in automatically from UN projections.
        # If a custom location is supplied, these data should be provided in datafiles read in by the pars.
        # Other demographic data like mortality and fertility are also available by
        # country, but these are loaded directly into the sim since they are not
        # stored as part of the people.
        location = sim['location']
        if sim['verbose']:
            print(f'Loading location-specific data for "{location}"')
        if use_age_data:
            try:
                age_data = hpdata.get_age_distribution(location, year=sim['start'], age_datafile=sim['age_datafile'])
                pop_trend = hpdata.get_total_pop(location, pop_datafile=sim['pop_datafile'])
                pop_age_trend = hpdata.get_age_distribution_over_time(location, popage_datafile=sim['popage_datafile'])
                total_pop = sum(age_data[:, 2])  # Return the total population
            except ValueError as E:
                warnmsg = f'Could not load age data for requested location "{location}" ({str(E)})'
                hpm.warn(warnmsg, die=True)

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

        uids, sexes, debuts, rel_sev, partners, cluster = set_static(n_agents, pars=sim.pars, sex_ratio=sim.pars['sex_ratio'])

        # Store output
        popdict = {}
        popdict['uid'] = uids
        popdict['age'] = ages
        popdict['sex'] = sexes
        popdict['debut'] = debuts
        popdict['rel_sev'] = rel_sev
        popdict['partners'] = partners
        popdict['cluster'] = cluster

        is_active = ages > debuts
        is_female = sexes == 0

        # Create the contacts
        lkeys = sim['acts'].keys() # TODO: consider a more robust way to do this
        if microstructure in ['random', 'default']:
            contacts = dict()
            current_partners = np.zeros((len(lkeys),n_agents))
            lno=0
            for lkey in lkeys:
                contacts[lkey], current_partners,_,_ = make_contacts(
                    lno=lno, tind=0, partners=partners[lno,:], current_partners=current_partners, ages=ages,
                    debuts=debuts, is_female=is_female, is_active=is_active, mixing=sim['mixing'][lkey],
                    layer_probs=sim['layer_probs'][lkey], f_cross_layer=sim['f_cross_layer'], m_cross_layer=sim['m_cross_layer'],
                    durations=sim['dur_pship'][lkey], acts=sim['acts'][lkey], age_act_pars=sim['age_act_pars'][lkey],
                    cluster=cluster, add_mixing=sim['add_mixing'], **kwargs
                )
                lno += 1

        else:
            errormsg = f'Microstructure type "{microstructure}" not found; choices are random or TBC'
            raise NotImplementedError(errormsg)

        popdict['contacts'] = contacts
        popdict['current_partners'] = current_partners

    else:
        ages = popdict['age']

    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    people = hpppl.People(sim.pars, pop_trend=pop_trend, pop_age_trend=pop_age_trend, **popdict) # List for storing the people

    sc.printv(f'Created {n_agents} agents, average age {ages.mean():0.2f} years', 2, verbose)

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
        p_count = hpu.sample(**ppars, size=n_agents)
        partners.append(p_count)


    return np.array(partners)


def set_static(new_n, existing_n=0, pars=None, sex_ratio=0.5):
    '''
    Set static population characteristics that do not change over time.
    Can be used when adding new births, in which case the existing popsize can be given.
    '''
    uid                 = np.arange(existing_n, existing_n+new_n, dtype=hpd.default_int)
    sex                 = np.random.binomial(1, sex_ratio, new_n)
    debut               = np.full(new_n, np.nan, dtype=hpd.default_float)
    debut[sex==1]       = hpu.sample(**pars['debut']['m'], size=sum(sex))
    debut[sex==0]       = hpu.sample(**pars['debut']['f'], size=new_n-sum(sex))
    n_layers            = len(pars['m_partners'].keys())
    partners            = np.full((n_layers, new_n), 0, dtype=hpd.default_int)
    partners[:, sex==1] = partner_count(n_agents=sum(sex), partner_pars=pars['m_partners'])
    partners[:, sex==0] = partner_count(n_agents=new_n-sum(sex), partner_pars=pars['f_partners'])
    cluster             = hpu.n_multinomial(pars['cluster_rel_sizes'], new_n).astype(int)
    rel_sev             = hpu.sample(**pars['sev_dist'], size=new_n) # Draw individual relative susceptibility factors

    return uid, sex, debut, rel_sev, partners, cluster


def validate_popdict(popdict, pars, verbose=True):
    '''
    Check that the popdict is the correct type, has the correct keys, and has
    the correct length
    '''

    # Check it's the right type
    try:
        popdict.keys() # Although not used directly, this is used in the error message below, and is a good proxy for a dict-like object
    except Exception as E:
        errormsg = f'The popdict should be a dictionary or hpv.People object, but instead is {type(popdict)}'
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


def _tidy_edgelist(f, m, mapping=None):
    ''' Helper function to convert lists to arrays and optionally map arrays '''
    if mapping is not None:
        mapping = np.array(mapping, dtype=hpd.default_int)
        m = mapping[m]
        f = mapping[f]
    output = dict(f=f, m=m)
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


def create_edgelist(tind, lno, partners, current_partners, mixing, age, is_active, is_female,
                        layer_probs, f_cross_layer, m_cross_layer, cluster, add_mixing):
    '''
    Create partnerships for a single layer. Eligible females and males are first identified if they are sexually active,
    underpartnered, and is available to partner for the specified type of relationship given their concurrency preference
    and whether they already have partners in other relationship layers. Concurrency preference is randomly assigned.
    Eligible individuals are randomly then chosen based on age- and sex-specific participation rates. The pairing
    algorithm iterates through each cluster and age bin of females, calculating relative preference of all participating
    males by multiplying the cluster- and age-specific mixing weights. Male partners are then randomly drawn based on
    these relative mixing weights. If there are not enough male partners to satiate all females in a cluster and age bin,
    females are randomly dropped with no new partner in this layer and time step.

    Args:
        lno                 (int): index of network layer
        partners            (int arr): array containing each agent's desired number of partners in this layer
        current_partners    (int arr): array containing each agent's actual current number of partners in this layer
        mixing              (float arr): age mixing matrix
        age                 (float arr): age
        is_active           (bool arr): whether or not people are sexually active
        is_female           (bool arr): whether each person is female
        layer_probs         (float arr): participation rates in this layer by age and sex
        cross_layer         (float): proportion of agents that have cross-layer relationships
        cluster             (int arr): array containing each agent's cluster id
        add_mixing          (float arr): additional mixing matrix
    '''

    # Useful variables
    n_agents        = len(age)
    n_layers        = current_partners.shape[0]
    f_active        =  is_female & is_active
    m_active        = ~is_female & is_active
    underpartnered  = current_partners[lno, :] < partners  # Indices of underpartnered people

    # Figure out how many new relationships to create by calculating the number of agents
    # who are underpartnered in this layer and either unpartnered in other layers or available
    # for cross-layer participation

    other_layers            = np.delete(np.arange(n_layers), lno)  # Indices of all other layers but this one
    other_partners          = current_partners[other_layers, :].any(axis=0)  # Whether or not people already partnered in other layers
    f_with_other_partners   = hpu.true(other_partners & is_female) # Indices of sexually active females with partners in other layers
    m_with_other_partners   = hpu.true(other_partners & ~is_female) # Indices of sexually active males with partners in other layers
    f_cross_inds            = hpu.binomial_filter(f_cross_layer, f_with_other_partners) # Indices who have cross-layer relationships
    m_cross_inds            = hpu.binomial_filter(m_cross_layer, m_with_other_partners) # Indices who have cross-layer relationships
    cross_layer_bools       = np.full(n_agents, False, dtype=bool) # Construct a boolean array indicating whether people have cross-layer relationships
    cross_layer_bools[f_cross_inds]  = True
    cross_layer_bools[m_cross_inds]  = True
    f_eligible              = f_active & underpartnered & (~other_partners | cross_layer_bools)
    m_eligible              = m_active & underpartnered & (~other_partners | cross_layer_bools)

    # Bin the females by age
    bins = layer_probs[0, :]  # Extract age bins
    cluster_range = np.unique(cluster)
    # Initialize
    f = np.array([], dtype=int)
    m = np.array([], dtype=int)
    # Filter males by participation rates
    m_eligible_inds = hpu.true(m_eligible)  # Inds of all eligible males across clusters
    m_participants = hpu.participation_filter(m_eligible_inds, age, layer_probs[2, :], bins=layer_probs[0, :])
    age_bins_m = np.digitize(age[m_participants], bins=bins) - 1  # Age bins of participating males

    # shuffle clusters
    np.random.shuffle(cluster_range)
    m_probs = np.ones(n_agents)  # Begin by assigning everyone equal probability of forming a new relationship

    for cl in cluster_range: # Loop through clusters for females
        # Try randomly select females for pairing
        f_eligible_inds = hpu.true(f_eligible * (cluster==cl))  # Inds of all eligible females in this cluster
        f_cl = hpu.participation_filter(f_eligible_inds, age, layer_probs[1, :], bins=layer_probs[0, :])
        if len(f_cl):
            # Draw male partners based on mixing matrices
            age_bins_f = np.digitize(age[f_cl], bins=bins) - 1  # Age bins of participating females in this cluster
            bin_range_f, males_needed = np.unique(age_bins_f, return_counts=True)  # For each female age bin, how many females need partners?
            # shuffle age bins
            bin_order = np.arange(len(bin_range_f))
            np.random.shuffle(bin_order)
            for ab, nm in zip(bin_range_f[bin_order], males_needed[bin_order]):  # Loop through the age bins of females and the number of males needed for each
                male_dist = mixing[:, ab + 1]  # Get female preferences for male partners in this age bin
                # Weight males according to the preferences of females of this age
                this_weighting = m_probs[m_participants] * male_dist[age_bins_m] * add_mixing[cl, cluster[m_participants]]
                if this_weighting.sum() > 0:
                    males_nonzero = hpu.true(this_weighting)  # Remove males with 0 weights
                    this_weighting_nonzero = this_weighting[males_nonzero]
                    f_inds = f_cl[age_bins_f == ab]  # inds of participating females in this age bin
                    if nm > len(this_weighting_nonzero):
                        #print(f'Warning, {nm} males desired but only {len(this_weighting_nonzero)} found.')
                        f_selected = f_inds[hpu.choose(nm, len(this_weighting_nonzero))] # randomly select females
                        nm = f_selected.size # number of new partnerships in this age bin
                    else:
                        f_selected = f_inds
                    m_selected = m_participants[males_nonzero[hpu.choose_w(this_weighting_nonzero, nm)]]  # Select males based on mixing weights
                    m_probs[m_selected] = 0 # remove males that get partnered
                    m = np.concatenate((m, m_selected)) # save selected males
                    f = np.concatenate((f, f_selected))
    # Count how many contacts there actually are
    new_pship_inds, new_pship_counts = np.unique(np.concatenate([f, m]), return_counts=True)
    if len(new_pship_inds) > 0:
        current_partners[lno, new_pship_inds] += new_pship_counts

    return f, m, current_partners, new_pship_inds, new_pship_counts



def make_contacts(lno=None, tind=None, partners=None, current_partners=None,
                  ages=None, debuts=None, is_female=None, is_active=None,
                  mixing=None, layer_probs=None, f_cross_layer=None, m_cross_layer=None,
                  durations=None, acts=None, age_act_pars=None,
                  cluster=None, add_mixing=None):
    '''
    Make contacts for a single layer as an edgelist. This will select sexually
    active male partners for sexually active females using age structure if given.
    '''

    # Create edgelist
    f,m,current_partners,new_pship_inds,new_pship_counts = create_edgelist(
        tind, lno, partners, current_partners, mixing, ages, is_active, is_female,
        layer_probs, f_cross_layer, m_cross_layer, cluster, add_mixing)

    # Convert edgelist into Contacts dict, with info about each partnership's duration,
    # coital frequency, etc
    output = {}

    if len(f) and len(m):
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
        output = _tidy_edgelist(f, m)
        n_partnerships = len(output['m'])
        output['age_f'] = ages[f]
        output['age_m'] = ages[m]
        output['dur'] = hpu.sample(**durations, size=n_partnerships)
        output['acts'] = scaled_acts
        output['start'] = np.array([tind] * n_partnerships, dtype=hpd.default_float)
        output['end'] = output['start'] + output['dur']
        output['cluster_f'] = cluster[f]
        output['cluster_m'] = cluster[m]

    return output, current_partners, new_pship_inds, new_pship_counts

