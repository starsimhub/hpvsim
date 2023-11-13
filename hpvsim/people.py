'''
Defines the People class and functions associated with making people and handling
the transitions between states (e.g., from susceptible to infected).
'''

#%% Imports
import numpy as np
import sciris as sc
from . import utils as hpu
from . import defaults as hpd
from . import base as hpb
from . import population as hppop
from . import plotting as hpplt
from . import parameters as hppar
from . import immunity as hpimm


__all__ = ['People']

class People(hpb.BasePeople):
    '''
    A class to perform all the operations on the people -- usually not invoked directly.

    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized. However, ages, contacts, etc. will need to be created separately --
    see ``hpv.make_people()`` instead.

    Note that this class handles the mechanics of updating the actual people, while
    ``hpv.BasePeople`` takes care of housekeeping (saving, loading, exporting, etc.).
    Please see the BasePeople class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as n_agents
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        pop_trend (dataframe): a dataframe of years and population sizes, if available
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::

        ppl1 = hpv.People(2000)

        sim = hpv.Sim()
        ppl2 = hpv.People(sim.pars)
    '''
    
    #%% Basic methods

    def __init__(self, pars, strict=True, pop_trend=None, pop_age_trend=None, **kwargs):

        # Initialize the BasePeople, which also sets things up for filtering
        super().__init__(pars)
        
        # Handle pars and settings

        # Other initialization
        self.pop_trend = pop_trend
        self.pop_age_trend = pop_age_trend
        self.init_contacts() # Initialize the contacts
        self.ng = self.pars['n_genotypes']
        self.na = len(self.pars['age_bin_edges'])-1

        self.lag_bins = np.linspace(0,50,51)
        self.rship_lags = dict()
        for lkey in self.layer_keys():
            self.rship_lags[lkey] = np.zeros(len(self.lag_bins)-1, dtype=hpd.default_float)

        # Store age bins
        self.age_bin_edges = self.pars['age_bin_edges'] # Age bins for age results

        if strict:
            self.lock() # If strict is true, stop further keys from being set (does not affect attributes)

        # Store flows to be computed during simulation
        self.init_flows()

        # Although we have called init(), we still need to call initialize()
        self.initialized = False
        
        # Store kwargs here for now, to be dealt with during initialize()
        self.kwargs = kwargs

        return


    def init_flows(self):
        ''' Initialize flows to be zero '''
        df = hpd.default_float
        self.flows              = {key: 0 for key in hpd.flow_keys}
        self.genotype_flows     = {key: np.zeros(self.ng, dtype=df) for key in hpd.genotype_flow_keys}
        self.age_flows          = {key: np.zeros(self.na, dtype=df) for key in hpd.flow_keys}
        self.sex_flows          = {f'{key}'         : np.zeros(2, dtype=df) for key in hpd.by_sex_keys}
        self.demographic_flows  = {f'{key}'         : 0 for key in hpd.dem_keys}
        return
    
    
    def scale_flows(self, inds):
        '''
        Return the scaled versions of the flows -- replacement for len(inds) 
        followed by scale factor multiplication
        '''
        return self.scale[inds].sum()


    def increment_age(self):
        ''' Let people age by one timestep '''
        self.age[self.alive] += self.dt
        return


    def initialize(self, sim_pars=None):
        ''' Perform initializations '''
        super().initialize() # Initialize states

        # Handle partners and contacts
        kwargs = self.kwargs
        if 'partners' in kwargs:
            self.partners[:] = kwargs.pop('partners') # Store the desired concurrency
        if 'current_partners' in kwargs:
            self.current_partners[:] = kwargs.pop('current_partners') # Store current actual number - updated each step though
            for ln,lkey in enumerate(self.layer_keys()):
                self.rship_start_dates[ln,self.current_partners[ln]>0] = 0
        if 'contacts' in kwargs:
            self.add_contacts(kwargs.pop('contacts')) # Also updated each step

        self.n_rships[:] = self.current_partners
        self.ever_partnered[:] = self.current_partners.sum(axis=0)>0

        # Handle all other values, e.g. age
        for key,value in kwargs.items():
            if self._lock:
                self.set(key, value)
            elif key in self._data:
                self[key][:] = value
            else:
                self[key] = value
        
        # Set the scale factor
        self.scale[:] = sim_pars['pop_scale']

        # Additional validation
        self.validate(sim_pars=sim_pars) # First, check that essential-to-match parameters match
        self.initialized = True

        return


    def update_states_pre(self, t, year=None):
        ''' Perform all state updates at the current timestep '''

        # Initialize
        self.t = t
        self.dt = self.pars['dt']
        self.init_flows()

        # Let people age by one time step
        self.increment_age()

        # Perform updates that are not genotype-specific
        update_freq = max(1, int(self.pars['dt_demog'] / self.pars['dt'])) # Ensure it's an integer not smaller than 1
        if t % update_freq == 0:

            # Apply death rates from other causes
            other_deaths, deaths_female, deaths_male    = self.apply_death_rates(year=year)
            self.demographic_flows['other_deaths']      = other_deaths
            self.sex_flows['other_deaths_by_sex'][0]    = deaths_female
            self.sex_flows['other_deaths_by_sex'][1]    = deaths_male

            # Add births
            new_births = self.add_births(year=year, sex_ratio=self.pars['sex_ratio'])
            self.demographic_flows['births'] = new_births

            # Check migration
            migration = self.check_migration(year=year)
            self.demographic_flows['migration'] = migration

        # Perform updates that are genotype-specific
        for g in range(self.pars['n_genotypes']):
            self.check_clearance(g) # check for clearance (need to do this first)

        # Perform updates that are not genotype specific
        deaths_by_age, deaths = self.check_cancer_deaths()
        self.flows['cancer_deaths'] = deaths
        self.age_flows['cancer_deaths'] = deaths_by_age

        # Before applying interventions or new infections, calculate the pool of susceptibles
        self.sus_pool = self.susceptible.all(axis=0) # True for people with no infection at the start of the timestep

        return

    def update_states_post(self, t, year=None):
        ''' State updates at the end of the current timestep '''
        ng = self.pars['n_genotypes']
        for g in range(ng):
            for key in ['cins','cancers']:  # update flows
                cases_by_age, cases = self.check_progress(key, g)
                self.flows[key] += cases  # Increment flows (summed over all genotypes)
                self.genotype_flows[key][g] = cases # Store flows by genotype
                self.age_flows[key] += cases_by_age # Increment flows by age (summed over all genotypes)


    #%% Disease progression methods
    def set_prognoses(self, inds, g, gpars, dt):
        '''
        Assigns prognoses for all infected women on day of infection.
        '''

        # Set length of infection, which is moderated by any prior cell-level immunity
        sev_imm = self.sev_imm[g, inds]

        # Determine how long before precancerous cell changes
        dur_precin = hpu.sample(**gpars['dur_precin'], size=len(inds))*(1-sev_imm) # Sample from distribution
        self.dur_precin[g, inds] = dur_precin
        self.dur_infection[g, inds] = dur_precin

        # Probability of progressing
        cin_probs = hppar.compute_severity(dur_precin, rel_sev=self.rel_sev[inds], pars=gpars['cin_fn'])
        cin_bools = hpu.binomial_arr(cin_probs)
        cin_inds = inds[cin_bools]
        nocin_inds = inds[~cin_bools]

        # Duration of CIN
        age_mod = np.ones(len(cin_inds))
        age_mod[self.age[cin_inds] >= self.pars['age_risk']['age']] = self.pars['age_risk']['risk']
        dur_cin = hpu.sample(**gpars['dur_cin'], size=len(cin_inds))* age_mod
        self.dur_cin[g, cin_inds] = dur_cin
        self.dur_infection[g, cin_inds] += dur_cin

        # Set date of clearance for those who don't develop precancer
        self.date_clearance[g, nocin_inds] = self.t + sc.randround(self.dur_precin[g, nocin_inds]/dt)

        # Set date of onset of precancer for those who develop precancer
        self.date_cin[g, cin_inds] = self.t + sc.randround(self.dur_precin[g, cin_inds]/dt)

        # Set infection severity and outcomes
        self.set_severity(inds[cin_bools], g, gpars, dt)

        return


    def set_severity(self, inds, g, gpars, dt, set_sev=True):
        '''
        Set severity levels for individual women
        Args:
            inds: indices of women to set severity for
            g: genotype index
            dt: timestep
            set_sev: whether or not to set initial severity
        '''

        # Calculate the probability of cancer for each woman
        dur_cin = self.dur_cin[g, inds]

        if gpars["cancer_fn"].get('method') == 'cin_integral':
            cancer_pars = sc.mergedicts(gpars["cancer_fn"], gpars["cin_fn"])
        else:
            cancer_pars = gpars["cancer_fn"]
        cancer_prob = hppar.compute_severity(dur_cin, rel_sev=self.rel_sev[inds], pars=cancer_pars)  # Calculate probability of cancer
        n_extra = self.pars['ms_agent_ratio']
        cancer_scale = self.pars['pop_scale'] / n_extra

        if n_extra == 1:
            cancer_prob_arr = hppar.compute_severity(dur_cin, rel_sev=self.rel_sev[inds], pars=cancer_pars)
        # Multiscale version
        elif n_extra > 1:

            # Firstly, determine who will transform based on severity values, and scale them to create more agents
            is_cancer = hpu.binomial_arr(cancer_prob) # Select who transforms - NB, this array gets extended later
            cancer_inds = inds[is_cancer] # Indices of those who transform
            self.scale[cancer_inds] = cancer_scale  # Shrink the weight of the original agents, but otherwise leave them the same

            # Create extra disease severity values for the extra agents
            full_size = (len(inds), n_extra)  # Main axis is indices, but include columns for multiscale agents
            extra_dur_cin = hpu.sample(**gpars['dur_cin'], size=full_size)
            extra_dur_precin = hpu.sample(**gpars['dur_precin'], size=full_size)
            extra_rel_sevs = np.ones(full_size)*self.rel_sev[inds][:,None]

            extra_cin_probs = hppar.compute_severity(extra_dur_precin, rel_sev=extra_rel_sevs, pars=gpars['cin_fn'])
            extra_cin_bools = hpu.binomial_arr(extra_cin_probs[:,1:])

            extra_cancer_probs = hppar.compute_severity(extra_dur_cin, rel_sev=extra_rel_sevs, pars=cancer_pars)  # Calculate probability of cancer
            extra_cancer_probs[:,1:][~extra_cin_bools] = 0
            # Based on the extra severity values, determine additional transformation probabilities
            extra_cancer_bools = hpu.binomial_arr(extra_cancer_probs[:,1:])
            extra_cancer_bools *= self.level0[inds, None]  # Don't allow existing cancer agents to make more cancer agents
            extra_cancer_counts = extra_cancer_bools.sum(axis=1)  # Find out how many new cancer cases we have
            n_new_agents = extra_cancer_counts.sum()  # Total number of new agents
            if n_new_agents:  # If we have more than 0, proceed
                extra_source_lists = []
                for i, count in enumerate(extra_cancer_counts):
                    ii = inds[i]
                    if count:  # At least 1 new cancer agent, plus person is not already a cancer agent
                        extra_source_lists.append([ii] * int(count))  # Duplicate the current index count times
                extra_source_inds = np.concatenate(extra_source_lists).flatten()  # Assemble the sources for these new agents
                n_new_agents = len(extra_source_inds)  # The same as above, *unless* a cancer agent tried to spawn more cancer agents

                # Create the new agents and assign them the same properties as the existing agents
                new_inds = self._grow(n_new_agents)
                for state in self.meta.states_to_set:
                    if state.ndim == 1:
                        self[state.name][new_inds] = self[state.name][extra_source_inds]
                    elif state.ndim == 2:
                        self[state.name][:, new_inds] = self[state.name][:, extra_source_inds]

                # Reset the states for the new agents
                self.level0[new_inds] = False
                self.level1[new_inds] = True
                self.scale[new_inds] = cancer_scale

                # Add the new indices onto the existing vectors
                inds = np.append(inds, new_inds)
                is_cancer = np.append(is_cancer, np.full(len(new_inds), fill_value=True))
                new_dur_precin = extra_dur_precin[:, 1:][extra_cancer_bools]
                new_dur_cin = extra_dur_cin[:, 1:][extra_cancer_bools]
                new_dur_infection = new_dur_precin + new_dur_cin
                self.dur_precin[g, new_inds] = new_dur_precin
                self.dur_cin[g, new_inds] = new_dur_cin
                self.dur_infection[g, new_inds] = new_dur_infection
                self.date_infectious[g, new_inds] = self.t
                self.date_cin[g, new_inds] = self.t + sc.randround(new_dur_precin / dt)
                dur_cin = np.append(dur_cin, new_dur_cin)

            # Finally, create an array for storing the transformation probabilities.
            # We've already figured out who's going to transform, so we fill the array with 1s for those who do.
            cancer_prob_arr = np.zeros(len(inds))
            cancer_prob_arr[is_cancer] = 1  # Make sure inds that got assigned cancer above dont get stochastically missed

        # Determine who goes to cancer
        is_cancer = hpu.binomial_arr(cancer_prob_arr)
        cancer_inds = inds[is_cancer]
        no_cancer_inds = inds[~is_cancer]  # Indices of those who eventually heal lesion/clear infection

        # Set date of clearance for those who don't go to cancer
        dur_inf = self.dur_infection[g, inds]
        time_to_clear = dur_inf[~is_cancer]
        self.date_clearance[g, no_cancer_inds] = np.fmax(self.date_clearance[g, no_cancer_inds],
                                                         self.date_exposed[g, no_cancer_inds] +
                                                         sc.randround(time_to_clear / dt))

        # Set dates for those who go to cancer.
        # Set date of onset of precancer and eventual severity outcomes for those who develop precancer
        dur_cin_transformed = dur_cin[is_cancer] # Duration of episomal infection for those who transform
        self.date_cancerous[g, cancer_inds] = self.date_cin[g, cancer_inds] + sc.randround(dur_cin_transformed/dt)
        self.dur_infection[g, cancer_inds] += self.dur_cin[g, cancer_inds]
        dur_cancer = hpu.sample(**self.pars['dur_cancer'], size=len(cancer_inds))
        self.date_dead_cancer[cancer_inds] = self.date_cancerous[g, cancer_inds] + sc.randround(dur_cancer / dt)
        self.dur_cancer[g, cancer_inds] = dur_cancer

        return


    #%% Methods for updating partnerships
    def dissolve_partnerships(self, t=None):
        ''' Dissolve partnerships '''

        n_dissolved = dict()

        for lno,lkey in enumerate(self.layer_keys()):
            layer = self.contacts[lkey]

            to_dissolve = (~self['alive'][layer['m']]) + (~self['alive'][layer['f']]) + ( (self.t*self.pars['dt']) > layer['end']).astype(bool)
            dissolved = layer.pop_inds(to_dissolve) # Remove them from the contacts list

            # Update current number of partners
            unique, counts = hpu.unique(np.concatenate([dissolved['f'],dissolved['m']]))
            self.current_partners[lno,unique] -= counts
            self.rship_end_dates[lno, unique] = self.t
            n_dissolved[lkey] = len(dissolved['f'])

        return n_dissolved # Return the number of dissolved partnerships by layer


    def create_partnerships(self, tind, mixing, layer_probs, f_cross_layer, m_cross_layer, dur_pship, acts, age_act_pars):
        '''
        Create partnerships. All the hard work of creating the contacts is done by hppop.make_contacts,
        which in turn relies on hpu.create_edgelist for creating the edgelist. This method is just a light wrapper
        that passes in the arguments in the right format and the updates relationship info stored in the People class.
        '''
        # Initialize
        new_pships = dict()

        # Loop over layers
        lno=0
        for lkey in self.layer_keys():
            pship_args = dict(
                lno=lno, tind=tind, partners=self.partners[lno], current_partners=self.current_partners, ages=self.age,
                debuts=self.debut, is_female=self.is_female, is_active=self.is_active, mixing=mixing[lkey],
                layer_probs=layer_probs[lkey], f_cross_layer=f_cross_layer, m_cross_layer=m_cross_layer,
                durations=dur_pship[lkey], acts=acts[lkey], age_act_pars=age_act_pars[lkey],
                cluster=self.cluster, add_mixing=self.pars['add_mixing']
            )
            new_pships[lkey], current_partners, new_pship_inds, new_pship_counts = hppop.make_contacts(**pship_args)

            # Update relationship info
            if len(new_pship_inds)>0:
                self.ever_partnered[new_pship_inds] = True
            self.current_partners[:] = current_partners
            if len(new_pship_inds):
                self.rship_start_dates[lno, new_pship_inds] = self.t
                self.n_rships[lno, new_pship_inds] += new_pship_counts
                lags = self.rship_start_dates[lno, new_pship_inds] - self.rship_end_dates[lno, new_pship_inds]
                self.rship_lags[lkey] += np.histogram(lags, self.lag_bins)[0]

            lno += 1

        self.add_contacts(new_pships)

        return



    #%% Methods for updating state
    def check_inds(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is false and which meet the date criterion '''
        if filter_inds is None:
            not_current = hpu.false(current)
        else:
            not_current = hpu.ifalsei(current, filter_inds)
        has_date = hpu.idefinedi(date, not_current)
        inds     = hpu.itrue(self.t >= date[has_date], has_date)
        return inds


    def check_inds_true(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is true and which meet the date criterion '''
        if filter_inds is None:
            current_inds = hpu.true(current)
        else:
            current_inds = hpu.itruei(current, filter_inds)
        has_date = hpu.idefinedi(date, current_inds)
        inds     = hpu.itrue(self.t >= date[has_date], has_date)
        return inds


    def check_progress(self, what, genotype):
        ''' Wrapper function for all the new progression checks '''
        if what=='cins':       cases_by_age, cases = self.check_cin(genotype)
        elif what=='cancers':   cases_by_age, cases = self.check_cancer(genotype)
        return cases_by_age, cases


    def check_cin(self, genotype):
        ''' Check for new progressions to CIN '''
        # Only include infectious females who haven't already cleared CIN or progressed to cancer
        filters = self.infectious[genotype,:]*self.is_female_alive*~(self.date_clearance[genotype,:]<=self.t)
        filter_inds = filters.nonzero()[0]
        inds = self.check_inds(self.cin[genotype,:], self.date_cin[genotype,:], filter_inds=filter_inds)
        self.cin[genotype, inds] = True

        # Age calculations
        cases_by_age = np.histogram(self.age[inds], bins=self.age_bin_edges, weights=self.scale[inds])[0]
        return cases_by_age, self.scale_flows(inds)

    def check_cancer(self, genotype):
        ''' Check for new progressions to cancer '''
        not_current = hpu.ifalsei(self.cancerous[genotype,:], hpu.true(self.cin[genotype,:]))
        has_date    = hpu.idefinedi(self.date_cancerous[genotype,:], not_current)
        inds        = hpu.itrue(self.t >= self.date_cancerous[genotype,has_date], has_date)

        # Set infectious states
        self.susceptible[:, inds] = False  # No longer susceptible to any genotype
        self.infectious[:, inds] = False  # No longer counted as infectious with any genotype
        self.inactive[:,inds] = True  # If this person has any other infections from any other genotypes, set them to inactive
        self.date_clearance[:, inds] = np.nan  # Remove their clearance dates for all genotypes

        # Deal with dysplasia states and dates
        for g in range(self.ng):
            if g != genotype:
                self.date_cancerous[g, inds] = np.nan  # Remove their date of cancer for all genotypes but the one currently causing cancer
                self.date_cin[g, inds] = np.nan

        # Set the properties related to cell changes and disease severity markers
        self.cancerous[genotype, inds] = True
        self.cin[:, inds] = False  # No longer counted as episomal with any genotype

        # Age results
        cases_by_age = np.histogram(self.age[inds], bins=self.age_bin_edges, weights=self.scale[inds])[0]

        return cases_by_age, self.scale_flows(inds)


    def check_cancer_deaths(self):
        '''
        Check for new deaths from cancer
        '''
        filter_inds = self.true('cancerous')
        inds = self.check_inds(self.dead_cancer, self.date_dead_cancer, filter_inds=filter_inds)
        self.remove_people(inds, cause='cancer')
        cases_by_age = np.histogram(self.age[inds], bins=self.age_bin_edges, weights=self.scale[inds])[0]

        # check which of these were detected by symptom or screening
        self.flows['detected_cancer_deaths'] += self.scale_flows(hpu.true(self.detected_cancer[inds]))

        return cases_by_age, self.scale_flows(inds)


    def check_clearance(self, genotype):
        '''
        Check for HPV clearance.
        '''
        f_filter_inds = (self.is_female_alive & self.infectious[genotype,:]).nonzero()[-1]
        m_filter_inds = (self.is_male_alive   & self.infectious[genotype,:]).nonzero()[-1]
        f_inds = self.check_inds_true(self.infectious[genotype,:], self.date_clearance[genotype,:], filter_inds=f_filter_inds)
        m_inds = self.check_inds_true(self.infectious[genotype,:], self.date_clearance[genotype,:], filter_inds=m_filter_inds)
        m_cleared_inds = m_inds # All males clear

        # For females, determine who clears and who controls
        if self.pars['hpv_control_prob']>0:
            latent_probs = np.full(len(f_inds), self.pars['hpv_control_prob'], dtype=hpd.default_float)
            latent_bools = hpu.binomial_arr(latent_probs)
            latent_inds = f_inds[latent_bools]

            if len(latent_inds):
                self.susceptible[genotype, latent_inds] = False  # should already be false
                self.infectious[genotype, latent_inds] = False
                self.inactive[genotype, latent_inds] = True
                self.date_clearance[genotype, latent_inds] = np.nan
                self.date_latent[genotype, latent_inds] = self.t
            f_cleared_inds = f_inds[~latent_bools]

        else:
            f_cleared_inds = f_inds

        cleared_inds = np.array(m_cleared_inds.tolist()+f_cleared_inds.tolist())

        # Now reset disease states
        if len(cleared_inds):
            self.susceptible[genotype, cleared_inds] = True
            self.infectious[genotype, cleared_inds] = False
            self.inactive[genotype, cleared_inds] = False # should already be false

        if len(f_cleared_inds):
            hpimm.update_peak_immunity(self, f_cleared_inds, imm_pars=self.pars, imm_source=genotype) # update immunity
            self.date_reactivated[genotype, f_cleared_inds] = np.nan

        # Whether infection is controlled on not, clear all cell changes and severity markeres
        self.cin[genotype, f_inds] = False
        self.date_cin[genotype, f_inds] = np.nan
        self.dur_cin[genotype, f_inds] = np.nan
        self.dur_precin[genotype, f_inds] = np.nan
        self.dur_infection[genotype, f_inds] = np.nan

        return


    def apply_death_rates(self, year=None):
        '''
        Apply death rates to remove people from the population
        NB people are not actually removed to avoid issues with indices
        '''

        death_pars = self.pars['death_rates']
        all_years = np.array(list(death_pars.keys()))
        base_year = all_years[0]
        age_bins = death_pars[base_year]['m'][:,0]
        age_inds = np.digitize(self.age, age_bins)-1
        death_probs = np.empty(len(self), dtype=hpd.default_float)
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]
        mx_f = death_pars[nearest_year]['f'][:,1]*self.pars['dt_demog']
        mx_m = death_pars[nearest_year]['m'][:,1]*self.pars['dt_demog']

        death_probs[self.is_female] = mx_f[age_inds[self.is_female]]
        death_probs[self.is_male] = mx_m[age_inds[self.is_male]]
        death_probs[self.age>100] = 1 # Just remove anyone >100
        death_probs[~self.alive] = 0
        death_probs *= self.pars['rel_death'] # Adjust overall death probabilities

        # Get indices of people who die of other causes
        death_inds = hpu.true(hpu.binomial_arr(death_probs))
        deaths_female = self.scale_flows(hpu.true(self.is_female[death_inds]))
        deaths_male = self.scale_flows(hpu.true(self.is_male[death_inds]))
        other_deaths = self.remove_people(death_inds, cause='other') # Apply deaths

        return other_deaths, deaths_female, deaths_male


    def add_births(self, year=None, new_births=None, ages=0, immunity=None, sex_ratio=0.5):
        '''
        Add more people to the population

        Specify either the year from which to retrieve the birth rate, or the absolute number
        of new people to add. Must specify one or the other. People are added in-place to the
        current `People` instance.
        '''

        assert (year is None) != (new_births is None), 'Must set either year or n_births, not both'

        if new_births is None:
            years = self.pars['birth_rates'][0]
            rates = self.pars['birth_rates'][1]
            this_birth_rate = self.pars['rel_birth']*np.interp(year, years, rates)*self.pars['dt_demog']/1e3
            new_births = sc.randround(this_birth_rate*self.n_alive_level0) # Crude births per 1000

        if new_births>0:
            # Generate other characteristics of the new people
            uids, sexes, debuts, rel_sev, partners, cluster = hppop.set_static(new_n=new_births, existing_n=len(self),
                                                                           pars=self.pars, sex_ratio=sex_ratio)
            # Grow the arrays`
            new_inds = self._grow(new_births)
            self.uid[new_inds]          = uids
            self.age[new_inds]          = ages
            self.scale[new_inds]        = self.pars['pop_scale']
            self.sex[new_inds]          = sexes
            self.debut[new_inds]        = debuts
            self.rel_sev[new_inds]      = rel_sev
            self.partners[:,new_inds]   = partners
            self.cluster[new_inds]      = cluster

            if immunity is not None:
                self.nab_imm[:,new_inds] = immunity


        return new_births*self.pars['pop_scale'] # These are not indices, so they scale differently


    def check_migration(self, year=None):
        """
        Check if people need to immigrate/emigrate in order to make the population
        size correct.
        """

        if self.pars['use_migration'] and self.pop_trend is not None:

            # Pull things out
            sim_start = self.pars['start']
            sim_pop0 = self.pars['n_agents']
            data_years = self.pop_trend.year.values
            data_pop = self.pop_trend.pop_size.values
            data_min = data_years[0]
            data_max = data_years[-1]
            age_dist_data = self.pop_age_trend[self.pop_age_trend.year == int(year)]

            # No migration if outside the range of the data
            if year < data_min:
                return 0
            elif year > data_max:
                return 0
            if sim_start < data_min: # Figure this out later, can't use n_agents then
                errormsg = 'Starting the sim earlier than the data is not hard, but has not been done yet'
                raise NotImplementedError(errormsg)

            # Do basic calculations
            data_pop0 = np.interp(sim_start, data_years, data_pop)
            scale = sim_pop0 / data_pop0 # Scale factor
            alive_inds = hpu.true(self.alive_level0)
            alive_ages = self.age[alive_inds].astype(int) # Return ages for everyone level 0 and alive
            count_ages = np.bincount(alive_ages, minlength=age_dist_data.shape[0]) # Bin and count them
            expected = age_dist_data['PopTotal'].values*scale # Compute how many of each age we would expect in population
            difference = (expected-count_ages).astype(int) # Compute difference between expected and simulated for each age
            n_migrate = np.sum(difference) # Compute total migrations (in and out)
            ages_to_remove = hpu.true(difference<0) # Ages where we have too many, need to apply emigration
            n_to_remove = difference[ages_to_remove] # Determine number of agents to remove for each age
            ages_to_add = hpu.true(difference>0) # Ages where we have too few, need to apply imigration
            n_to_add = difference[ages_to_add] # Determine number of agents to add for each age
            ages_to_add_list = np.repeat(ages_to_add, n_to_add)
            self.add_births(new_births=len(ages_to_add_list), ages=np.array(ages_to_add_list))

            # Remove people
            remove_frac = n_to_remove / count_ages[ages_to_remove]
            remove_probs = np.zeros(len(self))
            for ind,rf in enumerate(remove_frac):
                age = ages_to_remove[ind]
                inds_this_age = hpu.true((self.age>=age) * (self.age<age+1) * self.alive_level0)
                remove_probs[inds_this_age] = -rf
            migrate_inds = hpu.choose_w(remove_probs, -n_to_remove.sum())
            self.remove_people(migrate_inds, cause='emigration')  # Remove people

        else:
            n_migrate = 0

        return n_migrate*self.pars['pop_scale'] # These are not indices, so they scale differently



    #%% Methods to make events occur (death, infection, others TBC)
    def make_naive(self, inds):
        '''
        Make a set of people naive. This is used during dynamic resampling.

        Args:
            inds (array): list of people to make naive
        '''
        for key in self.meta.states:
            if key in ['susceptible']:
                self[key][:, inds] = True
            elif key in ['other_dead']:
                self[key][inds] = False
            else:
                self[key][:, inds] = False

        # Reset immunity
        for key in self.meta.imm_states:
            self[key][:, inds] = 0

        # Reset dates
        for key in self.meta.dates + self.meta.durs:
            self[key][:, inds] = np.nan

        return


    def infect(self, inds, g=None, layer=None):
        '''
        Infect people and determine their eventual outcomes.
        Method also deduplicates input arrays in case one agent is infected many times
        and stores who infected whom in infection_log list.

        Args:
            inds      (array): array of people to infect
            g         (int):   int of genotype to infect people with
            layer     (str):   contact layer this infection was transmitted on

        Returns:
            count (int): number of people infected
        '''

        if len(inds) == 0:
            return 0

        # Check whether anyone is already infected with genotype - this should not happen because we only
        # infect susceptible people
        if len(hpu.true(self.infectious[g,inds])):
            errormsg = f'Attempting to reinfect the following agents who are already infected with genotype {g}: {hpu.itruei(self.infectious[g,:],inds)}'
            raise ValueError(errormsg)

        dt = self.pars['dt']

        # Set date of infection and exposure
        base_t = self.t
        self.date_infectious[g,inds] = base_t
        if layer != 'reactivation':
            self.date_exposed[g,inds] = base_t

        # Count reinfections and remove any previous dates
        self.genotype_flows['reinfections'][g]  += self.scale_flows((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        self.flows['reinfections']              += self.scale_flows((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        self.n_infections[g,inds] += 1
        for key in ['date_clearance']:
            self[key][g, inds] = np.nan

        # Count reactivations and adjust latency status
        if layer == 'reactivation':
            self.genotype_flows['reactivations'][g] += self.scale_flows(inds)
            self.flows['reactivations']             += self.scale_flows(inds)
            self.age_flows['reactivations']         += np.histogram(self.age[inds], bins=self.age_bin_edges, weights=self.scale[inds])[0]
            self.latent[g, inds] = False # Adjust states -- no longer latent
            self.date_reactivated[g,inds]           = base_t

        # Update states, genotype info, and flows
        self.susceptible[g, inds]   = False # no longer susceptible
        self.infectious[g, inds]    = True  # now infectious
        self.inactive[g, inds]      = False  # no longer inactive

        # Add to flow results. Note, we only count these infectious in the results if they happened at this timestep
        if layer != 'seed_infection' and layer !='reactivation':
            # Create overall flows
            self.flows['infections']                += self.scale_flows(inds) # Add the total count to the total flow data
            self.genotype_flows['infections'][g]    += self.scale_flows(inds) # Add the count by genotype to the flow data
            self.age_flows['infections'][:]         += np.histogram(self.age[inds], bins=self.age_bin_edges, weights=self.scale[inds])[0]

            # Create by-sex flows
            infs_female = self.scale_flows(hpu.true(self.is_female[inds]))
            infs_male = self.scale_flows(hpu.true(self.is_male[inds]))
            self.sex_flows['infections_by_sex'][0] += infs_female
            self.sex_flows['infections_by_sex'][1] += infs_male

        # Now use genotype-specific prognosis probabilities to determine what happens.
        # Only women can progress beyond infection.
        f_inds = hpu.itruei(self.is_female,inds)
        m_inds = hpu.itruei(self.is_male,inds)

        # Compute disease progression for females
        if len(f_inds)>0:
            gpars = self.pars['genotype_pars'][g]
            self.set_prognoses(f_inds, g, gpars, dt)

        # Compute infection clearance for males
        if len(m_inds)>0:
            dur_infection = hpu.sample(**self.pars['dur_infection_male'], size=len(m_inds))
            self.dur_infection[g, m_inds] = dur_infection
            self.date_clearance[g, m_inds] = self.date_infectious[g, m_inds] + np.ceil(dur_infection/dt)  # Date they clear HPV infection (interpreted as the timestep on which they recover)

        return self.scale_flows(inds) # For incrementing counters


    def remove_people(self, inds, cause=None):
        ''' Remove people - used for death and migration '''

        if cause == 'other':
            self.date_dead_other[inds] = self.t
            self.dead_other[inds] = True
        elif cause == 'cancer':
            self.dead_cancer[inds] = True
        elif cause == 'emigration':
            self.emigrated[inds] = True
        elif cause == 'hiv':
            self.dead_hiv[inds] = True
        else:
            errormsg = f'Cause of death must be one of "other", "cancer", "emigration", or "hiv", not {cause}.'
            raise ValueError(errormsg)

        # Set states to false
        self.alive[inds] = False
        for state in self.meta.genotype_stock_keys:
            self[state][:, inds] = False
        for state in self.meta.intv_stock_keys:
            self[state][inds] = False
        for state in self.meta.other_stock_keys:
            self[state][inds] = False

        # Wipe future dates
        future_dates = [date.name for date in self.meta.dates]
        for future_date in future_dates:
            ndims = len(self[future_date].shape)
            if ndims == 1:
                iinds = (self[future_date][inds] > self.t).nonzero()[-1]
                if len(iinds):
                    self[future_date][inds[iinds]] = np.nan
            elif ndims == 2:
                genotypes_to_clear, iinds = (self[future_date][:, inds] >= self.t).nonzero()
                if len(iinds):
                    self[future_date][genotypes_to_clear, inds[iinds]] = np.nan

        return self.scale_flows(inds)


    #%% Analysis methods

    def plot(self, *args, **kwargs):
        '''
        Plot statistics of the population -- age distribution, numbers of contacts,
        and overall weight of contacts (number of contacts multiplied by beta per
        layer).

        Args:
            bins      (arr)   : age bins to use (default, 0-100 in one-year bins)
            width     (float) : bar width
            font_size (float) : size of font
            alpha     (float) : transparency of the plots
            fig_args  (dict)  : passed to pl.figure()
            axis_args (dict)  : passed to pl.subplots_adjust()
            plot_args (dict)  : passed to pl.plot()
            do_show   (bool)  : whether to show the plot
            fig       (fig)   : handle of existing figure to plot into
        '''
        fig = hpplt.plot_people(people=self, *args, **kwargs)
        return fig


    def story(self, uid, *args):
        '''
        Print out a short history of events in the life of the specified individual.

        Args:
            uid (int/list): the person or people whose story is being regaled
            args (list): these people will tell their stories too

        **Example**::

            sim = hpv.Sim(pop_type='hybrid', verbose=0)
            sim.run()
            sim.people.story(12)
            sim.people.story(795)
        '''

        def label_lkey(lkey):
            ''' Friendly name for common layer keys '''
            if lkey.lower() == 'a':
                llabel = 'default contact'
            if lkey.lower() == 'm':
                llabel = 'marital'
            elif lkey.lower() == 'c':
                llabel = 'casual'
            else:
                llabel = f'"{lkey}"'
            return llabel

        uids = sc.promotetolist(uid)
        uids.extend(args)

        for uid in uids:

            p = self[uid]
            sex = 'female' if p.sex == 0 else 'male'

            intro  = f'\nThis is the story of {uid}, a {p.age:.0f} year old {sex}.'
            intro += f'\n{uid} became sexually active at age {p.debut:.0f}.'
            if not p.susceptible:
                if ~np.isnan(p.date_infectious):
                    print(f'{intro}\n{uid} contracted HPV on timestep {p.date_infectious} of the simulation.')
                else:
                    print(f'{intro}\n{uid} did not contract HPV during the simulation.')

            total_contacts = 0
            no_contacts = []
            for lkey in p.contacts.keys():
                llabel = label_lkey(lkey)
                n_contacts = len(p.contacts[lkey])
                total_contacts += n_contacts
                if n_contacts:
                    print(f'{uid} is connected to {n_contacts} people in the {llabel} layer')
                else:
                    no_contacts.append(llabel)
            if len(no_contacts):
                nc_string = ', '.join(no_contacts)
                print(f'{uid} has no contacts in the {nc_string} layer(s)')
            print(f'{uid} has {total_contacts} contacts in total')

            events = []

            dates = {
                'date_HPV_clearance'      : 'HPV cleared',
            }

            for attribute, message in dates.items():
                date = getattr(p,attribute)
                if not np.isnan(date):
                    events.append((date, message))

            if len(events):
                for timestep, event in sorted(events, key=lambda x: x[0]):
                    print(f'On timestep {timestep:.0f}, {uid} {event}')
            else:
                print(f'Nothing happened to {uid} during the simulation.')
        return

