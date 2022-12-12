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
        self.na = len(self.pars['age_bins'])-1
        self.dysp_keys = ['cin1s', 'cin2s', 'cin3s', 'cancers']

        self.lag_bins = np.linspace(0,50,51)
        self.rship_lags = dict()
        for lkey in self.layer_keys():
            self.rship_lags[lkey] = np.zeros(len(self.lag_bins)-1, dtype=hpd.default_float)

        # Store age bins
        self.age_bins = self.pars['age_bins'] # Age bins for age results
        #self.asr_bins = self.pars['standard_pop'][0, :] # Age bins of the standard population

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


    def initialize(self, sim_pars=None, hiv_pars=None):
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
        self.set_pars(pars=sim_pars, hiv_pars=hiv_pars) # Replace the saved parameters with this simulation's
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

        # Check for HIV acquisitions
        if self.pars['model_hiv']:
            self.flows['hiv_infections'] = self.apply_hiv_rates(year=year)

        # Perform updates that are not genotype-specific
        update_freq = max(1, int(self.pars['dt_demog'] / self.pars['dt'])) # Ensure it's an integer not smaller than 1
        if t % update_freq == 0:

            # Apply death rates from other causes
            other_deaths, deaths_female, deaths_male    = self.apply_death_rates(year=year)
            self.demographic_flows['other_deaths']      = other_deaths
            self.sex_flows['other_deaths_by_sex'][0]    = deaths_female
            self.sex_flows['other_deaths_by_sex'][1]    = deaths_male

            # Add births
            new_births = self.add_births(year=year)
            self.demographic_flows['births'] = new_births

            # Check migration
            migration = self.check_migration(year=year)
            self.demographic_flows['migration'] = migration

        # Perform updates that are genotype-specific
        ng = self.pars['n_genotypes']
        for g in range(ng):
            for key in self.dysp_keys: # Loop over the keys related to dysplasia
                cases_by_age, cases = self.check_progress(key, g)
                self.flows[key] += cases # Increment flows (summed over all genotypes)
                self.genotype_flows[key][g] = cases # Store flows by genotype
                self.age_flows[key] += cases_by_age # Increment flows by age (summed over all genotypes)
            self.genotype_flows['cins'][g] = self.genotype_flows['cin1s'][g]+self.genotype_flows['cin2s'][g]+self.genotype_flows['cin3s'][g]
            self.check_clearance(g)

        # Perform updates that are not genotype specific
        self.flows['cancer_deaths'] = self.check_cancer_deaths()
        self.flows['cins'] = self.genotype_flows['cins'].sum()

        # Before applying interventions or new infections, calculate the pool of susceptibles
        self.sus_pool = self.susceptible.all(axis=0) # True for people with no infection at the start of the timestep

        return

    
    #%% Disease progression methods
    def set_prognoses(self, inds, g, dt, hiv_pars=None):
        '''
        Set prognoses for people following infection. Wrapper method that calls
        the 4 separate methods for setting and updating prognoses.
        '''
        gpars = self.pars['genotype_pars'][self.pars['genotype_map'][g]]
        self.set_dysp_rates(inds, g, gpars, hiv_dysp_rate=hiv_pars['dysp_rate'])  # Set variables that determine the probability that dysplasia begins
        dysp_inds = self.set_dysp_status(inds, g, dt)  # Set people's dysplasia status
        dysp_arrs = self.set_severity(dysp_inds, g, gpars, hiv_prog_rate=hiv_pars['prog_rate'])  # Set dysplasia severity and duration
        self.set_cin_grades(dysp_inds, g, dt, dysp_arrs=dysp_arrs)  # Set CIN grades and dates over time
        return


    def set_dysp_status(self, inds, g, dt):
        '''
        Use durations and dysplasia rates to determine whether HPV clears or progresses to dysplasia
        '''
        dur_precin  = self.dur_precin[g, inds]  # Array of durations of infection prior to dysplasia/clearance/control
        dysp_rate   = self.dysp_rate[g, inds]  # Array of dysplasia rates
        dysp_probs  = hpu.logf1(dur_precin, dysp_rate)  # Probability of developing dysplasia
        has_dysp    = hpu.binomial_arr(dysp_probs)  # Boolean array of those who have dysplasia
        nodysp_inds = inds[~has_dysp]  # Indices of those without dysplasia
        dysp_inds   = inds[has_dysp]  # Indices of those with dysplasia

        # Infection clears without causing dysplasia
        self.date_clearance[g, nodysp_inds] = self.date_infectious[g, nodysp_inds]+ np.ceil(self.dur_infection[g, nodysp_inds] / dt)  # Date they clear HPV infection (interpreted as the timestep on which they recover)

        # Infection progresses to dysplasia, which is initially classified as CIN1 - set dates for this
        excl_inds = hpu.true(self.date_cin1[g, dysp_inds] < self.t)  # Don't count CIN1s that were acquired before now
        self.date_cin1[g, dysp_inds[excl_inds]] = np.nan
        self.date_cin1[g, dysp_inds] = np.fmin(self.date_cin1[g, dysp_inds],
                                               self.date_infectious[g, dysp_inds] +
                                               sc.randround(self.dur_precin[g, dysp_inds] / dt))  # Date they develop CIN1 - minimum of the date from their new infection and any previous date

        return dysp_inds


    def set_severity(self, inds, g, gpars, hiv_prog_rate=None):
        ''' Set dysplasia severity and duration for women who develop dysplasia '''
        
        dysp_arrs = sc.objdict() # Store severity arrays
        
        # Evaluate duration of dysplasia prior to clearance/control/progression to cancer
        n_cols = self.pars['ms_agent_ratio'] if self.pars['use_multiscale'] else 1
        full_size = (len(inds), n_cols) # Main axis is indices, but include columns for multiscale agents
        dur_dysp = hpu.sample(**gpars['dur_dysp'], size=full_size)
        self.dur_infection[g, inds] += dur_dysp[:,0] # TODO: should this be mean(axis=1) instead?

        # Evaluate progression rates
        prog_rate = hpu.sample(dist='normal', par1=gpars['prog_rate'], par2=gpars['prog_rate_sd'], size=full_size)
        has_hiv = self.hiv[inds]
        if has_hiv.any():  # Figure out if any of these women have HIV
            immune_compromise = 1 - self.art_adherence[inds]  # Get the degree of immunocompromise
            modified_prog_rate = immune_compromise * hiv_prog_rate  # Calculate the modification to make to the progression rate
            modified_prog_rate[modified_prog_rate < 1] = 1
            prog_rate *= modified_prog_rate[:, None]  # Store progression rates -- see https://stackoverflow.com/questions/19388152/numpy-element-wise-multiplication-of-an-array-and-a-vector

        # Calculate peak dysplasia
        peak_dysp = hpu.logf1(dur_dysp, prog_rate)  # Maps durations + progression to severity
        
        dysp_arrs.dur_dysp  = dur_dysp
        dysp_arrs.prog_rate = prog_rate
        dysp_arrs.peak_dysp = peak_dysp

        return dysp_arrs


    def set_dysp_rates(self, inds, g, gpars, hiv_dysp_rate=None):
        '''
        Set dysplasia rates
        '''
        self.dysp_rate[g, inds] = gpars['dysp_rate']
        has_hiv = self.hiv[inds]
        if has_hiv.any():  # Figure out if any of these women have HIV
            immune_compromise = 1 - self.art_adherence[inds]  # Get the degree of immunocompromise
            modified_dysp_rate = immune_compromise * hiv_dysp_rate  # Calculate the modification to make to the dysplasia rate
            modified_dysp_rate[modified_dysp_rate < 1] = 1
            self.dysp_rate[g, inds] = self.dysp_rate[g, inds] * modified_dysp_rate  # Store dysplasia rates
        return


    def set_cin_grades(self, inds, g, dt, dysp_arrs):
        '''
        Set CIN clinical grades and dates of progression
        '''

        # Map severity to clinical grades
        ccut = self.pars['clinical_cutoffs']
        peak_dysp = dysp_arrs.peak_dysp[:,0] # Everything beyond 0 is multiscale agents
        prog_rate = dysp_arrs.prog_rate[:,0]
        dur_dysp  = dysp_arrs.dur_dysp[:,0]
        gpars = self.pars['genotype_pars'][self.pars['genotype_map'][g]]
        cancer_prob = gpars['cancer_prob']

        # Handle multiscale to create additional cancer agents
        n_extra = self.pars['ms_agent_ratio'] # Number of extra cancer agents per regular agent
        cancer_scale = self.pars['pop_scale'] / n_extra
        if self.pars['use_multiscale'] and n_extra  > 1:
            is_cin3 = peak_dysp > ccut['cin2']
            cancer_probs = np.zeros(len(inds))
            cancer_probs[is_cin3] = cancer_prob
            is_cancer = hpu.binomial_arr(cancer_probs)
            cancer_inds = inds[is_cancer]  # Duplicated below, but avoids need to append extra arrays
            self.scale[cancer_inds] = cancer_scale  # Shrink the weight of the original agents, but otherwise leave them the same
            extra_peak_dysp = dysp_arrs.peak_dysp[:, 1:]
            extra_cin3_bools = extra_peak_dysp > ccut['cin2']
            extra_cancer_probs = np.zeros_like(extra_cin3_bools, dtype=hpd.default_float) # For storing probs that CIN3 agents will advance to cancer
            extra_cancer_probs[extra_cin3_bools] = cancer_prob # Prob of cancer is zero for agents without CIN3
            extra_cancer_bools = hpu.binomial_arr(extra_cancer_probs)
            extra_cancer_bools *= self.level0[inds, None]  # Don't allow existing cancer agents to make more cancer agents
            extra_cancer_counts = extra_cancer_bools.sum(axis=1)  # Find out how many new cancer cases we have
            n_new_agents = extra_cancer_counts.sum()  # Total number of new agents
            if n_new_agents:  # If we have more than 0, proceed
                extra_source_lists = []
                for i, count in enumerate(extra_cancer_counts):
                    ii = inds[i]
                    if count:  # At least 1 new cancer agent, plus person is not already a cancer agent
                        extra_source_lists.append([ii] * int(count))  # Duplicate the curret index count times
                extra_source_inds = np.concatenate(
                    extra_source_lists).flatten()  # Assemble the sources for these new agents
                n_new_agents = len(
                    extra_source_inds)  # The same as above, *unless* a cancer agent tried to spawn more cancer agents

                # Create the new agents and assign them the same properties as the existing agents
                new_inds = self._grow(n_new_agents)
                for state in self.meta.all_states:
                    if state.ndim == 1:
                        self[state.name][new_inds] = self[state.name][extra_source_inds]
                    elif state.ndim == 2:
                        self[state.name][:, new_inds] = self[state.name][:, extra_source_inds]

                # Reset the states for the new agents
                self.level0[new_inds] = False
                self.level1[new_inds] = True
                self.scale[new_inds] = cancer_scale
                
                # Sneakily add the new indices onto the existing vectors
                inds = np.append(inds, new_inds)
                new_peak_dysp = extra_peak_dysp[extra_cancer_bools]
                new_prog_rate = dysp_arrs.prog_rate[:,1:][extra_cancer_bools]
                new_dur_dysp  = dysp_arrs.dur_dysp[:,1:][extra_cancer_bools]
                peak_dysp     = np.append(peak_dysp, new_peak_dysp)
                prog_rate     = np.append(prog_rate, new_prog_rate)
                dur_dysp      = np.append(dur_dysp,  new_dur_dysp)
                is_cancer = np.append(is_cancer, np.full(len(new_inds), fill_value=True))
            
        # Now check indices, including with our new cancer agents
        is_cin1 = peak_dysp > 0  # Boolean arrays of people who attain each clinical grade
        is_cin2 = peak_dysp > ccut['cin1']
        is_cin3 = peak_dysp > ccut['cin2']
        cancer_probs = np.zeros(len(inds))
        if self.pars['use_multiscale'] and n_extra > 1:
            cancer_probs[is_cancer] = 1 # Make sure inds that got assigned cancer above dont get stochastically missed
        else:
            cancer_probs[is_cin3] = cancer_prob
        is_cancer = hpu.binomial_arr(cancer_probs)
        cin2_inds = inds[is_cin2]  # Indices of those progress at least to CIN2
        cin3_inds = inds[is_cin3]  # Indices of those progress at least to CIN3
        cancer_inds = inds[is_cancer]  # Indices of those progress to cancer
        max_cin1_bools = is_cin1 * ~is_cin2   # Boolean of those who don't progress beyond CIN1
        max_cin2_bools = is_cin2 * ~is_cin3   # Boolean of those who don't progress beyond CIN2
        max_cin3_bools = is_cin3 * ~is_cancer # Boolean of those who don't progress beyond CIN3
        max_cin1_inds = inds[max_cin1_bools]  # Indices of those who don't progress beyond CIN1
        max_cin2_inds = inds[max_cin2_bools]  # Indices of those who don't progress beyond CIN2
        max_cin3_inds = inds[max_cin3_bools]  # Indices of those who don't progress beyond CIN3

        # Determine whether CIN1 clears or progresses to CIN2
        self.date_cin2[g, cin2_inds] = np.fmax(self.t, # Don't let people progress to CIN2 prior to the current timestep
                                               self.date_cin1[g, cin2_inds] +
                                               sc.randround(hpu.invlogf1(ccut['cin1'], prog_rate[is_cin2]) / dt))
        time_to_clear_cin1 = dur_dysp[max_cin1_bools]
        self.date_clearance[g, max_cin1_inds] = np.fmax(self.date_clearance[g, max_cin1_inds],
                                                        self.date_cin1[g, max_cin1_inds] +
                                                        sc.randround(time_to_clear_cin1 / dt))

        # Determine whether CIN2 clears or progresses to CIN3
        self.date_cin3[g, cin3_inds] = np.fmax(self.t, # Don't let people progress to CIN3 prior to the current timestep
                                               self.date_cin1[g, cin3_inds] +
                                               sc.randround(hpu.invlogf1(ccut['cin2'], prog_rate[is_cin3]) / dt))

        # Compute how much dysplasia time is left for those who clear (total dysplasia duration - dysplasia time spent prior to this grade)
        time_to_clear_cin2 = dur_dysp[max_cin2_bools] - (self.date_cin2[g, max_cin2_inds] - self.date_cin1[g, max_cin2_inds]) * self.pars['dt']
        self.date_clearance[g, max_cin2_inds] = np.fmax(self.date_clearance[g, max_cin2_inds],
                                                        self.date_cin2[g, max_cin2_inds] +
                                                        sc.randround(time_to_clear_cin2 / dt))

        # Determine whether CIN3 clears or progresses to cancer
        time_to_cancer = dur_dysp[is_cancer] - (self.date_cin3[g, cancer_inds] - self.date_cin1[g, cancer_inds]) * self.pars['dt']
        self.date_cancerous[g, cancer_inds] = np.fmax(self.t,
                                                      self.date_cin3[g, cancer_inds] +
                                                      sc.randround(time_to_cancer / dt))

        # Compute how much dysplasia time is left for those who clear (total dysplasia duration - dysplasia time spent prior to this grade)
        time_to_clear_cin3 = dur_dysp[max_cin3_bools] - (self.date_cin3[g, max_cin3_inds] - self.date_cin1[g, max_cin3_inds]) * self.pars['dt']
        self.date_clearance[g, max_cin3_inds] = np.fmax(self.date_clearance[g, max_cin3_inds],
                                                        self.date_cin3[g, max_cin3_inds] +
                                                        sc.randround(time_to_clear_cin3 / dt))

        # Record eventual deaths from cancer (assuming no survival without treatment)
        dur_cancer = hpu.sample(**self.pars['dur_cancer'], size=len(cancer_inds))
        self.date_dead_cancer[cancer_inds] = self.date_cancerous[g, cancer_inds] + sc.randround(dur_cancer / dt)
        self.dur_cancer[g, cancer_inds] = dur_cancer

        return

    
    def set_hiv_prognoses(self, inds, year=None):
        ''' Set HIV outcomes (for now only ART) '''
    
        art_cov = self.hiv_pars.art_adherence # Shorten
    
        # Extract index of current year
        all_years = np.array(list(art_cov.keys()))
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]
    
        # Figure out which age bin people belong to
        age_bins = art_cov[nearest_year][0, :]
        age_inds = np.digitize(self.age[inds], age_bins)
    
        # Apply ART coverage by age to people
        art_covs = art_cov[nearest_year][1,:]
        art_adherence = art_covs[age_inds]
        self.art_adherence[inds] = art_adherence
    
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


    def create_partnerships(self, tind, mixing, layer_probs, cross_layer, dur_pship, acts, age_act_pars, pref_weight=100):
        '''
        Create partnerships. All the hard work of creating the contacts is done by hppop.make_contacts,
        which in turn relies on hpu.create_edgelist for creating the edgelist. This method is just a light wrapper
        that passes in the arguments in the right format and the updates relationship info stored in the People class.
        '''
        # Initialize
        new_pships = dict()

        # Loop over layers
        for lno, lkey in enumerate(self.layer_keys()):
            pship_args = dict(
                lno=lno, tind=tind, partners=self.partners[lno], current_partners=self.current_partners,
                sexes=self.sex, ages=self.age, debuts=self.debut, is_female=self.is_female, is_active=self.is_active,
                mixing=mixing[lkey], layer_probs=layer_probs[lkey], cross_layer=cross_layer,
                pref_weight=pref_weight, durations=dur_pship[lkey], acts=acts[lkey], age_act_pars=age_act_pars[lkey]
            )
            new_pships[lkey], current_partners, new_pship_inds, new_pship_counts = hppop.make_contacts(**pship_args)

            # Update relationship info
            self.current_partners[:] = current_partners
            if len(new_pship_inds):
                self.rship_start_dates[lno, new_pship_inds] = self.t
                self.n_rships[lno, new_pship_inds] += new_pship_counts
                lags = self.rship_start_dates[lno, new_pship_inds] - self.rship_end_dates[lno, new_pship_inds]
                self.rship_lags[lkey] += np.histogram(lags, self.lag_bins)[0]

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
        if what=='cin1s':       cases_by_age, cases = self.check_cin1(genotype)
        elif what=='cin2s':     cases_by_age, cases = self.check_cin2(genotype)
        elif what=='cin3s':     cases_by_age, cases = self.check_cin3(genotype)
        elif what=='cancers':   cases_by_age, cases = self.check_cancer(genotype)
        return cases_by_age, cases

    def check_cin1(self, genotype):
        ''' Check for new progressions to CIN1 '''
        # Only include infectious females who haven't already cleared CIN1 or progressed to CIN2
        filters = self.infectious[genotype,:]*self.is_female*~(self.date_clearance[genotype,:]<=self.t)*(self.date_cin2[genotype,:]>=self.t)
        filter_inds = filters.nonzero()[0]
        inds = self.check_inds(self.cin1[genotype,:], self.date_cin1[genotype,:], filter_inds=filter_inds)
        self.cin1[genotype, inds] = True
        self.no_dysp[genotype, inds] = False
        # Age calculations
        cases_by_age = np.histogram(self.age[inds], bins=self.age_bins, weights=self.scale[inds])[0]

        return cases_by_age, self.scale_flows(inds)


    def check_cin2(self, genotype):
        ''' Check for new progressions to CIN2 '''
        filter_inds = self.true_by_genotype('cin1', genotype)
        inds = self.check_inds(self.cin2[genotype,:], self.date_cin2[genotype,:], filter_inds=filter_inds)
        self.cin2[genotype, inds] = True
        self.cin1[genotype, inds] = False # No longer counted as CIN1
        # Age calculations
        cases_by_age = np.histogram(self.age[inds], bins=self.age_bins, weights=self.scale[inds])[0]
        return cases_by_age, self.scale_flows(inds)


    def check_cin3(self, genotype):
        ''' Check for new progressions to CIN3 '''
        filter_inds = self.true_by_genotype('cin2', genotype)
        inds = self.check_inds(self.cin3[genotype,:], self.date_cin3[genotype,:], filter_inds=filter_inds)
        self.cin3[genotype, inds] = True
        self.cin2[genotype, inds] = False # No longer counted as CIN2
        # Age calculations
        cases_by_age = np.histogram(self.age[inds], bins=self.age_bins, weights=self.scale[inds])[0]
        return cases_by_age, self.scale_flows(inds)


    def check_cancer(self, genotype):
        ''' Check for new progressions to cancer '''
        filter_inds = self.true_by_genotype('cin3', genotype)
        inds = self.check_inds(self.cancerous[genotype,:], self.date_cancerous[genotype,:], filter_inds=filter_inds)
        cases_by_age = 0

        if len(inds):
            # First, set the SIR properties. Once a person has cancer, their are designated
            # as inactive for all genotypes they may be infected with
            self.susceptible[:, inds] = False # No longer susceptible to any genotype
            self.infectious[:, inds]  = False # No longer counted as infectious with any genotype
            self.date_clearance[:, inds] = np.nan # Remove their clearance dates for all genotypes
            for g in range(self.ng):
                if g!=genotype:
                    self.date_cancerous[g, inds] = np.nan # Remove their date of cancer for all genotypes but the one currently causing cancer
            self.inactive[:, inds] = True # If this person has any other infections from any other genotypes, set them to inactive

            # Next, set the dysplasia properties
            self.cancerous[genotype, inds] = True
            self.cin3[genotype, inds] = False # No longer counted as CIN3

            # Age results
            cases_by_age = np.histogram(self.age[inds], bins=self.age_bins, weights=self.scale[inds])[0]

        return cases_by_age, self.scale_flows(inds)


    def check_cancer_deaths(self):
        '''
        Check for new deaths from cancer
        '''
        filter_inds = self.true('cancerous')
        inds = self.check_inds(self.dead_cancer, self.date_dead_cancer, filter_inds=filter_inds)
        self.remove_people(inds, cause='cancer')
        if len(inds):
            cases_by_age = np.histogram(self.age[inds], bins=self.age_bins, weights=self.scale[inds])[0]

        # check which of these were detected by symptom or screening
        self.flows['detected_cancer_deaths'] += self.scale_flows(hpu.true(self.detected_cancer[inds]))

        return self.scale_flows(inds)


    def check_clearance(self, genotype):
        '''
        Check for HPV clearance.
        '''
        filter_inds = self.true_by_genotype('infectious', genotype)
        inds = self.check_inds_true(self.infectious[genotype,:], self.date_clearance[genotype,:], filter_inds=filter_inds)

        # Determine who clears and who controls
        latent_probs = np.full(len(inds), self.pars['hpv_control_prob'], dtype=hpd.default_float)
        latent_bools = hpu.binomial_arr(latent_probs)

        latent_inds = inds[latent_bools]
        cleared_inds = inds[~latent_bools]

        # Now reset disease states
        if len(cleared_inds):
            self.susceptible[genotype, cleared_inds] = True
            self.infectious[genotype, cleared_inds] = False
            self.inactive[genotype, cleared_inds] = False # should already be false
            hpimm.update_peak_immunity(self, cleared_inds, imm_pars=self.pars, imm_source=genotype) # update immunity

        if len(latent_inds):
            self.susceptible[genotype, latent_inds] = False # should already be false
            self.infectious[genotype, latent_inds] = False
            self.inactive[genotype, latent_inds] = True
            self.date_clearance[genotype, latent_inds] = np.nan

        # Whether infection is controlled on not, people have no dysplasia, so we clear all this info
        self.no_dysp[genotype, inds] = True
        self.cin1[genotype, inds] = False
        self.cin2[genotype, inds] = False
        self.cin3[genotype, inds] = False
        # self.peak_dysp[genotype, inds] = np.nan
        self.dysp_rate[genotype, inds] = np.nan
        # self.prog_rate[genotype, inds] = np.nan

        return


    def apply_hiv_rates(self, year=None):
        '''
        Apply HIV infection rates to population
        '''
        hiv_pars = self.hiv_pars.infection_rates
        all_years = np.array(list(hiv_pars.keys()))
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]
        hiv_year = hiv_pars[nearest_year]
        dt = self.pars['dt']

        hiv_probs = np.zeros(len(self), dtype=hpd.default_float)
        for sk in ['f','m']:
            hiv_year_sex = hiv_year[sk]
            age_bins = hiv_year_sex[:,0]
            hiv_rates = hiv_year_sex[:,1]*dt
            mf_inds = self.is_female if sk == 'f' else self.is_male
            mf_inds *= self.alive # Only include people alive
            age_inds = np.digitize(self.age[mf_inds], age_bins)
            hiv_probs[mf_inds]  = hiv_rates[age_inds]
        hiv_probs[self.hiv] = 0 # not at risk if already infected

        # Get indices of people who acquire HIV
        hiv_inds = hpu.true(hpu.binomial_arr(hiv_probs))
        self.hiv[hiv_inds] = True

        # Update prognoses for those with HIV
        if len(hiv_inds):
            
            self.set_hiv_prognoses(hiv_inds, year=year) # Set ART adherence for those with HIV

            for g in range(self.pars['n_genotypes']):
                gpars = self.pars['genotype_pars'][self.pars['genotype_map'][g]]
                nocin_inds = hpu.itruei((self.is_female & self.precin[g, :] & np.isnan(self.date_cin1[g, :])), hiv_inds) # Women with HIV who are scheduled to clear without dysplasia
                if len(nocin_inds): # Reevaluate whether these women will develop dysplasia
                    self.set_dysp_rates(nocin_inds, g, gpars, hiv_dysp_rate=self.pars['hiv_pars']['dysp_rate'])
                    self.set_dysp_status(nocin_inds, g, dt)

                cin_inds = hpu.itruei((self.is_female & self.infectious[g, :] & ~np.isnan(self.date_cin1[g, :])), hiv_inds) # Women with HIV who are scheduled to have dysplasia
                if len(cin_inds): # Reevaluate disease severity and progression speed for these women
                    dysp_arrs = self.set_severity(cin_inds, g, gpars, hiv_prog_rate=self.pars['hiv_pars']['prog_rate'])
                    self.set_cin_grades(cin_inds, g, dt, dysp_arrs=dysp_arrs)

        return self.scale_flows(hiv_inds)


    def apply_death_rates(self, year=None):
        '''
        Apply death rates to remove people from the population
        NB people are not actually removed to avoid issues with indices
        '''

        if self.t == 0:
            np.random.rand()
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


    def add_births(self, year=None, new_births=None, ages=0, immunity=None):
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
            uids, sexes, debuts, partners = hppop.set_static(new_n=new_births, existing_n=len(self), pars=self.pars)
            
            # Grow the arrays
            new_inds = self._grow(new_births)
            self.uid[new_inds]        = uids
            self.age[new_inds]        = ages
            self.scale[new_inds]      = self.pars['pop_scale']
            self.sex[new_inds]        = sexes
            self.debut[new_inds]      = debuts
            self.partners[:,new_inds] = partners

            if immunity is not None:
                self.imm[:,new_inds] = immunity


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
            n_alive = len(alive_inds)
            expected_old = np.interp(year, data_years, data_pop) * scale
            n_migrate_old = int(expected_old - n_alive)
            ages = self.age[alive_inds].astype(int) # Return ages for everyone level 0 and alive
            count_ages = np.bincount(ages, minlength=age_dist_data.shape[0]) # Bin and count them
            expected = age_dist_data['PopTotal'].values*scale # Compute how many of each age we would expect in population
            difference = np.array([int(i) for i in (expected - count_ages)]) # Compute difference between expected and simulated for each age
            n_migrate = np.sum(difference) # Compute total migrations (in and out)
            # print(f'old method has {n_migrate_old} migrations, new method has {n_migrate} migrations, difference of {abs(n_migrate_old-n_migrate)}')

            ages_to_remove = hpu.true(difference<0) # Ages where we have too many, need to apply emigration
            n_to_remove = [int(i) for i in difference[ages_to_remove]] # Determine number of agents to remove for each age
            ages_to_add = hpu.true(difference>0) # Ages where we have too few, need to apply imigration
            n_to_add = [int(i) for i in difference[ages_to_add]] # Determine number of agents to add for each age
            ages_to_add_list = np.repeat(ages_to_add, n_to_add)
            self.add_births(new_births=len(ages_to_add_list), ages=np.array(ages_to_add_list))

            for ind, diff in enumerate(n_to_remove): #TODO: is there a faster way to do this than in a for loop?
                age = ages_to_remove[ind]
                alive_this_age_inds = np.where(ages==age)[0]
                inds = hpu.choose(len(alive_this_age_inds), -diff)
                migrate_inds = alive_inds[alive_this_age_inds[inds]]
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


    def infect(self, inds, g=None, offset=None, dur=None, layer=None):
        '''
        Infect people and determine their eventual outcomes.
        Method also deduplicates input arrays in case one agent is infected many times
        and stores who infected whom in infection_log list.

        Args:
            inds      (array): array of people to infect
            g         (int):   int of genotype to infect people with
            offset    (array): if provided, the infections will occur at the timepoint self.t+offset
            dur       (array): if provided, the duration of the infections
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

        # Deal with genotype parameters
        genotype_pars   = self.pars['genotype_pars']
        genotype_map    = self.pars['genotype_map']
        dur_precin      = genotype_pars[genotype_map[g]]['dur_precin']

        # Set date of infection and exposure
        base_t = self.t + offset if offset is not None else self.t
        self.date_infectious[g,inds] = base_t
        if layer != 'reactivation':
            self.date_exposed[g,inds] = base_t

        # Count reinfections and remove any previous dates
        self.genotype_flows['reinfections'][g]  += self.scale_flows((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        self.flows['reinfections']              += self.scale_flows((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        for key in ['date_clearance', 'date_cin1', 'date_cin2', 'date_cin3']:
            self[key][g, inds] = np.nan

        # Count reactivations and adjust latency status
        if layer == 'reactivation':
            self.genotype_flows['reactivations'][g] += self.scale_flows(inds)
            self.flows['reactivations']             += self.scale_flows(inds)
            self.age_flows['reactivations']         += np.histogram(self.age[inds], bins=self.age_bins, weights=self.scale[inds])[0]
            self.latent[g, inds] = False # Adjust states -- no longer latent

        # Update states, genotype info, and flows
        self.susceptible[g, inds]   = False # no longer susceptible
        self.infectious[g, inds]    = True  # now infectious
        self.inactive[g, inds]      = False  # no longer inactive

        # Add to flow results. Note, we only count these infectious in the results if they happened at this timestep
        if offset is None:
            # Create overall flows
            self.flows['infections']                += self.scale_flows(inds) # Add the total count to the total flow data
            self.genotype_flows['infections'][g]    += self.scale_flows(inds) # Add the count by genotype to the flow data
            self.age_flows['infections'][:]         += np.histogram(self.age[inds], bins=self.age_bins, weights=self.scale[inds])[0]

            # Create by-sex flows
            infs_female = self.scale_flows(hpu.true(self.is_female[inds]))
            infs_male = self.scale_flows(hpu.true(self.is_male[inds]))
            self.sex_flows['infections_by_sex'][0] += infs_female
            self.sex_flows['infections_by_sex'][1] += infs_male

        # Now use genotype-specific prognosis probabilities to determine what happens.
        # Only women can progress beyond infection.
        f_inds = hpu.itruei(self.is_female,inds)
        m_inds = hpu.itruei(self.is_male,inds)

        # Determine the duration of the HPV infection without any dysplasia
        if dur is None:
            this_dur = hpu.sample(**dur_precin, size=len(inds))  # Duration of infection without dysplasia in years
        else:
            if len(dur) != len(inds):
                errormsg = f'If supplying durations of infections, they must be the same length as inds: {len(dur)} vs. {len(inds)}.'
                raise ValueError(errormsg)
            this_dur    = dur

        # Set durations
        self.dur_infection[g, inds] = this_dur  # Set the duration of infection
        self.dur_precin[g, inds]    = this_dur  # Set the duration of infection without dysplasia

        # Compute disease progression for females
        if len(f_inds)>0:
            self.set_prognoses(f_inds, g, dt, hiv_pars=self.pars['hiv_pars']) # Set prognoses

        # Compute infection clearance for males
        if len(m_inds)>0:
            self.date_clearance[g, m_inds] = self.date_infectious[g, m_inds] + np.ceil(self.dur_infection[g, m_inds]/dt)  # Date they clear HPV infection (interpreted as the timestep on which they recover)

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
        else:
            errormsg = f'Cause of death must be one of "other", "cancer", or "emigration", not {cause}.'
            raise ValueError(errormsg)

        # Set states to false
        self.alive[inds] = False
        for state in hpd.total_stock_keys:
            self[state][:, inds] = False
        for state in hpd.other_stock_keys:
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

