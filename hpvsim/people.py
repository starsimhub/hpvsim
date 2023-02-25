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
        # self.dysp_keys = ['dysplasias', 'cancers']

        self.lag_bins = np.linspace(0,50,51)
        self.rship_lags = dict()
        for lkey in self.layer_keys():
            self.rship_lags[lkey] = np.zeros(len(self.lag_bins)-1, dtype=hpd.default_float)

        # Store age bins
        self.age_bins = self.pars['age_bins'] # Age bins for age results

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


    def initialize(self, sim_pars=None, hivsim=None):
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
        self.set_pars(pars=sim_pars, hivsim=hivsim) # Replace the saved parameters with this simulation's
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
            _ = self.hivsim.step(year=year)

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
            self.check_transformation(g) # check for new transformations, persistence, or clearance
            cases_by_age, cases = self.check_cancer(g)
            self.update_severity(g)
            self.check_clearance(g)
            self.flows['cancers'] += cases  # Increment flows (summed over all genotypes)
            self.genotype_flows['cancers'][g] = cases  # Store flows by genotype
            self.age_flows['cancers'] += cases_by_age  # Increment flows by age (summed over all genotypes)

        # Perform updates that are not genotype specific
        self.flows['cancer_deaths'] = self.check_cancer_deaths()

        # Before applying interventions or new infections, calculate the pool of susceptibles
        self.sus_pool = self.susceptible.all(axis=0) # True for people with no infection at the start of the timestep

        return

    
    #%% Disease progression methods
    def set_prognoses(self, inds, g, gpars, dt):
        '''
        Assigns prognoses for all infected women on day of infection.
        '''

        # Set length of infection, which is moderated by any prior cell-level immunity
        cell_imm = self.cell_imm[g, inds]
        self.dur_episomal[g, inds]  = hpu.sample(**gpars['dur_episomal'], size=len(inds))*(1-cell_imm)
        self.dur_infection[g, inds] = self.dur_episomal[g, inds] # For women who transform, the length of time that they have transformed infection is added to this later

        # Set infection severity and outcomes
        self.set_severity_pars(inds, g, gpars)
        self.set_severity(inds, g, gpars, dt)

        return


    def set_severity_pars(self, inds, g, gpars):
        '''
        Set disease severity properties
        '''
        self.sev_rate[g, inds] = hpu.sample(dist='normal_pos', par1=gpars['sev_rate'], par2=gpars['sev_rate_sd'], size=len(inds)) # Sample
        self.sev_infl[g, inds] = gpars['sev_infl'] * self.rel_sev_infl[inds] # Store points of inflection

        return


    def set_severity(self, inds, g, gpars, dt, set_sev=True):
        '''
        Set severity levels for individual women
        '''

        # Firstly, calculate the overall maximal severity that each woman will have
        dur_episomal = self.dur_episomal[g, inds]
        sev_infl = self.sev_infl[g, inds]
        sev_rate = self.sev_rate[g, inds]
        sevs = hpu.logf2(dur_episomal, sev_infl, sev_rate)
        if set_sev:
            self.sev[g, inds] = 0 # Severity starts at 0 on day 1 of infection

        # Now figure out probabilities of cellular transformations preceding cancer, based on this severity level
        transform_prob = gpars['transform_prob']
        n_extra = self.pars['ms_agent_ratio']
        cancer_scale = self.pars['pop_scale'] / n_extra
        if n_extra > 1:
            transform_probs = hpu.transform_prob(transform_prob, sevs)
            is_transform = hpu.binomial_arr(transform_probs)
            transform_inds = inds[is_transform]
            self.scale[transform_inds] = cancer_scale  # Shrink the weight of the original agents, but otherwise leave them the same

            # Create extra disease severity values
            full_size = (len(inds), n_extra)  # Main axis is indices, but include columns for multiscale agents
            extra_sev_rate = hpu.sample(dist='normal_pos', par1=gpars['sev_rate'], par2=gpars['sev_rate_sd'], size=full_size)
            extra_dur_episomal = hpu.sample(**gpars['dur_episomal'], size=full_size)
            extra_sev_infl = gpars['sev_infl'] * self.rel_sev_infl[inds]# This assumes none of the extra agents have HIV...
            extra_sev_infl = extra_sev_infl[:, None] * np.full(fill_value=1, shape=full_size)
            extra_sev = hpu.logf2(extra_dur_episomal, extra_sev_infl, extra_sev_rate)

            # Based on the severity values, determine transformation probabilities
            extra_transform_probs = hpu.transform_prob(transform_prob, extra_sev[:, 1:])
            extra_transform_bools = hpu.binomial_arr(extra_transform_probs)
            extra_transform_bools *= self.level0[inds, None]  # Don't allow existing cancer agents to make more cancer agents
            extra_transform_counts = extra_transform_bools.sum(axis=1)  # Find out how many new cancer cases we have
            n_new_agents = extra_transform_counts.sum()  # Total number of new agents
            if n_new_agents:  # If we have more than 0, proceed
                extra_source_lists = []
                for i, count in enumerate(extra_transform_counts):
                    ii = inds[i]
                    if count:  # At least 1 new cancer agent, plus person is not already a cancer agent
                        extra_source_lists.append([ii] * int(count))  # Duplicate the current index count times
                extra_source_inds = np.concatenate(extra_source_lists).flatten()  # Assemble the sources for these new agents
                n_new_agents = len(extra_source_inds)  # The same as above, *unless* a cancer agent tried to spawn more cancer agents

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

                # Add the new indices onto the existing vectors
                inds = np.append(inds, new_inds)
                is_transform = np.append(is_transform, np.full(len(new_inds), fill_value=True))
                new_sev_rate = extra_sev_rate[:,1:][extra_transform_bools]
                new_dur_episomal = extra_dur_episomal[:,1:][extra_transform_bools]
                new_sev_infl = extra_sev_infl[:,1:][extra_transform_bools]
                self.sev_infl[g, new_inds] = new_sev_infl
                self.sev_rate[g, new_inds] = new_sev_rate
                self.dur_episomal[g, new_inds] = new_dur_episomal
                self.dur_infection[g, new_inds] = new_dur_episomal
                self.date_infectious[g, new_inds] = self.t
                self.date_exposed[g, new_inds] = self.t
                dur_episomal = np.append(dur_episomal, new_dur_episomal)

        # First check indices, including new cancer agents
        transform_probs = np.zeros(len(inds))
        if n_extra > 1:
            transform_probs[is_transform] = 1  # Make sure inds that got assigned cancer above dont get stochastically missed
        else:
            transform_probs = hpu.transform_prob(transform_prob, hpu.logf2(self.dur_episomal[g,inds], sev_infl, self.sev_rate[g,inds]))

        # Set dates of cin1, 2, 3 for all women who get infected
        self.date_cin1[g, inds] = self.t + sc.randround(hpu.invlogf2(self.pars['clinical_cutoffs']['precin'], self.sev_infl[g, inds], self.sev_rate[g, inds])/dt)
        self.date_cin2[g, inds] = self.t + sc.randround(hpu.invlogf2(self.pars['clinical_cutoffs']['cin1'], self.sev_infl[g, inds], self.sev_rate[g, inds])/dt)
        self.date_cin3[g, inds] = self.t + sc.randround(hpu.invlogf2(self.pars['clinical_cutoffs']['cin2'], self.sev_infl[g, inds], self.sev_rate[g, inds])/dt)
        # self.date_carcinoma[g, inds] = self.t + sc.randround(hpu.invlogf2(self.pars['clinical_cutoffs']['cin3'], sev_infl, self.sev_rate[g, inds])/dt)

        # Now handle women who transform - need to adjust their length of infection and set more dates
        is_transform = hpu.binomial_arr(transform_probs)
        transform_inds = inds[is_transform]
        no_cancer_inds = inds[~is_transform]  # Indices of those who eventually heal lesion/clear infection
        time_to_clear = dur_episomal[~is_transform]
        self.date_clearance[g, no_cancer_inds] = np.fmax(self.date_clearance[g, no_cancer_inds],
                                                         self.date_exposed[g, no_cancer_inds] +
                                                         sc.randround(time_to_clear / dt))

        self.date_transformed[g, transform_inds] = self.t + sc.randround(dur_episomal[is_transform] / dt)
        dur_transformed = hpu.sample(**self.pars['dur_transformed'], size=len(transform_inds))
        self.date_cancerous[g, transform_inds] = self.date_transformed[g, transform_inds] + sc.randround(dur_transformed / dt)
        self.dur_infection[g, transform_inds] = self.dur_infection[g, transform_inds] + dur_transformed

        dur_cancer = hpu.sample(**self.pars['dur_cancer'], size=len(transform_inds))
        self.date_dead_cancer[transform_inds] = self.date_cancerous[g, transform_inds] + sc.randround(dur_cancer / dt)
        self.dur_cancer[g, transform_inds] = dur_cancer

        return

    def update_severity(self, genotype):
        ''' Update disease severity for women with infection'''
        gpars = self.pars['genotype_pars']
        gmap = self.pars['genotype_map']
        fg_inds = hpu.true(self.is_female & self.infectious[genotype,:])
        sev_rate = self.sev_rate[genotype, fg_inds]
        sev_infl = self.sev_infl[genotype, fg_inds]
        dur_episomal = (self.t - self.date_exposed[genotype, fg_inds]) * self.dt
        if (dur_episomal<0).any():
            errormsg = 'Durations cannot be less than zero.'
            raise ValueError(errormsg)

        self.sev[genotype, fg_inds] = hpu.logf2(dur_episomal, sev_infl, sev_rate)
        if (np.isnan(self.sev[genotype, fg_inds])).any():
            errormsg = 'Invalid severity values.'
            raise ValueError(errormsg)

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

    def check_transformation(self, genotype):
        ''' Check for new transformations, clearance or persistence '''
        # Only include infectious, episomal females who haven't already cleared infection
        filter_inds = self.true_by_genotype('episomal', genotype)
        inds = self.check_inds(self.transformed[genotype,:], self.date_transformed[genotype,:], filter_inds=filter_inds)
        self.transformed[genotype, inds] = True  # Now transformed, cannot clear
        self.date_clearance[genotype, inds] = np.nan  # Remove their clearance dates
        return


    def check_cancer(self, genotype):
        ''' Check for new progressions to cancer '''
        filter_inds = self.true('transformed')
        inds = self.check_inds(self.cancerous[genotype,:], self.date_cancerous[genotype,:], filter_inds=filter_inds)

        # Set infectious states
        self.susceptible[:, inds] = False  # No longer susceptible to any genotype
        self.infectious[:, inds] = False  # No longer counted as infectious with any genotype
        self.inactive[:,inds] = True  # If this person has any other infections from any other genotypes, set them to inactive

        self.date_clearance[:, inds] = np.nan  # Remove their clearance dates for all genotypes

        # Deal with dysplasia states and dates
        for g in range(self.ng):
            if g != genotype:
                self.date_cancerous[g, inds] = np.nan  # Remove their date of cancer for all genotypes but the one currently causing cancer
                self.date_cin1[g, inds] = np.nan
                self.date_cin2[g, inds] = np.nan
                self.date_cin3[g, inds] = np.nan
            else:
                date_cin2 = self.date_cin2[g,inds]
                change_inds = hpu.true(date_cin2 > self.t)
                self.date_cin2[g,inds[change_inds]] = np.nan

                date_cin3 = self.date_cin3[g,inds]
                change_inds = hpu.true(date_cin3 > self.t)
                self.date_cin3[g,inds[change_inds]] = np.nan

        # Set the properties related to cell changes and disease severity markers
        self.cancerous[genotype, inds] = True
        self.episomal[:, inds] = False  # No longer counted as episomal with any genotype
        self.transformed[:, inds] = False  # No longer counted as transformed with any genotype
        self.sev[:, inds] = np.nan # NOTE: setting this to nan means this people no longer counts as CIN1/2/3, since those categories are based on this value

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
            female_cleared_inds = np.intersect1d(cleared_inds, self.f_inds) # Only give natural immunity to females
            hpimm.update_peak_immunity(self, female_cleared_inds, imm_pars=self.pars, imm_source=genotype) # update immunity

        if len(latent_inds):
            self.susceptible[genotype, latent_inds] = False # should already be false
            self.infectious[genotype, latent_inds] = False
            self.inactive[genotype, latent_inds] = True
            self.date_clearance[genotype, latent_inds] = np.nan

        # Whether infection is controlled on not, clear all cell changes and severity markeres
        self.episomal[genotype, inds] = False
        self.transformed[genotype, inds] = False
        self.sev[genotype, inds] = np.nan
        self.sev_rate[genotype, inds] = np.nan
        self.date_cin1[genotype, inds] = np.nan
        self.date_cin2[genotype, inds] = np.nan
        self.date_cin3[genotype, inds] = np.nan
        self.date_carcinoma[genotype, inds] = np.nan

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
            ages = self.age[alive_inds].astype(int) # Return ages for everyone level 0 and alive
            count_ages = np.bincount(ages, minlength=age_dist_data.shape[0]) # Bin and count them
            expected = age_dist_data['PopTotal'].values*scale # Compute how many of each age we would expect in population
            difference = np.array([int(i) for i in (expected - count_ages)]) # Compute difference between expected and simulated for each age
            n_migrate = np.sum(difference) # Compute total migrations (in and out)
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
        for key in ['date_clearance', 'date_transformed']:
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
        self.episomal[g, inds]      = True  # now episomal
        self.inactive[g, inds]      = False  # no longer inactive

        # Add to flow results. Note, we only count these infectious in the results if they happened at this timestep
        if layer != 'seed_infection':
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

        # Compute disease progression for females
        if len(f_inds)>0:
            gpars = self.pars['genotype_pars'][self.pars['genotype_map'][g]]
            self.set_prognoses(f_inds, g, gpars, dt)

        # Compute infection clearance for males
        if len(m_inds)>0:
            dur_infection = hpu.sample(**self.pars['dur_infection_male'], size=len(m_inds))
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
            pass # handled by hivsim
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

