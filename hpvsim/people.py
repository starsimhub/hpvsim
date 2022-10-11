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
    see ``hp.make_people()`` instead.

    Note that this class handles the mechanics of updating the actual people, while
    ``hp.BasePeople`` takes care of housekeeping (saving, loading, exporting, etc.).
    Please see the BasePeople class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as n_agents
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        pop_trend (dataframe): a dataframe of years and population sizes, if available
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::

        ppl1 = hp.People(2000)

        sim = hp.Sim()
        ppl2 = hp.People(sim.pars)
    '''

    def __init__(self, pars, strict=True, pop_trend=None, **kwargs):

        # Initialize the BasePeople, which also sets things up for filtering
        super().__init__(pars)
        
        # Handle pars and settings

        # Other initialization
        self.pop_trend = pop_trend
        self.init_contacts() # Initialize the contacts
        self.infection_log = [] # Record of infections - keys for ['source','target','date','layer']

        self.lag_bins = np.linspace(0,50,51)
        self.rship_lags = dict()
        for lkey in self.layer_keys():
            self.rship_lags[lkey] = np.zeros(len(self.lag_bins)-1, dtype=hpd.default_float)

        # Store age bins for standard population, used for age-standardized incidence calculations
        self.asr_bins = self.pars['standard_pop'][0, :] # Age bins of the standard population

        if strict:
            self.lock() # If strict is true, stop further keys from being set (does not affect attributes)

        # Store flows to be computed during simulation
        self.init_flows()

        # Although we have called init(), we still need to call initialize()
        self.initialized = False

        # Handle partners and contacts
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
            if strict:
                self.set(key, value)
            elif key in self._data:
                self[key][:] = value
            else:
                self[key] = value
        
        return


    def init_flows(self):
        ''' Initialize flows to be zero '''
        ng = self.pars['n_genotypes']
        df = hpd.default_float
        self.flows              = {f'{key}'         : np.zeros(ng, dtype=df) for key in hpd.flow_keys}
        for tf in hpd.total_flow_keys:
            self.flows[tf]      = 0
        self.total_flows        = {f'total_{key}'   : 0 for key in hpd.flow_keys}
        self.flows_by_sex       = {f'{key}'         : np.zeros(2, dtype=df) for key in hpd.by_sex_keys}
        self.demographic_flows  = {f'{key}'         : 0 for key in hpd.dem_keys}
        # self.intv_flows         = {f'{key}'         : 0 for key in hpd.intv_flow_keys}
        self.by_age_flows       = {'cancers_by_age' : np.zeros(len(self.asr_bins)-1)}

        return


    def increment_age(self):
        ''' Let people age by one timestep '''
        self.age[self.alive] += self.dt
        return


    def initialize(self, sim_pars=None, hiv_pars=None):
        ''' Perform initializations '''
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
            self.flows_by_sex['other_deaths_by_sex'][0] = deaths_female
            self.flows_by_sex['other_deaths_by_sex'][1] = deaths_male

            # Add births
            new_births = self.add_births(year=year)
            self.demographic_flows['births'] = new_births

            # Check migration
            migration = self.check_migration(year=year)
            self.demographic_flows['migration'] = migration

        # Perform updates that are genotype-specific
        ng = self.pars['n_genotypes']
        for g in range(ng):
            self.flows['cin1s'][g]              = self.check_cin1(g)
            self.flows['cin2s'][g]              = self.check_cin2(g)
            self.flows['cin3s'][g]              = self.check_cin3(g)
            new_cancers, cancers_by_age         = self.check_cancer(g)
            self.flows['cancers'][g]            += new_cancers
            self.by_age_flows['cancers_by_age'] += cancers_by_age
            self.flows['cins'][g]               = self.flows['cin1s'][g]+self.flows['cin2s'][g]+self.flows['cin3s'][g]
            self.check_clearance(g)

        # Perform updates that are not genotype specific
        self.flows['cancer_deaths'] = self.check_cancer_deaths()

        # Create total flows
        self.total_flows['total_cin1s'] = self.flows['cin1s'].sum()
        self.total_flows['total_cin2s'] = self.flows['cin2s'].sum()
        self.total_flows['total_cin3s'] = self.flows['cin3s'].sum()
        self.total_flows['total_cins']  = self.flows['cins'].sum()
        self.total_flows['total_cancers']  = self.flows['cancers'].sum()
        # self.total_flows['total_cancer_deaths']  = self.flows['cancer_deaths'].sum()

        # Before applying interventions or new infections, calculate the pool of susceptibles
        self.sus_pool = self.susceptible.all(axis=0) # True for people with no infection at the start of the timestep

        return


    #%% Methods for updating partnerships
    def dissolve_partnerships(self, t=None):
        ''' Dissolve partnerships '''

        n_dissolved = dict()

        for lno,lkey in enumerate(self.layer_keys()):
            layer = self.contacts[lkey]
            to_dissolve = (~self['alive'][layer['m']]) | (~self['alive'][layer['f']]) | ( (self.t*self.pars['dt']) > layer['end'])
            dissolved = layer.pop_inds(to_dissolve) # Remove them from the contacts list

            # Update current number of partners
            unique, counts = hpu.unique(np.concatenate([dissolved['f'],dissolved['m']]))
            self.current_partners[lno,unique] -= counts
            self.rship_end_dates[lno, unique] = self.t
            n_dissolved[lkey] = len(dissolved['f'])

        return n_dissolved # Return the number of dissolved partnerships by layer


    def create_parnterships(self, tind, mixing, layer_probs, cross_layer, dur_pship, acts, age_act_pars, pref_weight=100):
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


    def check_cin1(self, genotype):
        ''' Check for new progressions to CIN1 '''
        # Only include infectious females who haven't already cleared CIN1 or progressed to CIN2
        filters = self.infectious[genotype,:]*self.is_female*~(self.date_clearance[genotype,:]<=self.t)*(self.date_cin2[genotype,:]>=self.t)
        filter_inds = filters.nonzero()[0]
        inds = self.check_inds(self.cin1[genotype,:], self.date_cin1[genotype,:], filter_inds=filter_inds)
        self.cin1[genotype, inds] = True
        self.no_dysp[genotype, inds] = False
        return len(inds)


    def check_cin2(self, genotype):
        ''' Check for new progressions to CIN2 '''
        filter_inds = self.true_by_genotype('cin1', genotype)
        inds = self.check_inds(self.cin2[genotype,:], self.date_cin2[genotype,:], filter_inds=filter_inds)
        self.cin2[genotype, inds] = True
        self.cin1[genotype, inds] = False # No longer counted as CIN1
        return len(inds)


    def check_cin3(self, genotype):
        ''' Check for new progressions to CIN3 '''
        filter_inds = self.true_by_genotype('cin2', genotype)
        inds = self.check_inds(self.cin3[genotype,:], self.date_cin3[genotype,:], filter_inds=filter_inds)
        self.cin3[genotype, inds] = True
        self.cin2[genotype, inds] = False # No longer counted as CIN2
        return len(inds)


    def check_cancer(self, genotype):
        ''' Check for new progressions to cancer '''
        filter_inds = self.true_by_genotype('cin3', genotype)
        inds = self.check_inds(self.cancerous[genotype,:], self.date_cancerous[genotype,:], filter_inds=filter_inds)
        self.cancerous[genotype, inds] = True
        self.cin3[genotype, inds] = False # No longer counted as CIN3
        self.susceptible[:, inds] = False # No longer susceptible to any new genotypes
        self.date_clearance[:, inds] = np.nan

        # Calculations for age-standardized cancer incidence
        cases_by_age = 0
        if len(inds)>0:
            age_new_cases = self.age[inds] # Ages of new cases
            cases_by_age = np.histogram(age_new_cases, self.asr_bins)[0]

        return len(inds), cases_by_age


    def check_cancer_deaths(self):
        '''
        Check for new deaths from cancer
        '''
        filter_inds = self.true('cancerous')
        inds = self.check_inds(self.dead_cancer, self.date_dead_cancer, filter_inds=filter_inds)
        self.remove_people(inds, cause='cancer')

        # check which of these were detected by symptom or screening
        self.flows['detected_cancer_deaths'] += len(hpu.true(self.detected_cancer[inds]))

        return len(inds)


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
        self.peak_dysp[genotype, inds] = np.nan
        self.dysp_rate[genotype, inds] = np.nan
        self.prog_rate[genotype, inds] = np.nan

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
            
            hpu.set_HIV_prognoses(self, hiv_inds, year=year) # Set ART adherence for those with HIV

            for g in range(self.pars['n_genotypes']):
                nocin_inds = hpu.itruei((self.is_female & self.precin[g, :] & np.isnan(self.date_cin1[g, :])), hiv_inds) # Women with HIV who are scheduled to clear without dysplasia
                if len(nocin_inds): # Reevaluate whether these women will develop dysplasia
                    hpu.set_dysp_rates(self, nocin_inds, g, hiv_dysp_rate=self.pars['hiv_pars']['dysp_rate'])
                    hpu.set_dysp_status(self, nocin_inds, g, dt)

                cin_inds = hpu.itruei((self.is_female & self.infectious[g, :] & ~np.isnan(self.date_cin1[g, :])), hiv_inds) # Women with HIV who are scheduled to have dysplasia
                if len(cin_inds): # Reevaluate disease severity and progression speed for these women
                    hpu.set_severity(self, cin_inds, g, hiv_prog_rate=self.pars['hiv_pars']['prog_rate'])
                    hpu.set_cin_grades(self, cin_inds, g, dt)

        return len(hiv_inds)


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
        mx_f = death_pars[nearest_year]['f'][:,1]
        mx_m = death_pars[nearest_year]['m'][:,1]

        death_probs[self.is_female] = mx_f[age_inds[self.is_female]]
        death_probs[self.is_male] = mx_m[age_inds[self.is_male]]
        death_probs[self.age>100] = 1 # Just remove anyone >100
        death_probs[~self.alive] = 0

        # Get indices of people who die of other causes
        death_inds = hpu.true(hpu.binomial_arr(death_probs))
        deaths_female = len(hpu.true(self.is_female[death_inds]))
        deaths_male = len(hpu.true(self.is_male[death_inds]))
        other_deaths = self.remove_people(death_inds, cause='other') # Apply deaths

        return other_deaths, deaths_female, deaths_male


    def add_births(self, year=None, new_births=None):
        '''
        Add more people to the population

        Specify either the year from which to retrieve the birth rate, or the absolute number
        of new people to add. Must specify one or the other. People are added in-place to the
        current `People` instance
        '''

        assert (year is None) != (new_births is None), 'Must set either year or n_births, not both'

        if new_births is None:
            this_birth_rate = sc.smoothinterp(year, self.pars['birth_rates'][0], self.pars['birth_rates'][1], smoothness=0)[0]/1e3
            new_births = sc.randround(this_birth_rate*self.n_alive) # Crude births per 1000

        if new_births>0:
            # Generate other characteristics of the new people
            uids, sexes, debuts, partners = hppop.set_static(new_n=new_births, existing_n=len(self), pars=self.pars)

            # Grow the arrays
            self._grow(new_births)
            self['uid'][-new_births:] = uids
            self['age'][-new_births:] = 0
            self['sex'][-new_births:] = sexes
            self['debut'][-new_births:] = debuts
            self['partners'][:,-new_births:] = partners

        return new_births


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
            alive_inds = hpu.true(self.alive)
            n_alive = len(alive_inds) # Actual number of alive agents
            expected = np.interp(year, data_years, data_pop)*scale
            n_migrate = int(expected - n_alive)

            # Apply emigration
            if n_migrate < 0:
                inds = hpu.choose(n_alive, -n_migrate)
                migrate_inds = alive_inds[inds]
                self.remove_people(migrate_inds, cause='emigration') # Remove people

            # Apply immigration -- TODO, add age?
            elif n_migrate > 0:
                self.add_births(new_births=n_migrate)

        else:
            n_migrate = 0

        return n_migrate



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

        dt = self.pars['dt']

        # Deal with genotype parameters
        genotype_pars   = self.pars['genotype_pars']
        genotype_map    = self.pars['genotype_map']
        dur_precin      = genotype_pars[genotype_map[g]]['dur_precin']
        dysp_rate       = genotype_pars[genotype_map[g]]['dysp_rate']

        # Set date of infection and exposure
        base_t = self.t + offset if offset is not None else self.t
        self.date_infectious[g,inds] = base_t
        if layer != 'reactivation':
            self.date_exposed[g,inds] = base_t

        # Count reinfections and remove any previous dates
        self.flows['reinfections'][g]           += len((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        self.total_flows['total_reinfections']  += len((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        for key in ['date_clearance', 'date_cin1', 'date_cin2', 'date_cin3']:
            self[key][g, inds] = np.nan

        # Count reactivations and adjust latency status
        if layer == 'reactivation':
            self.flows['reactivations'][g] += len(inds)
            self.total_flows['total_reactivations'] += len(inds)
            self.latent[g, inds] = False # Adjust states -- no longer latent

        # Update states, genotype info, and flows
        self.susceptible[g, inds]   = False # no longer susceptible
        self.infectious[g, inds]    = True  # now infectious
        self.inactive[g, inds]      = False  # no longer inactive

        # Add to flow results. Note, we only count these infectious in the results if they happened at this timestep
        if offset is None:
            # Create overall flows
            self.total_flows['total_infections']    += len(inds) # Add the total count to the total flow data
            self.flows['infections'][g]             += len(inds) # Add the count by genotype to the flow data

            # Create by-sex flows
            infs_female = len(hpu.true(self.is_female[inds]))
            infs_male = len(hpu.true(self.is_male[inds]))
            self.flows_by_sex['total_infections_by_sex'][0] += infs_female
            self.flows_by_sex['total_infections_by_sex'][1] += infs_male

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
            hpu.set_prognoses(self, f_inds, g, dt, hiv_pars=self.pars['hiv_pars']) # Set prognoses

        # Compute infection clearance for males
        if len(m_inds)>0:
            self.date_clearance[g, m_inds] = self.date_infectious[g, m_inds] + np.ceil(self.dur_infection[g, m_inds]/dt)  # Date they clear HPV infection (interpreted as the timestep on which they recover)

        return len(inds) # For incrementing counters


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

        self.susceptible[:, inds] = False
        self.infectious[:, inds] = False
        self.inactive[:, inds] = False
        self.cin1[:, inds] = False
        self.cin2[:, inds] = False
        self.cin3[:, inds] = False
        self.cancerous[:, inds] = False
        self.alive[inds] = False

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

        return len(inds)


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

            sim = cv.Sim(pop_type='hybrid', verbose=0)
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

