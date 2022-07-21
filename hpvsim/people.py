'''
Defines the People class and functions associated with making people and handling
the transitions between states (e.g., from susceptible to infected).
'''

#%% Imports
import numpy as np
import sciris as sc
from collections import defaultdict
from . import version as hpv
from . import utils as hpu
from . import defaults as hpd
from . import base as hpb
from . import population as hppop
from . import parameters as hppar
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
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as pop_size
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::

        ppl1 = hp.People(2000)

        sim = hp.Sim()
        ppl2 = hp.People(sim.pars)
    '''

    def __init__(self, pars, strict=True, **kwargs):
        
        # Initialize the BasePeople, which also sets things up for filtering  
        super().__init__()

        # Handle pars and settings
        self.set_pars(pars)
        self.version = hpv.__version__ # Store version info

        # Other initialization
        self.t = 0 # Keep current simulation time
        self._lock = False # Prevent further modification of keys
        self.meta = hpd.PeopleMeta() # Store list of keys and dtypes
        self.contacts = None
        self.init_contacts() # Initialize the contacts
        self.infection_log = [] # Record of infections - keys for ['source','target','date','layer']

        # Set person properties -- all floats except for UID
        for key in self.meta.person:
            if key == 'uid':
                self[key] = np.arange(self.pars['pop_size'], dtype=hpd.default_int)
            elif key in ['partners', 'current_partners']:
                self[key] = np.full((self.pars['n_partner_types'], self.pars['pop_size']), np.nan, dtype=hpd.default_float)
            else:
                self[key] = np.full(self.pars['pop_size'], np.nan, dtype=hpd.default_float)

        # Set health states -- only susceptible is true by default -- booleans except exposed by genotype which should return the genotype that ind is exposed to
        for key in self.meta.states:
            if key in ['dead_other', 'vaccinated', 'screened', 'treated', 'diagnosed']: # ALl false at the beginning
                self[key] = np.full(self.pars['pop_size'], False, dtype=bool)
            elif key == 'alive':  # All true at the beginning
                self[key] = np.full(self.pars['pop_size'], True, dtype=bool)
            elif key == 'susceptible':
                self[key] = np.full((self.pars['n_genotypes'], self.pars['pop_size']), True, dtype=bool)
            else:
                self[key] = np.full((self.pars['n_genotypes'], self.pars['pop_size']), False, dtype=bool)

        # Set dates and durations -- both floats
        for key in self.meta.dates + self.meta.durs:
            if key in ['date_dead_other', 'date_vaccinated','date_screened', 'date_next_screen', 'date_treated']:
                self[key] = np.full(self.pars['pop_size'], np.nan, dtype=hpd.default_float)
            else:
                self[key] = np.full((self.pars['n_genotypes'], self.pars['pop_size']), np.nan, dtype=hpd.default_float)

        # Set genotype states, which store info about which genotype a person is exposed to
        for key in self.meta.imm_states:  # Everyone starts out with no immunity; TODO, reconsider this
            if key == 't_imm_event':
                self[key] = np.zeros((self.pars['n_imm_sources'], self.pars['pop_size']), dtype=hpd.default_int)
            else:
                self[key] = np.zeros((self.pars['n_imm_sources'], self.pars['pop_size']), dtype=hpd.default_float)
        for key in self.meta.intv_states:
            self[key] = np.zeros(self.pars['pop_size'], dtype=hpd.default_int)

        # Store the dtypes used in a flat dict
        self._dtypes = {key:self[key].dtype for key in self.keys()} # Assign all to float by default
        if strict:
            self.lock() # If strict is true, stop further keys from being set (does not affect attributes)

        # Store flows to be computed during simulation
        self.init_flows()

        # Although we have called init(), we still need to call initialize()
        self.initialized = False

        # Handle partners and contacts
        if 'partners' in kwargs:
            self.partners = kwargs.pop('partners') # Store the desired concurrency
        if 'current_partners' in kwargs:
            self.current_partners = kwargs.pop('current_partners') # Store current actual number - updated each step though
        if 'contacts' in kwargs:
            self.add_contacts(kwargs.pop('contacts')) # Also updated each step

        # Handle all other values, e.g. age
        for key,value in kwargs.items():
            if strict:
                self.set(key, value)
            else:
                self[key] = value

        if 'death_age' in kwargs:
            self.date_dead_other = ((self.death_age-self.age)/self.pars['dt']).astype(int)

        return


    def init_flows(self):
        ''' Initialize flows to be zero '''
        ng = self.pars['n_genotypes']
        df = hpd.default_float
        self.flows              = {f'{key}'                 : np.zeros(ng, dtype=df) for key in hpd.flow_keys}
        self.total_flows        = {f'total_{key}'           : 0 for key in hpd.flow_keys}
        self.flows_by_sex       = {f'{key}'                 : np.zeros(2, dtype=df) for key in hpd.by_sex_keys}
        self.demographic_flows  = {f'{key}'                 : 0 for key in hpd.dem_keys}
        return


    def increment_age(self):
        ''' Let people age by one timestep '''
        self.age += self.dt
        return


    def initialize(self, sim_pars=None):
        ''' Perform initializations '''
        self.validate(sim_pars=sim_pars) # First, check that essential-to-match parameters match
        self.set_pars(sim_pars) # Replace the saved parameters with this simulation's
        self.initialized = True
        return


    def update_states_pre(self, t, year=None, resfreq=None):
        ''' Perform all state updates at the current timestep '''

        # Initialize
        self.t = t
        self.dt = self.pars['dt']
        self.resfreq = resfreq if resfreq is not None else 1

        # Let people age by one time step
        self.increment_age()

        # Perform updates that are not genotype-specific
        if t%self.resfreq==0:

            self.init_flows()  # Only reinitialize flows to zero every nth step, where n is the requested result frequency

            # Apply death rates from other causes
            other_deaths, deaths_female, deaths_male    = self.apply_death_rates(year=year)
            self.demographic_flows['other_deaths']      = other_deaths
            self.flows_by_sex['other_deaths_by_sex'][0] = deaths_female
            self.flows_by_sex['other_deaths_by_sex'][1] = deaths_male

            # Add births
            new_births, new_people = self.add_births(year=year)
            self.demographic_flows['births'] = new_births
            self.addtoself(new_people) # New births are added to the population

        # Perform updates that are genotype-specific
        ng = self.pars['n_genotypes']
        for g in range(ng):
            self.flows['cin1s'][g]          += self.check_cin1(g)
            self.flows['cin2s'][g]          += self.check_cin2(g)
            self.flows['cin3s'][g]          += self.check_cin3(g)
            if t%self.resfreq==0:
                self.flows['cins'][g]       += self.flows['cin1s'][g]+self.flows['cin2s'][g]+self.flows['cin3s'][g]
            self.flows['cancers'][g]        += self.check_cancer(g)
            self.flows['cancer_deaths'][g]  += self.check_cancer_deaths(g)
            self.check_clearance(g)

        # Create total flows
        self.total_flows['total_cin1s']     += self.flows['cin1s'].sum()
        self.total_flows['total_cin2s']     += self.flows['cin2s'].sum()
        self.total_flows['total_cin3s']     += self.flows['cin3s'].sum()
        self.total_flows['total_cins']      += self.flows['cins'].sum()
        self.total_flows['total_cancers']   += self.flows['cancers'].sum()
        self.total_flows['total_cancer_deaths']   += self.flows['cancer_deaths'].sum()

        # Before applying interventions or new infections, calculate the pool of susceptibles
        self.sus_pool = self.susceptible.nonzero()

        return


    #%% Methods for updating partnerships
    def dissolve_partnerships(self, t=None):
        ''' Dissolve partnerships '''

        n_dissolved = dict()

        for lno,lkey in enumerate(self.layer_keys()):
            dissolve_inds = hpu.true(self.t*self.pars['dt']>self.contacts[lkey]['end']) # Get the partnerships due to end
            dissolved = self.contacts[lkey].pop_inds(dissolve_inds) # Remove them from the contacts list

            # Update current number of partners
            unique, counts = hpu.unique(np.concatenate([dissolved['f'],dissolved['m']]))
            self.current_partners[lno,unique] -= counts
            n_dissolved[lkey] = len(dissolve_inds)

        return n_dissolved # Return the number of dissolved partnerships by layer


    def create_partnerships(self, t=None, n_new=None, pref_weight=100, scale_factor=None):
        ''' Create new partnerships '''

        new_pships = dict()
        mixing = self.pars['mixing']
        layer_probs = self.pars['layer_probs']

        for lno,lkey in enumerate(self.layer_keys()):

            # Intialize storage
            new_pships[lkey] = dict()
            new_pship_probs = np.ones(len(self)) # Begin by assigning everyone equal probability of forming a new relationship. This will be used for males, and for females if no layer_probs are provided
            new_pship_probs[~self.is_active] *= 0  # Blank out people not yet active
            underpartnered = (self.current_partners[lno, :] < self.partners[lno,:]) + (np.isnan(self.current_partners[lno,:])*self.is_active)  # Whether or not people are underpartnered in this layer, also selects newly active people
            new_pship_probs[underpartnered] *= pref_weight  # Increase weight for those who are underpartnerned

            if layer_probs is not None: # If layer probabilities have been provided, we use them to select females by age
                bins = layer_probs[lkey][0, :] # Extract age bins
                other_layers = np.delete(np.arange(len(self.layer_keys())),lno) # Indices of all other layers but this one
                already_partnered = (self.current_partners[other_layers,:]>0).sum(axis=0) # Whether or not people already partnered in other layers
                f_eligible = self.is_active * self.is_female * ~already_partnered * underpartnered # Females who are underpartnered in this layer and aren't already partnered in other layers are eligible to be selected
                f_eligible_inds = hpu.true(f_eligible)
                age_bins_f = np.digitize(self.age[f_eligible_inds], bins=bins) - 1  # Age bins of eligible females
                bin_range_f = np.unique(age_bins_f)  # Range of bins
                new_pship_inds_f = []  # Initialize new female contact list
                for ab in bin_range_f:  # Loop over age bins
                    these_f_contacts = hpu.binomial_filter(layer_probs[lkey][1][ab], f_eligible_inds[age_bins_f == ab])  # Select females according to their participation rate in this layer
                    new_pship_inds_f += these_f_contacts.tolist()
                new_pship_inds_f = np.array(new_pship_inds_f)

            else: # No layer probabilities have been provided, so we just select a specified number of new relationships for females
                this_n_new = int(n_new[lkey] * scale_factor)
                # Draw female partners
                new_pship_inds_f = hpu.choose_w(probs=new_pship_probs*self.is_female, n=this_n_new, unique=True)
                sorted_f_inds = self.age[new_pship_inds_f].argsort()
                new_pship_inds_f = new_pship_inds_f[sorted_f_inds]

            if len(new_pship_inds_f)>0:

                # Draw male partners based on mixing matrices if provided
                if mixing is not None:
                    bins = mixing[lkey][:, 0]
                    m_active_inds = hpu.true(self.is_active*self.is_male) # Males eligible to be selected
                    age_bins_f = np.digitize(self.age[new_pship_inds_f], bins=bins) - 1 # Age bins of females that are entering new relationships
                    age_bins_m = np.digitize(self.age[m_active_inds], bins=bins) - 1 # Age bins of eligible males
                    bin_range_f, males_needed = np.unique(age_bins_f, return_counts=True)  # For each female age bin, how many females need partners?
                    weighting = new_pship_probs*self.is_male # Weight males according to how underpartnered they are so they're ready to be selected
                    new_pship_inds_m = []  # Initialize the male contact list
                    for ab,nm in zip(bin_range_f, males_needed):  # Loop through the age bins of females and the number of males needed for each
                        male_dist = mixing[lkey][:, ab+1]  # Get the distribution of ages of the male partners of females of this age
                        this_weighting = weighting[m_active_inds] * male_dist[age_bins_m]  # Weight males according to the age preferences of females of this age
                        selected_males = hpu.choose_w(this_weighting, nm, unique=False)  # Select males
                        new_pship_inds_m += m_active_inds[selected_males].tolist()  # Extract the indices of the selected males and add them to the contact list
                    new_pship_inds_m = np.array(new_pship_inds_m)

                # Otherwise, do rough age assortativity
                else:
                    new_pship_inds_m  = hpu.choose_w(probs=new_pship_probs*self.is_male, n=this_n_new, unique=True)
                    sorted_m_inds = self.age[new_pship_inds_m].argsort()
                    new_pship_inds_m = new_pship_inds_m[sorted_m_inds]

                # Increment the number of current partners
                new_pship_inds      = np.concatenate([new_pship_inds_f, new_pship_inds_m])
                self.current_partners[lno,new_pship_inds] += 1

                # Handle acts: these must be scaled according to age
                acts = hpu.sample(**self['pars']['acts'][lkey], size=len(new_pship_inds_f))
                kwargs = dict(acts=acts,
                              age_act_pars=self['pars']['age_act_pars'][lkey],
                              age_f=self.age[new_pship_inds_f],
                              age_m=self.age[new_pship_inds_m],
                              debut_f=self.debut[new_pship_inds_f],
                              debut_m=self.debut[new_pship_inds_m]
                              )
                scaled_acts = hppop.age_scale_acts(**kwargs)
                keep_inds = scaled_acts > 0  # Discard partnerships with zero acts (e.g. because they are "post-retirement")
                f = new_pship_inds_f[keep_inds]
                m = new_pship_inds_m[keep_inds]
                scaled_acts = scaled_acts[keep_inds]
                final_n_new = len(f)

                # Add everything to a contacts dictionary
                new_pships[lkey]['f']       = f
                new_pships[lkey]['m']       = m
                new_pships[lkey]['dur']     = hpu.sample(**self['pars']['dur_pship'][lkey], size=final_n_new)
                new_pships[lkey]['start']   = np.array([t*self['pars']['dt']]*final_n_new, dtype=hpd.default_float)
                new_pships[lkey]['end']     = new_pships[lkey]['start'] + new_pships[lkey]['dur']
                new_pships[lkey]['acts']    = scaled_acts
                new_pships[lkey]['age_f']   = self.age[f]
                new_pships[lkey]['age_m']   = self.age[m]

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
        self.hpv[genotype, inds] = False
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
        self.susceptible[:, inds] = False
        self.date_clearance[:, inds] = np.nan
        return len(inds)


    def check_cancer_deaths(self, genotype):
        '''
        Check for new deaths from cancer
        '''
        filter_inds = self.true_by_genotype('cancerous', genotype)
        inds = self.check_inds(self.dead_cancer[genotype,:], self.date_dead_cancer[genotype,:], filter_inds=filter_inds)
        self.make_die(inds, genotype=genotype, cause='cancer')
        return len(inds)


    def check_death(self):
        '''
        Check for new deaths
        '''
        filter_inds = self.true('alive')
        inds = self.check_inds(self.dead_other, self.date_dead_other, filter_inds=filter_inds)
        deaths_female = len(hpu.true(self.is_female[inds]))
        deaths_male = len(hpu.true(self.is_male[inds]))
        # Apply deaths
        new_other_deaths = self.make_die(inds, cause='other')
        return new_other_deaths, deaths_female, deaths_male


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
        self.susceptible[genotype, cleared_inds] = True
        self.infectious[genotype, inds] = False
        self.hpv[genotype, inds]  = False
        self.cin1[genotype, inds] = False
        self.cin2[genotype, inds] = False
        self.cin3[genotype, inds] = False

        if len(latent_inds):
            self.latent[genotype, latent_inds] = True
            self.date_clearance[genotype, latent_inds] = np.nan

        # Update immunity
        hpimm.update_peak_immunity(self, cleared_inds, imm_pars=self.pars, imm_source=genotype)

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
        death_probs = np.full(len(self), np.nan, dtype=hpd.default_float)
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]
        mx_f = death_pars[nearest_year]['f'][:,1]
        mx_m = death_pars[nearest_year]['m'][:,1]

        death_probs[self.is_female_alive] = mx_f[age_inds[self.is_female_alive]]
        death_probs[self.is_male_alive] = mx_f[age_inds[self.is_male_alive]]
        death_probs[(self.age>100) & self.alive] = 1 # Just remove anyone >100

        # Get indices of people who die of other causes
        death_inds = hpu.true(hpu.binomial_arr(death_probs))
        deaths_female = len(hpu.true(self.is_female[death_inds]))
        deaths_male = len(hpu.true(self.is_male[death_inds]))
        other_deaths = self.make_die(death_inds, cause='other') # Apply deaths

        return other_deaths, deaths_female, deaths_male


    def add_births(self, year=None):
        ''' Method to add births '''

        this_birth_rate = sc.smoothinterp(year, self.pars['birth_rates'][0], self.pars['birth_rates'][1])[0]/1e3
        new_births = round(this_birth_rate*self.n_alive) # Crude births per 1000

        # Generate other characteristics of the new people
        uids, sexes, debuts, partners = hppop.set_static(new_n=new_births, existing_n=len(self), pars=self.pars)
        pars = {
            'dt': self['dt'],
            'pop_size': new_births,
            'n_genotypes': self.pars['n_genotypes'],
            'n_imm_sources': self.pars['n_imm_sources'],
            'n_partner_types': self.pars['n_partner_types']
        }
        new_people = People(pars=pars, uid=uids, age=np.zeros(new_births), sex=sexes, debut=debuts, partners=partners, strict=False)

        return new_births, new_people


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
        for key in self.meta.imm_by_source_states:
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
        durpars         = genotype_pars[genotype_map[g]]['dur']

        # Set all dates
        base_t = self.t + offset if offset is not None else self.t
        self.date_infectious[g,inds] = base_t

        # Count reinfections
        self.flows['reinfections'][g]           += len((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        self.total_flows['total_reinfections']  += len((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        for key in ['date_clearance']:
            self[key][g, inds] = np.nan

        # Count reactivations
        if layer == 'reactivation':
            self.flows['reactivations'][g] += len(inds)
            self.total_flows['total_reactivations'] += len(inds)
            self.latent[g, inds] = False # Adjust states -- no longer latent

        # Update states, genotype info, and flows
        self.susceptible[g, inds]   = False # Adjust states - set susceptible to false
        self.infectious[g, inds]    = True # Adjust states - set infectious to true
        self.hpv[g, inds]           = True # Adjust states - set hpv to true

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
        f_inds = self.is_female[inds].nonzero()[-1]
        m_inds = self.is_male[inds].nonzero()[-1]

        # Determine the duration of the HPV infection without any dysplasia
        if dur is None:
            this_dur = hpu.sample(**durpars['none'], size=len(inds))  # Duration of infection without dysplasia in years
            this_dur_f = self.dur_hpv[g, inds[self.is_female[inds]]]
        else:
            if len(dur) != len(inds):
                errormsg = f'If supplying durations of infections, they must be the same length as inds: {len(dur)} vs. {len(inds)}.'
                raise ValueError(errormsg)
            this_dur    = dur
            this_dur_f  = dur[self.is_female[inds]]

        self.dur_hpv[g, inds] = this_dur  # Set the duration of infection
        self.dur_disease[g, inds] = this_dur  # Set the initial duration of disease as the length of the period without dysplasia - this is then extended for those who progress

        # Compute disease progression for females and skip for makes; males are updated below
        if len(f_inds)>0:

            fg_inds = inds[self.is_female[inds]] # Subset the indices so we're only looking at females with this genotype
            hpu.set_prognoses(self, fg_inds, g, this_dur_f)

        if len(m_inds)>0:
            self.date_clearance[g, inds[m_inds]] = self.date_infectious[g, inds[m_inds]] + np.ceil(self.dur_hpv[g, inds[m_inds]]/dt)  # Date they clear HPV infection (interpreted as the timestep on which they recover)

        return len(inds) # For incrementing counters


    def make_die(self, inds, genotype=None, cause=None):
        ''' Make people die of all other causes (background mortality) '''

        if cause=='other':
            self.dead_other[inds] = True
        elif cause=='cancer':
            self.dead_cancer[genotype, inds] = True
        else:
            errormsg = f'Cause of death must be one of "other" or "cancer", not {cause}.'
            raise ValueError(errormsg)

        self.susceptible[:, inds] = False
        self.infectious[:, inds] = False
        self.cin1[:, inds] = False
        self.cin2[:, inds] = False
        self.cin3[:, inds] = False
        self.cancerous[:, inds] = False
        self.alive[inds] = False

        # Remove dead people from contact network by setting the end date of any partnership they're in to now
        for contacts in self.contacts.values():
            m_inds = hpu.findinds(contacts['m'], inds)
            f_inds = hpu.findinds(contacts['f'], inds)
            pships_to_end = contacts.pop_inds(np.concatenate([f_inds, m_inds]))
            pships_to_end['end']*=0+self.t # Reset end date to now
            contacts.append(pships_to_end)
            
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

