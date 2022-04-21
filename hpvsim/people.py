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
from . import plotting as hpplt
from . import immunity as hpi


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

        # Handle pars and population size
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
            else:
                self[key] = np.full(self.pars['pop_size'], np.nan, dtype=hpd.default_float)

        # Set health states -- only susceptible is true by default -- booleans except exposed by genotype which should return the genotype that ind is exposed to
        for key in self.meta.states:
            val = (key in ['susceptible', 'naive']) # Default value is True for susceptible and naive, false otherwise
            self[key] = np.full(self.pars['pop_size'], val, dtype=bool)

        # Set dates and durations -- both floats
        for key in self.meta.dates + self.meta.durs:
            self[key] = np.full(self.pars['pop_size'], np.nan, dtype=hpd.default_float)

        # Set genotype states, which store info about which genotype a person is exposed to
        for key in self.meta.genotype_states:
            self[key] = np.full(self.pars['pop_size'], np.nan, dtype=hpd.default_float)
        for key in self.meta.by_genotype_states:
            self[key] = np.full((self.pars['n_genotypes'], self.pars['pop_size']), False, dtype=bool)
        for key in self.meta.imm_states:  # Everyone starts out with no immunity
            self[key] = np.zeros((self.pars['n_genotypes'], self.pars['pop_size']), dtype=hpd.default_float)
        for key in self.meta.imm_by_source_states:  # Everyone starts out with no immunity
            self[key] = np.zeros((self.pars['n_genotypes'], self.pars['pop_size']), dtype=hpd.default_float)

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

        return


    def init_flows(self):
        ''' Initialize flows to be zero '''
        self.flows = {key:0 for key in hpd.new_result_flows}
        self.flows_genotype = {}
        for key in hpd.new_result_flows_by_genotype:
            self.flows_genotype[key] = np.zeros(self.pars['n_genotypes'], dtype=hpd.default_float)
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


    def update_states_pre(self, t):
        ''' Perform all state updates at the current timestep '''

        # Initialize
        self.t = t
        self.dt = self.pars['dt']
        self.is_inf = self.true('infectious') # For storing the interim values since used in every subsequent calculation

        # Perform updates
        self.init_flows() # Initialize flows for this timestep to zero
        self.increment_age() # Let people age by one time step
        self.flows['new_other_deaths'] += self.apply_death_rates() # Apply death rates 
        self.flows['new_births'], new_people = self.add_births() # Add births
        self.flows['new_recoveries'] += self.check_recovery() 
        # Lots more to be added here

        return new_people


    #%% Methods for updating partnerships
    def dissolve_partnerships(self, t=None):
        ''' Dissolve partnerships '''

        n_dissolved = dict()

        for lkey in self.layer_keys():
            dissolve_inds = hpu.true(self.t*self.pars['dt']>self.contacts[lkey]['end']) # Get the partnerships due to end
            dissolved = self.contacts[lkey].pop_inds(dissolve_inds) # Remove them from the contacts list

            # Update current number of partners
            unique, counts = np.unique(np.concatenate([dissolved['f'],dissolved['m']]), return_counts=True)
            self.current_partners[lkey][unique] -= counts
            n_dissolved[lkey] = len(dissolve_inds)

        return n_dissolved # Return the number of dissolved partnerships by layer


    def create_partnerships(self, t=None, n_new=None, pref_weight=100):
        ''' Create new partnerships '''

        new_pships = dict()
        for lkey in self.layer_keys():
            new_pships[lkey] = dict()
            
            # Define probabilities of entering new partnerships
            new_pship_probs                     = np.ones(len(self)) # Begin by assigning everyone equal probability of forming a new relationship
            new_pship_probs[~self.is_active]    *= 0 # Blank out people not yet active
            underpartnered                      = hpu.true(self.current_partners[lkey]<self.partners[lkey]) # Indices of those who have fewer partners than desired
            new_pship_probs[underpartnered]     *= pref_weight # Increase weight for those who are underpartnerned

            # Draw female and male partners separately
            new_pship_inds_f    = hpu.choose_w(probs=new_pship_probs*self.is_female, n=n_new[lkey], unique=True)
            new_pship_inds_m    = hpu.choose_w(probs=new_pship_probs*self.is_male, n=n_new[lkey], unique=True)
            new_pship_inds      = np.concatenate([new_pship_inds_f, new_pship_inds_m])
            self.current_partners[lkey][new_pship_inds] += 1

            # Add everything to a contacts dictionary
            new_pships[lkey]['f']       = new_pship_inds_f
            new_pships[lkey]['m']       = new_pship_inds_m
            new_pships[lkey]['dur']     = hpu.sample(**self['pars']['dur_pship'][lkey], size=n_new[lkey])
            new_pships[lkey]['start']   = np.array([t*self['pars']['dt']]*n_new[lkey],dtype=hpd.default_float)
            new_pships[lkey]['end']     = new_pships[lkey]['start'] + new_pships[lkey]['dur']
            new_pships[lkey]['acts']    = hpu.sample(**self['pars']['acts'][lkey], size=n_new[lkey]) # Acts per year for this pair, assumed constant over the duration of the partnership (TODO: EMOD uses a decay factor for this, consider?)

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


    def check_recovery(self, inds=None, filter_inds='is_inf'):
        '''
        Check for recovery.

        More complex than other functions to allow for recovery to be manually imposed
        for a specified set of indices.
        '''

        # Handle more flexible options for setting indices
        if filter_inds == 'is_inf':
            filter_inds = self.is_inf
        if inds is None:
            inds = self.check_inds(self.recovered, self.date_recovered, filter_inds=filter_inds)

        # Now reset all disease states
        self.infectious[inds]       = False
        self.recovered[inds]        = True
        self.recovered_genotype[inds] = self.infectious_genotype[inds]
        self.susceptible[inds] = True
        self.infectious_genotype[inds] = np.nan
        self.infectious_by_genotype[:, inds] = False

        return len(inds)


    def apply_death_rates(self):
        '''
        Apply death rates to remove people from the population
        NB people are not actually removed to avoid issues with indices
        '''

        # Get age-dependent death rates. TODO: careful with rates vs probabilities!
        age_inds = np.digitize(self.age,self.pars['death_rates']['f'][:,0])-1
        death_probs = np.full(len(self), np.nan, dtype=hpd.default_float)
        death_probs[self.f_inds] = self.pars['death_rates']['f'][age_inds[self.f_inds],2]*self.dt
        death_probs[self.m_inds] = self.pars['death_rates']['m'][age_inds[self.m_inds],2]*self.dt

        # Get indices of people who die of other causes, removing anyone already dead
        death_inds = hpu.true(hpu.binomial_arr(death_probs))
        already_dead = self.other_dead[death_inds]
        death_inds = death_inds[~already_dead]  # Unique indices in deaths that are not already dead

        # Apply deaths
        new_other_deaths = self.make_die_other(death_inds)
        return new_other_deaths


    def add_births(self):
        ''' Method to add births '''
        this_birth_rate = sc.smoothinterp(self.t+self.pars['start'], self.pars['birth_rates'][0], self.pars['birth_rates'][1])*self.dt
        new_births = round(this_birth_rate[0]*len(self)/1000) # Crude births per 1000

        # Generate other characteristics of the new people
        uids, sexes, debuts, partners = hppop.set_static(new_n=new_births, existing_n=len(self), pars=self.pars)
        pars = {
            'pop_size': new_births,
            'n_genotypes': self.pars['n_genotypes']
        }
        new_people = People(pars=pars, uid=uids, age=np.zeros(new_births), sex=sexes, debut=debuts, partners=partners, strict=False)

        return new_births, new_people


    #%% Methods to make events occur (death, infection, others TBC)
    def make_naive(self, inds, reset_vx=False):
        '''
        Make a set of people naive. This is used during dynamic resampling.

        Args:
            inds (array): list of people to make naive
            reset_vx (bool): whether to reset vaccine-derived immunity
        '''
        for key in self.meta.states:
            if key in ['susceptible', 'naive']:
                self[key][inds] = True
            else:
                if (key != 'vaccinated') or reset_vx: # Don't necessarily reset vaccination
                    self[key][inds] = False

        # Reset genotype states
        for key in self.meta.genotype_states:
            self[key][inds] = np.nan
        for key in self.meta.by_genotype_states:
            self[key][:, inds] = False

        # Reset immunity
        non_vx_inds = inds if reset_vx else inds[~self['vaccinated'][inds]]
        for key in self.meta.imm_by_source_states:
            self[key][:, non_vx_inds] = 0

        # Reset dates
        for key in self.meta.dates + self.meta.durs:
            if (key != 'date_vaccinated') or reset_vx: # Don't necessarily reset vaccination
                self[key][inds] = np.nan

        return


    def make_nonnaive(self, inds, set_recovered=False, date_recovered=0):
        '''
        Make a set of people non-naive.

        This can be done either by setting only susceptible and naive states,
        or else by setting them as if they have been infected and recovered.
        '''
        self.make_naive(inds) # First make them naive and reset all other states

        # Make them non-naive
        for key in ['susceptible', 'naive']:
            self[key][inds] = False

        if set_recovered:
            self.date_recovered[inds] = date_recovered # Reset date recovered
            self.check_recovered(inds=inds, filter_inds=None) # Set recovered

        return



    def infect(self, inds, source=None, layer=None, genotype=0):
        '''
        Infect people and determine their eventual outcomes.
        Method also deduplicates input arrays in case one agent is infected many times
        and stores who infected whom in infection_log list.

        Args:
            inds     (array): array of people to infect
            source   (array): source indices of the people who transmitted this infection (None if an importation or seed infection)
            layer    (str):   contact layer this infection was transmitted on
            genotype (int):   the genotype people are being infected by

        Returns:
            count (int): number of people infected
        '''

        if len(inds) == 0:
            return 0

        # Remove duplicates
        inds, unique = np.unique(inds, return_index=True)
        if source is not None:
            source = source[unique]

        # Keep only susceptibles
        keep = self.susceptible[inds] # Unique indices in inds and source that are also susceptible
        inds = inds[keep]
        if source is not None:
            source = source[keep]

        genotype_label = self.pars['genotype_map'][genotype]

        n_infections = len(inds)
        durpars      = self.pars['dur']

        # Update states, genotype info, and flows
        self.susceptible[inds]  = False
        self.naive[inds]        = False
        self.infectious[inds]   = True
        self.infectious_genotype[inds] = genotype
        self.infectious_by_genotype[genotype, inds] = True
        self.recovered[inds]    = False
        self.flows['new_infections']   += len(inds)
        self.flows_genotype['new_infections_by_genotype'][genotype] += len(inds)

        # # Record transmissions. TODO: this works, but slows does runtime by a LOT
        # for i, target in enumerate(inds):
        #     entry = dict(source=source[i] if source is not None else None, target=target, date=self.t, layer=layer, genotype=genotype_label)
        #     self.infection_log.append(entry)

        # Set the dates of infection and recovery -- for now, just assume everyone recovers
        dt = self.pars['dt']
        self.date_infectious[inds] = self.t
        dur_inf2rec = hpu.sample(**durpars['inf2rec'], size=len(inds)) # Duration of infection in YEARS
        self.date_recovered[inds] = self.date_infectious[inds] + np.ceil(dur_inf2rec/dt)  # Date they recover (interpreted as the timestep on which they recover)
        self.dur_disease[inds] = dur_inf2rec

        # Update immunity
        hpi.update_peak_immunity(self, inds, imm_pars=self.pars, imm_source=genotype)

        return n_infections # For incrementing counters


    def make_die_other(self, inds):
        ''' Make people die of all other causes (background mortality) '''

        self.other_dead[inds] = True
        self.susceptible[inds] = False
        self.infectious[inds] = False
        self.recovered[inds] = False

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
            if lkey.lower() == 'r':
                llabel = 'regular'
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
                'date_recovered'      : 'recovered',
            }

            for attribute, message in dates.items():
                date = getattr(p,attribute)
                if not np.isnan(date):
                    events.append((date, message))

            # for infection in self.infection_log:
            #     lkey = infection['layer']
            #     llabel = label_lkey(lkey)
            #     if infection['target'] == uid:
            #         if lkey:
            #             events.append((infection['date'], f'was infected with COVID by {infection["source"]} via the {llabel} layer'))
            #         else:
            #             events.append((infection['date'], 'was infected with COVID as a seed infection'))

            #     if infection['source'] == uid:
            #         x = len([a for a in self.infection_log if a['source'] == infection['target']])
            #         events.append((infection['date'],f'gave COVID to {infection["target"]} via the {llabel} layer ({x} secondary infections)'))

            if len(events):
                for timestep, event in sorted(events, key=lambda x: x[0]):
                    print(f'On timestep {timestep:.0f}, {uid} {event}')
            else:
                print(f'Nothing happened to {uid} during the simulation.')
        return

