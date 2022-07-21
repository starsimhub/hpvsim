'''
Define core Sim classes
'''

# Imports
import numpy as np
import pandas as pd
import sciris as sc
from . import base as hpb
from . import misc as hpm
from . import defaults as hpd
from . import utils as hpu
from . import population as hppop
from . import parameters as hppar
from . import analysis as hpa
from . import plotting as hpplt
from .settings import options as hpo
from . import immunity as hpimm
from . import interventions as hpi


# Define the model
class Sim(hpb.BaseSim):

    def __init__(self, pars=None, datafile=None, label=None,
                 popfile=None, people=None, version=None, **kwargs):

        # Set attributes
        self.label         = label    # The label/name of the simulation
        self.created       = None     # The datetime the sim was created
        self.datafile      = datafile # The name of the data file
        self.popfile       = popfile  # The population file
        self.data          = None     # The data
        self.popdict       = people   # The population dictionary
        self.people        = None     # Initialize these here so methods that check their length can see they're empty
        self.t             = None     # The current time in the simulation (during execution); outside of sim.step(), its value corresponds to next timestep to be computed
        self.results       = {}       # For storing results
        self.summary       = None     # For storing a summary of the results
        self.initialized   = False    # Whether or not initialization is complete
        self.complete      = False    # Whether a simulation has completed running
        self.results_ready = False    # Whether or not results are ready
        self._default_ver  = version  # Default version of parameters used
        self._orig_pars    = None     # Store original parameters to optionally restore at the end of the simulation

        # Make default parameters (using values from parameters.py)
        default_pars = hppar.make_pars(version=version) # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes

        # Update pars and load data
        self.update_pars(pars, **kwargs)   # Update the parameters, if provided
        self.load_data(datafile) # Load the data, if provided

        return


    def load_data(self, datafile=None, **kwargs):
        ''' Load the data to calibrate against, if provided '''
        if datafile is not None: # If a data file is provided, load it
            self.data = hpm.load_data(datafile=datafile, **kwargs)
        return


    def initialize(self, reset=False, init_states=True, **kwargs):
        '''
        Perform all initializations on the sim.
        '''
        self.t = 0  # The current time index
        self.validate_pars() # Ensure parameters have valid values
        self.set_seed() # Reset the random seed before the population is created
        self.init_genotypes() # Initialize the genotypes
        self.init_immunity() # initialize information about immunity
        self.init_people(reset=reset, init_states=init_states, **kwargs) # Create all the people (the heaviest step)
        self.init_results() # After initializing the genotypes, create the results structure
        self.init_interventions()  # Initialize the interventions...
        self.init_analyzers()  # ...and the analyzers...
        self.validate_imm_pars()  # Once the population and interventions are initialized, validate the immunity parameters
        self.set_seed() # Reset the random seed again so the random number stream is consistent
        self.initialized   = True
        self.complete      = False
        self.results_ready = False
        return self


    def layer_keys(self):
        '''
        Attempt to retrieve the current layer keys.
        '''
        try:
            keys = list(self['acts'].keys()) # Get keys from acts
        except: # pragma: no cover
            keys = []
        return keys


    def reset_layer_pars(self, layer_keys=None, force=False):
        '''
        Reset the parameters to match the population.

        Args:
            layer_keys (list): override the default layer keys (use stored keys by default)
            force (bool): reset the parameters even if they already exist
        '''
        if layer_keys is None:
            if self.people is not None: # If people exist
                layer_keys = self.people.contacts.keys()
            elif self.popdict is not None:
                layer_keys = self.popdict['layer_keys']
        hppar.reset_layer_pars(self.pars, layer_keys=layer_keys, force=force)
        return


    def validate_layer_pars(self):
        '''
        Handle layer parameters, since they need to be validated after the population
        creation, rather than before.
        '''

        # First, try to figure out what the layer keys should be and perform basic type checking
        layer_keys = self.layer_keys()
        layer_pars = hppar.layer_pars # The names of the parameters that are specified by layer
        for lp in layer_pars:
            val = self[lp]
            if sc.isnumber(val): # It's a scalar instead of a dict, assume it's all contacts
                self[lp] = {k:val for k in layer_keys}

        # Handle key mismatches
        for lp in layer_pars:
            lp_keys = set(self.pars[lp].keys())
            if lp != 'layer_probs':
                if not lp_keys == set(layer_keys):
                    errormsg = 'At least one layer parameter is inconsistent with the layer keys; all parameters must have the same keys:'
                    errormsg += f'\nsim.layer_keys() = {layer_keys}'
                    for lp2 in layer_pars: # Fail on first error, but re-loop to list all of them
                        errormsg += f'\n{lp2} = ' + ', '.join(self.pars[lp2].keys())
                    raise sc.KeyNotFoundError(errormsg)

            # TODO: add validation here for layer_probs

        # Handle mismatches with the population
        if self.people is not None:
            pop_keys = set(self.people.contacts.keys())
            if pop_keys != set(layer_keys): # pragma: no cover
                if not len(pop_keys):
                    errormsg = f'Your population does not have any layer keys, but your simulation does {layer_keys}. If you called cv.People() directly, you probably need cv.make_people() instead.'
                    raise sc.KeyNotFoundError(errormsg)
                else:
                    errormsg = f'Please update your parameter keys {layer_keys} to match population keys {pop_keys}. You may find sim.reset_layer_pars() helpful.'
                    raise sc.KeyNotFoundError(errormsg)

        return

    def validate_imm_pars(self):
        '''
        Handle immunity parameters, since they need to be validated after the population and intervention
        creation, rather than before.
        '''

        # Handle sources, as we need to init the people and interventions first
        self.pars['n_imm_sources'] = self.pars['n_genotypes'] + len(self.pars['vaccine_map'])
        for key in self.people.meta.imm_states:
            if key == 't_imm_event':
                self.people[key] = np.zeros((self.pars['n_imm_sources'], self.pars['pop_size']), dtype=hpd.default_int)
            else:
                self.people[key] = np.zeros((self.pars['n_imm_sources'], self.pars['pop_size']), dtype=hpd.default_float)

        return


    def validate_pars(self, validate_layers=True):
        '''
        Some parameters can take multiple types; this makes them consistent.
        Args:
            validate_layers (bool): whether to validate layer parameters as well via validate_layer_pars() -- usually yes, except during initialization
        '''

        # Handle types
        for key in ['pop_size']:
            try:
                self[key] = int(self[key])
            except Exception as E:
                errormsg = f'Could not convert {key}={self[key]} of {type(self[key])} to integer'
                raise ValueError(errormsg) from E

        # Handle start
        if self['start'] in [None, 0]: # Use default start
            self['start'] = 2015

        # Handle end and n_years
        if self['end']:
            self['n_years'] = int(self['end'] - self['start'])
            if self['n_years'] <= 0:
                errormsg = f"Number of years must be >0, but you supplied start={str(self['start'])} and end={str(self['end'])}, which gives n_years={self['n_years']}"
                raise ValueError(errormsg)
        else:
            if self['n_years']:
                self['end'] = self['start'] + self['n_years']
            else:
                errormsg = f'You must supply one of n_years and end."'
                raise ValueError(errormsg)
        
        # Construct other things that keep track of time
        self.years          = sc.inclusiverange(self['start'],self['end'])
        self.yearvec        = sc.inclusiverange(start=self['start'], stop=self['end'], step=self['dt'])
        self.npts           = len(self.yearvec)
        self.tvec          = np.arange(self.npts)
        
        # Handle population network data
        network_choices = ['random', 'default']
        choice = self['network']
        if choice and choice not in network_choices: # pragma: no cover
            choicestr = ', '.join(network_choices)
            errormsg = f'Population type "{choice}" not available; choices are: {choicestr}'
            raise ValueError(errormsg)

        # Handle analyzers and interventions - TODO, genotypes will also go here
        for key in ['interventions', 'analyzers']: # Ensure all of them are lists
            self[key] = sc.dcp(sc.tolist(self[key], keepnone=False)) # All of these have initialize functions that run into issues if they're reused
        for i,interv in enumerate(self['interventions']):
            if isinstance(interv, dict): # It's a dictionary representation of an intervention
                self['interventions'][i] = hpi.InterventionDict(**interv)

        # Optionally handle layer parameters
        if validate_layers:
            self.validate_layer_pars()

        # Handle verbose
        if self['verbose'] == 'brief':
            self['verbose'] = -1
        if not sc.isnumber(self['verbose']): # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self["verbose"])} "{self["verbose"]}"'
            raise ValueError(errormsg)

        return


    def validate_init_conditions(self, init_hpv_prev):
        '''
        Initial prevalence values can be supplied with different amounts of detail.
        Here we flesh out any missing details so that the initial prev values are
        by age and genotype. We also check the prevalence values are ok.
        '''

        def validate_arrays(vals, n_age_brackets=None):
            ''' Little helper function to check prevalence values '''
            if n_age_brackets is not None:
                if len(vals) != n_age_brackets:
                    errormsg = f'The initial prevalence values must either be the same length as the age brackets: {len(vals)} vs {n_age_brackets}.'
                    raise ValueError(errormsg)
            else:
                if len(vals) != 1:
                    errormsg = f'No age brackets were supplied, but more than one prevalence value was supplied ({len(vals)}). An array of prevalence values can only be supplied along with an array of corresponding age brackets.'
                    raise ValueError(errormsg)
            if vals.any() < 0 or vals.any() > 1:
                errormsg = f'The initial prevalence values must either between 0 and 1, not {vals}.'
                raise ValueError(errormsg)

            return

        # If values have been provided, validate them
        sex_keys = {'m', 'f'}
        tot_keys = ['all', 'total', 'tot', 'average', 'avg']
        n_age_brackets = None

        if init_hpv_prev is not None:
            if sc.checktype(init_hpv_prev, dict):
                # Get age brackets if supplied
                if 'age_brackets' in init_hpv_prev.keys():
                    age_brackets = init_hpv_prev.pop('age_brackets')
                    n_age_brackets = len(age_brackets)
                else:
                    age_brackets = np.array([150])

                # Handle the rest of the keys
                var_keys = list(init_hpv_prev.keys())
                if (len(var_keys)==1 and var_keys[0] not in tot_keys) or (len(var_keys)>1 and set(var_keys) != sex_keys):
                    errormsg = f'Could not understand the initial prevalence provided: {init_hpv_prev}. If supplying a dictionary, please use "m" and "f" keys or "tot". '
                    raise ValueError(errormsg)
                if len(var_keys) == 1:
                    k = var_keys[0]
                    init_hpv_prev = {sk: sc.promotetoarray(init_hpv_prev[k]) for sk in sex_keys}

                # Now set the values
                for k, vals in init_hpv_prev.items():
                    init_hpv_prev[k] = sc.promotetoarray(vals)

            elif sc.checktype(init_hpv_prev, 'arraylike') or sc.isnumber(init_hpv_prev):
                # If it's an array, assume these values apply to males and females
                init_hpv_prev = {sk: sc.promotetoarray(init_hpv_prev) for sk in sex_keys}
                age_brackets = np.array([150])

            else:
                errormsg = f'Initial prevalence values of type {type(init_hpv_prev)} not recognized, must be a dict, an array, or a float.'
                raise ValueError(errormsg)

            # Now validate the arrays
            for sk, vals in init_hpv_prev.items():
                validate_arrays(vals, n_age_brackets)

        # If values haven't been supplied, assume zero
        else:
            init_hpv_prev = {'f': np.array([0]), 'm': np.array([0])}
            age_brackets = np.array([150])

        return init_hpv_prev, age_brackets


    def init_genotypes(self):
        ''' Initialize the genotypes '''
        if self._orig_pars and 'genotypes' in self._orig_pars:
            self['genotypes'] = self._orig_pars.pop('genotypes')  # Restore

        for i, genotype in enumerate(self['genotypes']):
            if isinstance(genotype, hpimm.genotype):
                if not genotype.initialized:
                    genotype.initialize(self)
            else:  # pragma: no cover
                errormsg = f'Genotype {i} ({genotype}) is not a hp.genotype object; please create using cv.genotype()'
                raise TypeError(errormsg)

        if not len(self['genotypes']):
            print('No genotypes provided, will assume only simulating HPV 16 by default')
            hpv16 = hpimm.genotype('hpv16')
            hpv16.initialize(self)
            self['genotypes'] = [hpv16]

        len_pars = len(self['genotype_pars'])
        len_map = len(self['genotype_map'])
        assert len_pars == len_map, f"genotype_pars and genotype_map must be the same length, but they're not: {len_pars} ≠ {len_map}"
        self['n_genotypes'] = len_pars  # Each genotype has an entry in genotype_pars
        self['n_imm_sources'] = len_pars

        return

    def init_immunity(self, create=False):
        ''' Initialize immunity matrices '''
        hpimm.init_immunity(self, create=create)
        return


    def init_results(self, frequency='annual', add_data=True):
        '''
        Create the main results structure.
        The prefix "new" is used for flow variables, i.e. counting new events (infections/deaths) on each timestep
        The prefix "n" is used for stock variables, i.e. counting the total number in any given state (sus/inf/etc) on any particular timestep
        The prefix "cum" is used for cumulative variables, i.e. counting the total number that have ever been in a given state at some point in the sim

        Arguments:
            sim         (hp.Sim)        : a sim
            frequency   (str or float)  : the frequency with which to save results: accepts 'annual', 'dt', or a float which is interpreted as a fraction of a year, e.g. 0.2 will save results every 0.2 years
            add_data    (bool)          : whether or not to add data to the result structures
        '''

        # Handle frequency
        if type(frequency) == str:
            if frequency == 'annual':
                resfreq = int(1 / self['dt'])
            elif frequency == 'dt':
                resfreq = 1
            else:
                errormsg = f'Result frequency not understood: must be "annual", "dt" or a float, but you provided {frequency}.'
                raise ValueError(errormsg)
        elif type(frequency) == float:
            if frequency < self['dt']:
                errormsg = f'You requested results with frequency {frequency}, but this is smaller than the simulation timestep {self["dt"]}.'
                raise ValueError(errormsg)
            else:
                resfreq = int(frequency / self['dt'])
        self.resfreq = resfreq

        # Construct the tvec that will be used with the results
        points_to_use = np.arange(0, self.npts, self.resfreq)
        self.res_yearvec = self.yearvec[points_to_use]
        self.res_npts = len(self.res_yearvec)
        self.res_tvec = np.arange(self.res_npts)

        # Function to create results
        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = hpb.Result(*args, **kwargs, npts=self.res_npts)
            return output

        ng = self['n_genotypes']
        results = dict()

        # Create new flows
        for lkey,llab,cstride,g in zip(['total_',''], ['Total ',''], [0.95,np.linspace(0.2,0.8,ng)], [0,ng]):  # key, label, and color stride by level (total vs genotype-specific)
            for flow,name,cmap in zip(hpd.flow_keys, hpd.flow_names, hpd.flow_colors):
                results[f'{lkey+flow}'] = init_res(f'{llab} {name}', color=cmap(cstride), n_rows=g)

        # Create stocks
        for lkey,llabel,cstride,g in zip(['total_',''], ['Total number','Number'], [0.95,np.linspace(0.2,0.8,ng)], [0,ng]):
            for stock, name, cmap in zip(hpd.stock_keys, hpd.stock_names, hpd.stock_colors):
                results[f'n_{lkey+stock}'] = init_res(f'{llabel} {name}', color=cmap(cstride), n_rows=g)

        # Create incidence and prevalence results
        for lkey,llab,cstride,g in zip(['total_',''], ['Total ',''], [0.95,np.linspace(0.2,0.8,ng)], [0,ng]):  # key, label, and color stride by level (total vs genotype-specific)
            for var,name,cmap in zip(hpd.inci_keys, hpd.inci_names, hpd.inci_colors):
                for which in ['incidence', 'prevalence']:
                    results[f'{lkey+var}_{which}'] = init_res(llab+name+' '+which, color=cmap(cstride), n_rows=g)

        # Create demographic flows
        for var, name, color in zip(hpd.dem_keys, hpd.dem_names, hpd.dem_colors):
            results[f'{var}'] = init_res(f'{name}', color=color)

        # Create results by sex
        for var, name, color in zip(hpd.by_sex_keys, hpd.by_sex_colors, hpd.by_sex_colors):
            results[f'{var}'] = init_res(f'{name}', color=color, n_rows=2)

        # Vaccination results
        results['new_vaccinated'] = init_res('Newly vaccinated by genotype', n_rows=ng)
        results['new_total_vaccinated'] = init_res('Newly vaccinated')
        results['cum_vaccinated'] = init_res('Cumulative number vaccinated by genotype', n_rows=ng)
        results['cum_total_vaccinated'] = init_res('Cumulative number vaccinated')
        results['new_doses'] = init_res('New doses')
        results['cum_doses'] = init_res('Cumulative doses')

        # Detectable HPV
        results['n_detectable_hpv'] = init_res('Number with detectable HPV', n_rows=ng)
        results['n_total_detectable_hpv'] = init_res('Number with detectable HPV')
        results['detectable_hpv_prevalence'] = init_res('Detectable HPV prevalence', n_rows=ng, color=hpd.stock_colors[0](np.linspace(0.9,0.5,ng)))
        results['total_detectable_hpv_prevalence'] = init_res('Total detectable HPV prevalence')

        # Other results
        results['r_eff'] = init_res('Effective reproduction number', scale=False, n_rows=ng)
        results['doubling_time'] = init_res('Doubling time', scale=False, n_rows=ng)
        results['n_alive'] = init_res('Number alive')
        results['n_alive_by_sex'] = init_res('Number alive by sex', n_rows=2)
        results['cdr'] = init_res('Crude death rate', scale=False)
        results['cbr'] = init_res('Crude birth rate', scale=False, color='#fcba03')

        # Time vector
        results['year'] = self.res_yearvec
        results['t'] = self.res_tvec

        # Final items
        self.rescale_vec   = self['pop_scale']*np.ones(self.res_npts) # Not included in the results, but used to scale them
        self.results = results
        self.results_ready = False

        return


    def init_people(self, popdict=None, init_states=False, reset=False, verbose=None, **kwargs):
        '''
        Create the people and the network.

        Use ``init_states=False`` for creating a fresh People object for use
        in future simulations

        Args:
            popdict         (any):  pre-generated people of various formats.
            init_states     (bool): whether to initialize states (default false when called directly)
            reset           (bool): whether to regenerate the people even if they already exist
            verbose         (int):  detail to print
            kwargs          (dict): passed to hp.make_people()
        '''

        # Handle inputs
        if verbose is None:
            verbose = self['verbose']
        if popdict is not None:
            self.popdict = popdict
        if verbose > 0:
            resetstr= ''
            if self.people:
                resetstr = ' (resetting people)' if reset else ' (warning: not resetting sim.people)'
            print(f'Initializing sim{resetstr} with {self["pop_size"]:0n} people')
        if self.popfile and self.popdict is None: # If there's a popdict, we initialize it
            self.load_population(init_people=False)

        # Actually make the people
        microstructure = self['network']
        self.people, total_pop = hppop.make_people(self, reset=reset, verbose=verbose, microstructure=microstructure, **kwargs)
        self.people.initialize(sim_pars=self.pars) # Fully initialize the people
        self.reset_layer_pars(force=False) # Ensure that layer keys match the loaded population
        if init_states:
            init_hpv_prev = sc.dcp(self['init_hpv_prev'])
            init_hpv_prev, age_brackets = self.validate_init_conditions(init_hpv_prev)
            self.init_states(age_brackets=age_brackets, init_hpv_prev=init_hpv_prev)

        # If no pop_scale has been provided, try to get it from the location
        if self['pop_scale'] is None:
            if self['location'] is None or total_pop is None:
                self['pop_scale'] = 1
            else:
                self['pop_scale'] = total_pop/self['pop_size']

        return self


    def init_interventions(self):
        ''' Initialize and validate the interventions '''

        # Initialization
        if self._orig_pars and 'interventions' in self._orig_pars:
            self['interventions'] = self._orig_pars.pop('interventions') # Restore

        for i,intervention in enumerate(self['interventions']):
            if isinstance(intervention, hpi.Intervention):
                intervention.initialize(self)
        return


    def finalize_interventions(self):
        for intervention in self['interventions']:
            if isinstance(intervention, hpimm.Intervention):
                intervention.finalize(self)


    def init_analyzers(self):
        ''' Initialize the analyzers '''
        if self._orig_pars and 'analyzers' in self._orig_pars:
            self['analyzers'] = self._orig_pars.pop('analyzers') # Restore

        for analyzer in self['analyzers']:
            if isinstance(analyzer, hpa.Analyzer):
                analyzer.initialize(self)
        return


    def finalize_analyzers(self):
        for analyzer in self['analyzers']:
            if isinstance(analyzer, hpa.Analyzer):
                analyzer.finalize(self)


    def reset_layer_pars(self, layer_keys=None, force=False):
        '''
        Reset the parameters to match the population.

        Args:
            layer_keys (list): override the default layer keys (use stored keys by default)
            force (bool): reset the parameters even if they already exist
        '''
        if layer_keys is None:
            if self.people is not None: # If people exist
                layer_keys = self.people.contacts.keys()
            elif self.popdict is not None:
                layer_keys = self.popdict['layer_keys']
        hppar.reset_layer_pars(self.pars, layer_keys=layer_keys, force=force)
        return


    def init_states(self, age_brackets=None, init_hpv_prev=None, init_cin_prev=None, init_cancer_prev=None):
        '''
        Initialize prior immunity and seed infections
        '''

        # Shorten key variables
        ng = self['n_genotypes']

        # Assign people to age buckets
        age_inds = np.digitize(self.people.age, age_brackets)

        # Assign probabilities of having HPV to each age/sex group
        hpv_probs = np.full(len(self.people), np.nan, dtype=hpd.default_float)
        hpv_probs[self.people.f_inds] = init_hpv_prev['f'][age_inds[self.people.f_inds]]
        hpv_probs[self.people.m_inds] = init_hpv_prev['m'][age_inds[self.people.m_inds]]
        hpv_probs[~self.people.is_active] = 0 # Blank out people who are not yet sexually active

        # Get indices of people who have HPV
        hpv_inds = hpu.true(hpu.binomial_arr(hpv_probs))

        # Determine which genotype people are infected with
        if self['init_hpv_dist'] is None: # No type distribution provided, assume even split
            genotypes = np.random.randint(0, ng, len(hpv_inds))
        else:
            # Error checking
            if not sc.checktype(self['init_hpv_dist'], dict):
                errormsg = f'Please provide initial HPV type distribution as a dictionary keyed by genotype, not {self["init_hpv_dist"]}'
                raise ValueError(errormsg)
            if set(self['init_hpv_dist'].keys())!=set(self['genotype_map'].values()):
                errormsg = f'The HPV types provided in the initial HPV type distribution are not the same as the HPV types being simulated: {self["init_hpv_dist"].keys()} vs {self["genotype_map"].values()}.'
                raise ValueError(errormsg)

            type_dist = np.array(list(self['init_hpv_dist'].values()))
            genotypes = hpu.choose_w(type_dist, len(hpv_inds), unique=False)

        # Figure of duration of infection and infect people
        genotype_pars = self.pars['genotype_pars']
        genotype_map = self.pars['genotype_map']

        for g in range(ng):
            durpars = genotype_pars[genotype_map[g]]['dur']
            dur_hpv = hpu.sample(**durpars['none'], size=len(hpv_inds))
            t_imm_event = np.floor(np.random.uniform(-dur_hpv, 0) / self['dt'])
            _ = self.people.infect(inds=hpv_inds[genotypes==g], g=g, offset=t_imm_event[genotypes==g], dur=dur_hpv[genotypes==g], layer='seed_infection')

        # Check for CINs
        cin1_filters = (self.people.date_cin1<0) * (self.people.date_cin2 > 0)
        self.people.cin1[cin1_filters.nonzero()] = True
        cin2_filters = (self.people.date_cin2<0) * (self.people.date_cin3 > 0)
        self.people.cin2[cin2_filters.nonzero()] = True
        cin3_filters = (self.people.date_cin3<0) * (self.people.date_cancerous > 0)
        self.people.cin3[cin3_filters.nonzero()] = True

        return


    def step(self):
        ''' Step through time and update values '''

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            raise AlreadyRunError('Simulation already complete (call sim.initialize() to re-run)')

        # Shorten key variables
        dt = self['dt'] # Timestep
        t = self.t
        ng = self['n_genotypes']
        condoms = self['condoms']
        eff_condoms = self['eff_condoms']
        beta = self['beta']
        gen_pars = self['genotype_pars']
        imm_kin_pars = self['imm_kin']
        trans = np.array([self['transf2m'],self['transm2f']]) # F2M first since that's the order things are done later

        # Update demographics and partnerships
        old_pop_size = len(self.people)
        self.people.update_states_pre(t=t, year=self.yearvec[t], resfreq=self.resfreq) # NB this also ages people, applies deaths, and generates new births

        people = self.people # Shorten
        n_dissolved = people.dissolve_partnerships(t=t) # Dissolve partnerships
        new_pop_size = len(people)
        people.create_partnerships(t=t, n_new=n_dissolved, scale_factor=new_pop_size/old_pop_size) # Create new partnerships (maintaining the same overall partnerhip rate)
        n_people = len(people)

        # Apply interventions
        for i,intervention in enumerate(self['interventions']):
            intervention(self) # If it's a function, call it directly

        contacts = people.contacts # Shorten

        # Assign sus_imm values, i.e. the protection against infection based on prior immune history
        if self['use_waning']:
            has_imm = hpu.true(people.peak_imm.sum(axis=0)).astype(hpd.default_int)
            if len(has_imm):
                hpu.update_immunity(people.imm, t, people.t_imm_event, has_imm, imm_kin_pars, people.peak_imm)
            hpimm.check_immunity(people)
        else:
            people.imm = people.peak_imm

        # Precalculate aspects of transmission that don't depend on genotype (acts, condoms)
        fs, ms, frac_acts, whole_acts, effective_condoms = [], [], [], [], []
        for lkey, layer in contacts.items():
            fs.append(layer['f'])
            ms.append(layer['m'])

            # Get the number of acts per timestep for this partnership type
            acts = layer['acts'] * dt
            fa, wa = np.modf(layer['acts'] * dt)
            frac_acts.append(fa)
            whole_acts.append(wa.astype(hpd.default_int))
            effective_condoms.append(hpd.default_float(condoms[lkey] * eff_condoms))

        # Shorten more variables
        inf = people.infectious
        sus = people.susceptible
        sus_imm = people.sus_imm

        # Get indices of infected/susceptible people by genotype
        f_inf_genotypes, f_inf_inds, f_sus_genotypes, f_sus_inds = hpu.get_sources_targets(inf, sus, ~people.sex.astype(bool))  # Males and females infected with this genotype
        m_inf_genotypes, m_inf_inds, m_sus_genotypes, m_sus_inds = hpu.get_sources_targets(inf, sus,  people.sex.astype(bool))  # Males and females infected with this genotype

        # Calculate relative transmissibility by stage of infection
        rel_trans_pars = self['rel_trans']
        rel_trans = people.infectious[:].astype(hpd.default_float)
        rel_trans[people.cin1] *= rel_trans_pars['cin1']
        rel_trans[people.cin2] *= rel_trans_pars['cin2']
        rel_trans[people.cin3] *= rel_trans_pars['cin3']
        rel_trans[people.cancerous] *= rel_trans_pars['cancerous']

        # Loop over layers
        ln = 0 # Layer number
        for lkey, layer in contacts.items():
            f = fs[ln]
            m = ms[ln]

            # Compute transmissions
            for g in range(ng):
                f_source_inds = hpu.get_discordant_pairs2(f_inf_inds[f_inf_genotypes==g], m_sus_inds[m_sus_genotypes==g], f, m, n_people)
                m_source_inds = hpu.get_discordant_pairs2(m_inf_inds[m_inf_genotypes==g], f_sus_inds[f_sus_genotypes==g], m, f, n_people)

                foi_frac = 1 - frac_acts[ln] * beta * trans[:, None] * (1 - effective_condoms[ln])  # Probability of not getting infected from any fractional acts
                foi_whole = (1 - beta * trans[:, None] * (1 - effective_condoms[ln])) ** whole_acts[ln]  # Probability of not getting infected from whole acts
                foi = (1 - (foi_whole * foi_frac)).astype(hpd.default_float)

                discordant_pairs = [[f_source_inds, f[f_source_inds], m[f_source_inds], f_inf_genotypes[f_inf_genotypes==g], foi[0,:]],
                                    [m_source_inds, m[m_source_inds], f[m_source_inds], m_inf_genotypes[m_inf_genotypes==g], foi[1,:]]]

                # Compute transmissibility for each partnership
                for pship_inds, sources, targets, genotypes, this_foi in discordant_pairs:
                    betas = this_foi[pship_inds] * (1. - sus_imm[g,targets]) * rel_trans[g,sources] # Pull out the transmissibility associated with this partnership
                    target_inds = hpu.compute_infections(betas, targets)  # Calculate transmission
                    target_inds, unique_inds = np.unique(target_inds, return_index=True)  # Due to multiple partnerships, some people will be counted twice; remove them
                    people.infect(inds=target_inds, g=g, layer=lkey)  # Actually infect people

            ln += 1

        # Determine if there are any reactivated infections on this timestep
        for g in range(ng):
            latent_inds = hpu.true(people.latent[g,:])
            if len(latent_inds):
                age_inds = np.digitize(people.age[latent_inds], self['hpv_reactivation']['age_cutoffs'])-1 # convert ages to indices
                reactivation_probs = self['hpv_reactivation']['hpv_reactivation_probs'][age_inds]
                is_reactivated = hpu.binomial_arr(reactivation_probs)
                reactivated_inds = latent_inds[is_reactivated]
                people.infect(inds=reactivated_inds, g=g, layer='reactivation')


        # Index for results
        idx = int(t / self.resfreq)

        # Store whether people have any grade of CIN
        people.cin = people.cin1 + people.cin2 + people.cin3

        # Update counts for this time step: flows
        for key,count in people.total_flows.items():
            self.results[key][idx] += count
        for key,count in people.demographic_flows.items():
            self.results[key][idx] += count
        for key,count in people.flows.items():
            for genotype in range(ng):
                self.results[key][genotype][idx] += count[genotype]
        for key,count in people.flows_by_sex.items():
            for sex in range(2):
                self.results[key][sex][idx] += count[sex]

        # Make stock updates every nth step, where n is the frequency of result output
        if t % self.resfreq == 0:

            # Create total stocks
            for key in hpd.stock_keys:
                if key not in ['alive', 'vaccinated']:  # These are all special cases
                    for g in range(ng):
                        self.results[f'n_{key}'][g, idx] = people.count_by_genotype(key, g)
                if key not in ['susceptible']:
                    # For n_infectious, n_cin1, etc, we get the total number where this state is true for at least one genotype
                    self.results[f'n_total_{key}'][idx] = np.count_nonzero(people[key].sum(axis=0))
                elif key == 'susceptible':
                    # For n_total_susceptible, we get the total number of infections that could theoretically happen in the population, which can be greater than the population size
                    self.results[f'n_total_{key}'][idx] = people.count(key)

            # Compute detectable hpv prevalence
            hpv_test_pars = hppar.get_screen_pars('hpv')
            for state in ['hpv', 'cin1', 'cin2', 'cin3']:
                hpv_pos_probs = np.zeros(len(people))
                for g in range(ng):
                    tp_inds = hpu.true(people[state][g,:])
                    hpv_pos_probs[tp_inds] = hpv_test_pars['test_positivity'][state][self['genotype_map'][g]]
                    hpv_pos_inds = hpu.true(hpu.binomial_arr(hpv_pos_probs))
                    self.results['n_detectable_hpv'][g, idx] = len(hpv_pos_inds)
                    self.results['n_total_detectable_hpv'][idx] += len(hpv_pos_inds)

            # Save number alive
            self.results['n_alive'][idx] = len(people.alive.nonzero()[0])
            self.results['n_alive_by_sex'][0,idx] = len((people.alive*people.is_female).nonzero()[0])
            self.results['n_alive_by_sex'][1,idx] = len((people.alive*people.is_male).nonzero()[0])

        # Apply analyzers
        for i,analyzer in enumerate(self['analyzers']):
            analyzer(self)

        # Tidy up
        self.t += 1
        if self.t == self.npts:
            self.complete = True

        return


    def run(self, do_plot=False, until=None, restore_pars=True, reset_seed=True, verbose=None):
        ''' Run the model once '''
        # Initialization steps -- start the timer, initialize the sim and the seed, and check that the sim hasn't been run
        T = sc.timer()

        if not self.initialized:
            self.initialize()
            self._orig_pars = sc.dcp(self.pars) # Create a copy of the parameters, to restore after the run, in case they are dynamically modified

        if verbose is None:
            verbose = self['verbose']

        if reset_seed:
            # Reset the RNG. If the simulation is newly created, then the RNG will be reset by sim.initialize() so the use case
            # for resetting the seed here is if the simulation has been partially run, and changing the seed is required
            self.set_seed()

        # Check for AlreadyRun errors
        errormsg = None
        until = self.npts if until is None else self.get_t(until)
        if until > self.npts:
            errormsg = f'Requested to run until t={until} but the simulation end is t={self.npts}'
        if self.t >= until: # NB. At the start, self.t is None so this check must occur after initialization
            errormsg = f'Simulation is currently at t={self.t}, requested to run until t={until} which has already been reached'
        if self.complete:
            errormsg = 'Simulation is already complete (call sim.initialize() to re-run)'
        if self.people.t not in [self.t, self.t-1]: # Depending on how the sim stopped, either of these states are possible
            errormsg = f'The simulation has been run independently from the people (t={self.t}, people.t={self.people.t}): if this is intentional, manually set sim.people.t = sim.t. Remember to save the people object before running the sim.'
        if errormsg:
            raise AlreadyRunError(errormsg)

        # Main simulation loop
        while self.t < until:

            # Check if we were asked to stop
            elapsed = T.toc(output=True)
            if self['timelimit'] and elapsed > self['timelimit']:
                sc.printv(f"Time limit ({self['timelimit']} s) exceeded; call sim.finalize() to compute results if desired", 1, verbose)
                return
            elif self['stopping_func'] and self['stopping_func'](self):
                sc.printv("Stopping function terminated the simulation; call sim.finalize() to compute results if desired", 1, verbose)
                return

            # Print progress
            if verbose:
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.yearvec[self.t]} ({self.t:2.0f}/{self.npts}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose>0:
                    if not (self.t % int(1.0/verbose)):
                        sc.progressbar(self.t+1, self.npts, label=string, length=20, newline=True)

            # Do the heavy lifting -- actually run the model!
            self.step()

        # If simulation reached the end, finalize the results
        if self.complete:
            self.finalize(verbose=verbose, restore_pars=restore_pars)
            sc.printv(f'Run finished after {elapsed:0.2f} s.\n', 1, verbose)

        return self


    def finalize(self, verbose=None, restore_pars=True):
        ''' Compute final results '''

        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise AlreadyRunError('Simulation has already been finalized')

        # Fix the last timepoint
        if self.resfreq>1:
            for reskey in hpd.flow_keys:
                self.results[reskey][:,-1] *= self.resfreq/(self.t % self.resfreq) # Scale
                self.results[f'total_{reskey}'][-1] *= self.resfreq/(self.t % self.resfreq) # Scale
            self.results['births'][-1] *= self.resfreq/(self.t % self.resfreq) # Scale
            self.results['other_deaths'][-1] *= self.resfreq/(self.t % self.resfreq) # Scale
        # Scale the results
        for reskey in self.result_keys():
            if self.results[reskey].scale:
                self.results[reskey].values *= self.rescale_vec

        # Finalize analyzers and interventions
        self.finalize_analyzers()
        # self.finalize_interventions()

        # Final settings
        self.results_ready = True # Set this first so self.summary() knows to print the results
        self.t -= 1 # During the run, this keeps track of the next step; restore this be the final day of the sim

        # Perform calculations on results
        self.compute_results(verbose=verbose) # Calculate the rest of the results
        self.results = sc.objdict(self.results) # Convert results to a odicts/objdict to allow e.g. sim.results.diagnoses

        # Optionally print summary output
        if verbose: # Verbose is any non-zero value
            if verbose>0: # Verbose is any positive number
                self.summarize() # Print medium-length summary of the sim
            else:
                self.brief() # Print brief summary of the sim

        return


    def compute_results(self, verbose=None):
        ''' Perform final calculations on the results '''
        self.compute_states()
        self.compute_summary()
        return


    def compute_states(self):
        '''
        Compute prevalence, incidence, and other states.
        '''
        res = self.results

        # Compute HPV incidence and prevalence
        self.results['total_hpv_incidence'][:]  = res['total_infections'][:]/ res['n_total_susceptible'][:]
        self.results['hpv_incidence'][:]        = res['infections'][:]/ res['n_susceptible'][:]
        self.results['total_hpv_prevalence'][:] = res['n_total_infectious'][:] / res['n_alive'][:]
        self.results['hpv_prevalence'][:]       = res['n_infectious'][:] / res['n_alive'][:]
        self.results['detectable_hpv_prevalence'][:] = res['n_detectable_hpv'][:] / res['n_alive'][:]
        self.results['total_detectable_hpv_prevalence'][:] = res['n_total_detectable_hpv'][:] / res['n_alive'][:]

        # Compute CIN and cancer prevalence
        alive_females = res['n_alive_by_sex'][0,:]
        self.results['total_cin1_prevalence'][:]    = res['n_total_cin1'][:] / alive_females
        self.results['total_cin2_prevalence'][:]    = res['n_total_cin2'][:] / alive_females
        self.results['total_cin3_prevalence'][:]    = res['n_total_cin3'][:] / alive_females
        self.results['total_cin_prevalence'][:]     = res['n_total_cin'][:] / alive_females
        self.results['total_cancer_prevalence'][:]  = res['n_total_cancerous'][:] / alive_females
        self.results['cin1_prevalence'][:]          = res['n_cin1'][:] / alive_females
        self.results['cin2_prevalence'][:]          = res['n_cin2'][:] / alive_females
        self.results['cin3_prevalence'][:]          = res['n_cin3'][:] / alive_females
        self.results['cin_prevalence'][:]           = res['n_cin'][:] / alive_females
        self.results['cancer_prevalence'][:]        = res['n_cancerous'][:] / alive_females

        # Compute CIN and cancer incidence. Technically the denominator should be number susceptible
        # to CIN/cancer, not number alive, but should be small enough that it won't matter (?)
        at_risk_females = alive_females - res['n_cancerous'].values.sum(axis=0)
        scale_factor = 1e5  # Cancer and CIN incidence are displayed as rates per 100k women
        demoninator = at_risk_females / scale_factor
        self.results['total_cin1_incidence'][:]    = res['total_cin1s'][:] / demoninator
        self.results['total_cin2_incidence'][:]    = res['total_cin2s'][:] / demoninator
        self.results['total_cin3_incidence'][:]    = res['total_cin3s'][:] / demoninator
        self.results['total_cin_incidence'][:]     = res['total_cins'][:] / demoninator
        self.results['total_cancer_incidence'][:]  = res['total_cancers'][:] / demoninator
        self.results['cin1_incidence'][:]          = res['cin1s'][:] / demoninator
        self.results['cin2_incidence'][:]          = res['cin2s'][:] / demoninator
        self.results['cin3_incidence'][:]          = res['cin3s'][:] / demoninator
        self.results['cin_incidence'][:]           = res['cins'][:] / demoninator
        self.results['cancer_incidence'][:]        = res['cancers'][:] / demoninator

        # Demographic results
        self.results['cdr'][:]  = self.results['other_deaths'][:] / (self.results['n_alive'][:])
        self.results['cbr'][:]  = self.results['births'][:] / (self.results['n_alive'][:])

        # Vaccination results
        self.results['cum_vaccinated'][:] = np.cumsum(self.results['new_vaccinated'][:], axis=0)
        self.results['cum_total_vaccinated'][:] = np.cumsum(self.results['new_total_vaccinated'][:])
        self.results['cum_doses'][:] = np.cumsum(self.results['new_doses'][:])

        return


    def compute_summary(self, t=None, update=True, output=False, require_run=False):
        '''
        Compute the summary dict and string for the sim. Used internally; see
        sim.summarize() for the user version.

        Args:
            t (int/str): day or date to compute summary for (by default, the last point)
            update (bool): whether to update the stored sim.summary
            output (bool): whether to return the summary
            require_run (bool): whether to raise an exception if simulations have not been run yet
        '''
        if t is None:
            t = -1

        # Compute the summary
        if require_run and not self.results_ready:
            errormsg = 'Simulation not yet run'
            raise RuntimeError(errormsg)

        summary = sc.objdict()
        for key in self.result_keys('total'):
            summary[key] = self.results[key][t]

        # Update the stored state
        if update:
            self.summary = summary

        # Optionally return
        if output:
            return summary
        else:
            return


    def summarize(self, full=False, t=None, sep=None, output=False):
        '''
        Print a medium-length summary of the simulation, drawing from the last time
        point in the simulation by default. Called by default at the end of a sim run.
        point in the simulation by default. Called by default at the end of a sim run.
        See also sim.disp() (detailed output) and sim.brief() (short output).

        Args:
            full   (bool):    whether or not to print all results (by default, only cumulative)
            t      (int/str): day or date to compute summary for (by default, the last point)
            sep    (str):     thousands separator (default ',')
            output (bool):    whether to return the summary instead of printing it

        **Examples**::

            sim = cv.Sim(label='Example sim', verbose=0) # Set to run silently
            sim.run() # Run the sim
            sim.summarize() # Print medium-length summary of the sim
            sim.summarize(t=24, full=True) # Print a "slice" of all sim results on day 24
        '''
        # Compute the summary
        summary = self.compute_summary(t=t, update=False, output=True)

        # Construct the output string
        if sep is None: sep = hpo.sep # Default separator
        labelstr = f' "{self.label}"' if self.label else ''
        string = f'Simulation{labelstr} summary:\n'
        for key in self.result_keys():
            if full or key.startswith('total') and 'by_sex' not in key:
                val = np.round(summary[key])
                string += f'   {val:10,.0f} {self.results[key].name.lower()}\n'.replace(',', sep) # Use replace since it's more flexible

        # Print or return string
        if not output:
            print(string)
        else:
            return string


    def plot(self, *args, **kwargs):
        ''' Plot the outputs of the model '''
        fig = hpplt.plot_sim(sim=self, *args, **kwargs)
        return fig


    def compute_fit(self):
        '''
        Compute fit between model and data.
        '''
        return self.fit


class AlreadyRunError(RuntimeError):
    '''
    This error is raised if a simulation is run in such a way that no timesteps
    will be taken. This error is a distinct type so that it can be safely caught
    and ignored if required, but it is anticipated that most of the time, calling
    sim.run() and not taking any timesteps, would be an inadvertent error.
    '''
    pass
