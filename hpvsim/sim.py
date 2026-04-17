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
from . import hiv as hphiv
from . import analysis as hpa
from . import plotting as hpplt
from . import immunity as hpimm
from . import interventions as hpi
from .settings import options as hpo


# Define the model
class Sim(hpb.BaseSim):

    def __init__(self, pars=None, datafile=None, label=None,
                 popfile=None, popdict=None, people=None, version=None, hiv_datafile=None, art_datafile=None,
                 **kwargs):

        # Set attributes
        self.label         = label    # The label/name of the simulation
        self.created       = None     # The datetime the sim was created
        self.datafile      = datafile # The name of the data file
        self.art_datafile  = art_datafile # The name of the ART data file
        self.hiv_datafile  = hiv_datafile # The name of the HIV data file
        self.popfile       = popfile  # The population file
        self.data          = None     # The data
        self.popdict       = popdict  # The population dictionary
        self.people        = people   # People object
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
        default_location = sc.dcp(default_pars['location']) # Pull out the default location here
        default_pars['location'] = None # Don't load the default location here
        super().__init__(default_pars) # Initialize and set the parameters as attributes

        # Load data, including datafile that are used to create additional optional parameters
        self.load_data(datafile) # Load the data, if provided

        # Update parameters, including demographic data
        if pars is None:
            pars = dict(location=default_location)
        else:
            if not pars.get('location') or pars['location'] is None:
                pars['location'] = default_location

        self.update_pars(pars, **kwargs)   # Update the parameters

        return


    def load_data(self, datafile=None, **kwargs):
        ''' Load the data to calibrate against, if provided '''
        if datafile is not None: # If a data file is provided, load it
            self.data = hpm.load_data(datafile=datafile, check_date=True, **kwargs)
        return


    def initialize(self, reset=True, init_states=True, init_intvs=True, init_analyzers=True, **kwargs):
        '''
        Perform all initializations on the sim.
        '''
        self.t = 0  # The current time index
        self.validate_pars()  # Ensure parameters have valid values
        self.validate_dt()
        self.init_time_vecs()  # Initialise time vectors
        hpu.set_seed(self['rand_seed']) # Reset the random seed before the population is created
        self.init_genotypes() # Initialize the genotypes
        self.init_results() # After initializing the genotypes and people, create the results structure
        if init_intvs: self.init_interventions()  # Initialize the interventions BEFORE the people, because then vaccination interventions get counted in immunity structures
        self.init_immunity() # Includes immunity matrices and cumulative dysplasia arrays
        self.init_people(reset=reset, init_states=init_states, **kwargs) # Create all the people (the heaviest step)
        if init_analyzers: self.init_analyzers()  # ...and the analyzers...
        hpu.set_seed(self['rand_seed']+1)  # Reset the random seed to the default run seed, so that if the simulation is run with reset_seed=False right after initialization, it will still produce the same output
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
                    errormsg = f'Your population does not have any layer keys, but your simulation does {layer_keys}. If you called hpv.People() directly, you probably need hpv.make_people() instead.'
                    raise sc.KeyNotFoundError(errormsg)
                else:
                    errormsg = f'Please update your parameter keys {layer_keys} to match population keys {pop_keys}. You may find sim.reset_layer_pars() helpful.'
                    raise sc.KeyNotFoundError(errormsg)

        return


    def validate_dt(self):
        '''
        Check that 1/dt is an integer value, otherwise results and time vectors will have mismatching shapes.
        init_results explicitly makes this assumption by casting resfrequency = int(1/dt).
        '''
        dt = self['dt']
        reciprocal = 1.0 / dt   # Compute the reciprocal of dt
        if not reciprocal.is_integer():  # Check if reciprocal is not a whole (integer) number
            # Round the reciprocal
            reciprocal = int(reciprocal)
            rounded_dt = 1.0 / reciprocal
            self['dt'] = rounded_dt
            if self['verbose']:
                warnmsg = f"Warning: Provided time step dt: {dt} resulted in a non-integer number of steps/year. Rounded to {rounded_dt}."
                print(warnmsg)


    def validate_pars(self, validate_layers=True):
        '''
        Some parameters can take multiple types; this makes them consistent.

        Args:
            validate_layers (bool): whether to validate layer parameters as well via validate_layer_pars() -- usually yes, except during initialization
        '''

        # Handle types
        for key in ['n_agents']:
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
                errormsg = 'You must supply one of n_years and end."'
                raise ValueError(errormsg)

        # Handle population network data
        network_choices = ['random', 'default']
        choice = self['network']
        if choice and choice not in network_choices: # pragma: no cover
            choicestr = ', '.join(network_choices)
            errormsg = f'Population type "{choice}" not available; choices are: {choicestr}'
            raise ValueError(errormsg)

        # Handle analyzers and interventions
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


    def init_time_vecs(self):
        '''
        Construct vectors things that keep track of time
        '''
        self.years      = sc.inclusiverange(self['start'],self['end'])
        self.yearvec    = sc.inclusiverange(start=self['start'], stop=self['end']+1-self['dt'], step=self['dt']) # Includes all the timepoints in the last year
        self.npts       = len(self.yearvec)
        self.tvec       = np.arange(self.npts)


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


    def init_genotypes(self, upper_dysp_lim=200):
        ''' Initialize the genotype parameters '''
        if self._orig_pars and 'genotypes' in self._orig_pars:
            self['genotypes'] = self._orig_pars.pop('genotypes')  # Restore

        default_gpars   = hppar.get_genotype_pars()
        user_gpars      = sc.dcp(self['genotype_pars'])
        self['genotype_pars'] = sc.objdict()

        # Handle special input cases
        if self['genotypes'] == 'all':
            self['genotypes'] = default_gpars.keys()
        if not len(self['genotypes']):
            print('No genotypes provided: simulating 16, 18, and 5 other pooled HR types (31, 33, 45, 52, 58).')
            self['genotypes'] = [16,18,'hi5']

        # Loop over genotypes
        for i, g in enumerate(self['genotypes']):

            # Standardize format of genotype inputs
            if sc.isnumber(g): g = f'hpv{g}' # Convert e.g. 16 to hpv16
            if sc.checktype(g,str):
                if not g in default_gpars.keys():
                    errormsg = f'Genotype {i} ({g}) is not one of the inbuilt options.'
                    raise ValueError(errormsg)
            else:
                errormsg = f'Format {type(g)} is not understood.'
                raise ValueError(errormsg)

            # Add to genotype_par dict
            self['genotype_pars'][g] = default_gpars[g]
            self['genotype_map'][i] = g

        # Loop over user-supplied genotype parameters that can overwrite values
        if len(user_gpars):
            for g,gpars in user_gpars.items():

                # Standardize format of genotype inputs
                if sc.isnumber(g): g = f'hpv{g}'  # Convert e.g. 16 to hpv16
                if sc.checktype(g, str):
                    if not g in self['genotype_pars'].keys():
                        errormsg = f'Parameters provided for genotype {g}, but it is not in the sim.'
                        raise ValueError(errormsg)
                    else:
                        for gparname,gparval in gpars.items():
                            if gparname in self['genotype_pars'][g].keys():
                                printmsg = f"Resetting parameter '{gparname}' from {self['genotype_pars'][g][gparname]} to {gparval} for genotype {g}"
                                sc.printv(printmsg, 1, self['verbose'])
                                self['genotype_pars'][g][gparname] = gparval
                            else:
                                errormsg = f"Parameter {gparname} does not exist for genotype {g}"
                                raise ValueError(errormsg)

        self['n_genotypes'] = len(self['genotype_pars'])  # Each genotype has an entry in genotype_pars

        # Set the number of immunity sources
        self['n_imm_sources'] = len(self['genotypes'])

        return


    def init_results(self, frequency='annual', add_data=True):
        '''
        Create the main results structure.
        The prefix "n" is used for stock variables, i.e. counting the total number in any given state (sus/inf/etc) on any particular timestep

        Arguments:
            sim         (hpv.Sim)       : a sim
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
        if not self.resfreq > 0:
            errormsg = f'The results frequence should be a positive integer, not {self.resfreq}: dt may be too large'
            raise ValueError(errormsg)

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

        # Initialize storage
        results = sc.objdict()

        ng = self['n_genotypes'] # Number of genotypes
        na = len(self['age_bin_edges']) - 1 # Number of age bins

        # Create flows
        for flow in hpd.flows:
            results[flow.name] = init_res(flow.label, color=flow.color)
            if flow.by_genotype: results[flow.name+'_by_genotype'] = init_res(flow.label+' by genotype', n_rows=ng)
            results[flow.name+'_by_age'] = init_res(flow.label+' by age', n_rows=na, color=flow.color)

        # Create stocks
        for stock in hpd.PeopleMeta().stock_states:
            results[f'n_{stock.name}'] = init_res(stock.label, color=stock.color)
            if stock.shape == 'n_genotypes': results[f'n_{stock.name}_by_genotype'] = init_res(stock.label+' by genotype', n_rows=ng)

        # Only by-age stock result we will need is number infectious, susceptible, and with cin, for HPV and CIN prevalence/incidence calculations
        results['n_infectious_by_age']             = init_res('Number infectious by age', n_rows=na, color=stock.color)
        results['n_females_infectious_by_age']     = init_res('Number of females infectious by age', n_rows=na, color=stock.color)
        results['n_susceptible_by_age']            = init_res('Number susceptible by age', n_rows=na, color=stock.color)
        results['n_precin_by_age']                 = init_res('Number Pre-CIN by age', n_rows=na, color=stock.color)
        results['n_cin_by_age']                    = init_res('Number CIN by age', n_rows=na, color=stock.color)

        # Create incidence and prevalence results
        for var,name,color in zip(hpd.inci_keys, hpd.inci_names, hpd.inci_colors):
            results[f'{var}_incidence']             = init_res(name+' incidence', color=color)
            results[f'{var}_incidence_by_genotype'] = init_res(name+' incidence by genotype', n_rows=ng)
            results[f'{var}_incidence_by_age']      = init_res(name+' incidence by age', n_rows=na, color=color)

        # Create demographic flows
        for var, name, color in zip(hpd.dem_keys, hpd.dem_names, hpd.dem_colors):
            results[var] = init_res(name, color=color)

        # Create results by sex
        for var, name, color in zip(hpd.by_sex_keys, hpd.by_sex_colors, hpd.by_sex_colors):
            results[var] = init_res(name, color=color, n_rows=2)

        # Create ASR results using standard populations
        results['asr_cancer_incidence'] = init_res('Age-adjusted cervical cancer incidence', scale=False)
        results['asr_cancer_mortality'] = init_res('Age-adjusted cervical cancer mortality', scale=False)

        stock_colors = [i for i in set(hpd.PeopleMeta().stock_colors) if i is not None]

        # Type distributions by cytology
        for var, name in zip(hpd.type_dist_keys, hpd.type_dist_names):
            results[var+'_genotype_dist'] = init_res(name, n_rows=ng, color=stock_colors[0])

        # Vaccination results
        results['new_vaccinated'] = init_res('Newly vaccinated')
        results['cum_vaccinated'] = init_res('Cumulative number vaccinated')
        results['new_doses'] = init_res('New doses')
        results['cum_doses'] = init_res('Cumulative doses')

        # Therapeutic vaccine results
        results['new_txvx_doses'] = init_res('New therapeutic vaccine doses')
        results['new_tx_vaccinated'] = init_res('Newly received therapeutic vaccine')
        results['cum_txvx_doses'] = init_res('Cumulative therapeutic vaccine doses')
        results['cum_tx_vaccinated'] = init_res('Total received therapeutic vaccine')

        # Screen & treat results
        results['new_screens'] = init_res('New screens')
        results['new_screened'] = init_res('Newly screened')
        results['new_cin_treatments'] = init_res('New CIN treatments')
        results['new_cin_treated'] = init_res('Newly treated for CINs')
        results['new_cancer_treatments'] = init_res('New cancer treatments')
        results['new_cancer_treated'] = init_res('Newly treated for cancer')
        results['cum_screens'] = init_res('Cumulative screens')
        results['cum_screened'] = init_res('Cumulative number screened')
        results['cum_cin_treatments'] = init_res('Cumulative CIN treatments')
        results['cum_cin_treated'] = init_res('Cumulative number treated for CINs')
        results['cum_cancer_treatments'] = init_res('Cumulative cancer treatments')
        results['cum_cancer_treated'] = init_res('Cumulative number treated for cancer')

        # Additional cancer results
        results['cancer_mortality'] = init_res('Cancer mortality')

        # Other results
        results['n_alive'] = init_res('Number alive')
        results['n_alive_by_sex'] = init_res('Number alive by sex', n_rows=2)
        results['n_alive_by_age'] = init_res('Number alive by age', n_rows=na)
        results['n_females_alive_by_age'] = init_res('Number females alive by age', n_rows=na)
        results['cdr'] = init_res('Crude death rate', scale=False)
        results['cbr'] = init_res('Crude birth rate', scale=False, color='#fcba03')
        results['hpv_prevalence'] = init_res('HPV prevalence', color=stock_colors[0])
        results['hpv_prevalence_by_genotype'] = init_res('HPV prevalence', n_rows=ng, color=stock_colors[0])
        results['hpv_prevalence_by_age'] = init_res('HPV prevalence by age', n_rows=na, color=stock_colors[0])
        results['precin_prevalence'] = init_res('Pre-CIN prevalence', color=stock_colors[0])
        results['precin_prevalence_by_genotype'] = init_res('Pre-CIN prevalence by genotype', n_rows=ng, color=stock_colors[0])
        results['precin_prevalence_by_age'] = init_res('Pre-CIN prevalence by age', n_rows=na, color=stock_colors[0])
        results['cin_prevalence'] = init_res('CIN prevalence', color=stock_colors[1])
        results['cin_prevalence_by_genotype'] = init_res('CIN prevalence by genotype', n_rows=ng, color=stock_colors[1])
        results['cin_prevalence_by_age'] = init_res('CIN prevalence by age', n_rows=na, color=stock_colors[1])
        results['female_hpv_prevalence_by_age'] = init_res('Female HPV prevalence by age', n_rows=na, color=stock_colors[3])

        # Time vector
        results['year'] = self.res_yearvec
        results['t'] = self.res_tvec

        # Final items
        self.results = results
        self.results_ready = False

        return


    def init_interventions(self):
        ''' Initialize and validate the interventions '''

        # Initialization
        self.interventions = sc.autolist()

        # Translate the intervention specs into actual interventions
        for i,intervention in enumerate(self['interventions']):
            if isinstance(intervention, type) and issubclass(intervention, hpi.Intervention):
                intervention = intervention() # Convert from a class to an instance of a class
            if isinstance(intervention, hpi.Intervention):
                intervention.initialize(self)
                self.interventions += intervention
            elif callable(intervention):
                self.interventions += intervention
            else:
                errormsg = f'Intervention {intervention} does not seem to be a valid intervention: must be a function or hpv.Intervention subclass'
                raise TypeError(errormsg)

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
            kwargs          (dict): passed to hpv.make_people()
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
            print(f'Initializing sim{resetstr} with {self["n_agents"]:0n} agents')
        if self.popfile and self.popdict is None: # If there's a popdict, we initialize it
            self.load_population(init_people=False) #TODO: no method for this

        # Make the people
        self.people, total_pop = hppop.make_people(self, reset=reset, verbose=verbose, microstructure=self['network'], **kwargs)

        # Figure out the scale factors
        # Case 1: total pop and location both provided
        if self['total_pop'] is not None and total_pop is not None: # If no pop_scale has been provided, try to get it from the location
            msg = f"Rescaling the population of the chosen location to {self['total_pop']}"
            if self['verbose']: print(msg)
            total_pop = self['total_pop']

        # Case 2: no location provided but total pop provided
        elif total_pop is None and self['total_pop'] is not None:
            total_pop = self['total_pop']
            
        # Case 3: neither total pop, location, nor pop scale provided
        if self['pop_scale'] is None:
            if total_pop is None:
                self['pop_scale'] = 1.0

        # Resolve cases 1 & 2 by creating the pop scal
            else:
                self['pop_scale'] = total_pop/self['n_agents']

        self['ms_agent_ratio'] = int(self['ms_agent_ratio'])

        # Deal with HIV
        self.init_hiv() # Creates the hivsim object, which is stored in the sim
        self.hivsim.init_states(self.people) # Adds some states to the people

        # Finish initialization
        self.people.initialize(sim_pars=self.pars) # Fully initialize the people
        self.people.hivsim = self.hivsim
        self.reset_layer_pars(force=False) # Ensure that layer keys match the loaded population
        if init_states:
            init_hpv_prev = sc.dcp(self['init_hpv_prev'])
            init_hpv_prev, age_brackets = self.validate_init_conditions(init_hpv_prev)
            self.init_states(age_brackets=age_brackets, init_hpv_prev=init_hpv_prev)

        return self

    def init_analyzers(self):
        ''' Initialize the analyzers '''

        self.analyzers = sc.autolist()

        def convert_analyzer(analyzer):
            ''' Helper function to turn strings into analyzers '''
            choices = hpa.analyzer_map.keys()
            if not analyzer in choices:
                errormsg = f'Analyzer {analyzer} not understood: choices are {choices}.'
                raise ValueError(errormsg)
            else:
                analyzer = hpa.analyzer_map[analyzer]
            return analyzer

        # Interpret analyzers
        for ai, analyzer in enumerate(self['analyzers']):
            if isinstance(analyzer, str):
                analyzer_list = sc.tolist(
                    convert_analyzer(analyzer))  # If not a list, turn it into one - for consistency of processing
                for az in analyzer_list:
                    if isinstance(az, str): az = convert_analyzer(az)  # It might still be a string
                    self.analyzers += az()  # Unpack list
            else:
                if isinstance(analyzer, type) and issubclass(analyzer, hpa.Analyzer):
                    analyzer = analyzer()  # Convert from a class to an instance of a class
                if not (isinstance(analyzer, hpa.Analyzer) or callable(analyzer)):
                    errormsg = f'Analyzer {analyzer} does not seem to be a valid analyzer: must be a function or hpv.Analyzer subclass'
                    raise TypeError(errormsg)
                self.analyzers += analyzer  # Add it in

        for analyzer in self.analyzers:
            if isinstance(analyzer, hpa.Analyzer):
                analyzer.initialize(self)

        return


    def init_immunity(self, create=True):
        ''' Initialize immunity matrices and cumulative dysplasia '''

        # Initialize immunity arrays
        hpimm.init_immunity(self, create=create)
        return


    def init_hiv(self):
        ''' Initialize states, attributes, and parameters relating to HIV '''
        if self.pars['model_hiv']:
            if self.hiv_datafile is None or self.art_datafile is None:
                raise ValueError('Must supply HIV and ART datafiles to model HIV.')
        self.hivsim = hphiv.HIVsim(self, hiv_datafile=self.hiv_datafile, art_datafile=self.art_datafile,
                                   hiv_pars=self['hiv_pars'])
        return


    def finalize_analyzers(self):
        for analyzer in self.analyzers:
            if isinstance(analyzer, hpa.Analyzer):
                analyzer.finalize(self)


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
        hpv_probs[self.people.f_inds] = init_hpv_prev['f'][age_inds[self.people.f_inds]]*self.pars['rel_init_prev']
        hpv_probs[self.people.m_inds] = init_hpv_prev['m'][age_inds[self.people.m_inds]]*self.pars['rel_init_prev']
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

        for g in range(ng):
            self.people.infect(inds=hpv_inds[genotypes==g], g=g, layer='seed_infection')

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
        mixing = self['mixing']
        layer_probs = self['layer_probs']
        f_cross_layer = self['f_cross_layer']
        m_cross_layer = self['m_cross_layer']
        acts = self['acts']
        dur_pship = self['dur_pship']
        age_act_pars = self['age_act_pars']
        trans = np.array([self['transf2m'],self['transm2f']]) # F2M first since that's the order things are done later
        year = self.yearvec[t]

        # Make HIV-related updates
        if self.pars['model_hiv']:
            self.hivsim.step(people=self.people, year=year)

        # Update demographics, states, and partnerships
        self.people.update_states_pre(t=t, year=year) # This also ages people, applies deaths, and generates new births
        people = self.people # Shorten
        people.dissolve_partnerships(t=t) # Dissolve partnerships
        tind = self.yearvec[t] - self['start']
        people.create_partnerships(tind, mixing, layer_probs, f_cross_layer, m_cross_layer, dur_pship, acts, age_act_pars)

        # Apply interventions
        for i,intervention in enumerate(self.interventions):
            intervention(self) # If it's a function, call it directly

        # Assign sus_imm values, i.e. the protection against infection based on prior immune history
        if self['use_waning']:
            inds = hpu.true(people.peak_imm.sum(axis=0)).astype(hpd.default_int)
            if len(inds):
                ss = people.t_imm_event[:, inds].shape
                t_since_boost = (t - people.t_imm_event[:,inds]).ravel()
                current_imm = imm_kin_pars[t_since_boost].reshape(ss) # Get people's current level of immunity
                people.nab_imm[:,inds] = current_imm*people.peak_imm[:,inds] # Set immunity relative to peak
        else:
            people.nab_imm[:] = people.peak_imm
        hpimm.check_immunity(people)

        # Shorten more variables
        gen_betas = np.array([g['rel_beta'] * beta for g in gen_pars.values()], dtype=hpd.default_float)
        sus_imm = people.sus_imm
        rel_sus = people.rel_sus
        rel_trans = people.rel_trans

        inf = people.infectious.copy() # calculate transmission based on infectiousness at start of timestep i.e. someone infected in one layer cannot transmit the infection via a different layer in the same timestep

        # Loop over layers
        for lkey, layer in people.contacts.items():

            sus = people.susceptible.copy() # for each layer, update who's still susceptible

            # Shorten variables
            f = layer['f']
            m = layer['m']
            acts = layer['acts'] * dt
            frac_acts, whole_acts = np.modf(acts)
            whole_acts = whole_acts.astype(hpd.default_int)
            effective_condoms = hpd.default_float(condoms[lkey] * eff_condoms)

            # Compute transmissions by genotype
            for g in range(ng):

                f_source_inds = (inf[g][f] & sus[g][m]).nonzero()[0]  # get female sources where female partner is infectious with genotype and male partner is susceptible to that genotype
                m_source_inds = (inf[g][m] & sus[g][f]).nonzero()[0]  # get male sources where the male partner is infectious with genotype and the female partner is susceptible to that genotype

                foi_frac = 1 - frac_acts * gen_betas[g] * trans[:, None] * (1 - effective_condoms)  # Probability of not getting infected from any fractional acts
                foi_whole = (1 - gen_betas[g] * trans[:, None] * (1 - effective_condoms)) ** whole_acts  # Probability of not getting infected from whole acts
                foi = (1 - (foi_whole * foi_frac)).astype(hpd.default_float)

                discordant_pairs = [[f_source_inds, f[f_source_inds], m[f_source_inds], foi[0,:]],
                                    [m_source_inds, m[m_source_inds], f[m_source_inds], foi[1,:]]]

                # Compute transmissibility for each partnership
                for pship_inds, sources, targets, this_foi in discordant_pairs:
                    betas = this_foi[pship_inds] * (1. - sus_imm[g,targets]) * rel_sus[targets] * rel_trans[sources]# Pull out the transmissibility associated with this partnership
                    transmissions = (np.random.random(len(betas)) < betas).nonzero()[0] # Apply probabilities to determine partnerships in which transmission occurred
                    target_inds   = targets[transmissions] # Extract indices of those who got infected
                    target_inds, unique_inds = np.unique(target_inds, return_index=True)  # Due to multiple partnerships, some people will be counted twice; remove them
                    people.infect(inds=target_inds, g=g, layer=lkey)  # Infect people

        # Determine if there are any reactivated infections on this timestep
        for g in range(ng):
            latent_inds = hpu.true(people.latent[g,:])
            if len(latent_inds):
                sev_imm = people.sev_imm[g, latent_inds]
                reactivation_probs = np.full_like(latent_inds, self['hpv_reactivation'] * dt, dtype=hpd.default_float)
                reactivation_probs *= (1 - sev_imm)
                if self['model_hiv']:
                    # determine if any of these inds have HIV and adjust their probs
                    hiv_latent_inds = latent_inds[hpu.true(people.hiv[latent_inds])]
                    if len(hiv_latent_inds):
                        immune_compromise = 1 - people.art_adherence[hiv_latent_inds]
                        mod = immune_compromise * self.hivsim['hiv_pars']['rel_reactivation_prob']
                        mod[mod < 1] = 1
                        reactivation_probs[hpu.true(people.hiv[latent_inds])] *= mod
                is_reactivated = hpu.binomial_arr(reactivation_probs)
                reactivated_inds = latent_inds[is_reactivated]
                people.infect(inds=reactivated_inds, g=g, layer='reactivation')

        # Updates after infection
        self.people.update_states_post(t=t, year=year)

        # Index for results
        idx = int(t / self.resfreq)

        # Update counts for this time step: flows
        for key,count in people.flows.items():
            self.results[key][idx] += count
        for key,count in people.demographic_flows.items():
            self.results[key][idx] += count
        for key,count in people.genotype_flows.items():
            flow_ind = [flow.name for flow in hpd.flows].index(key)
            if hpd.flows[flow_ind].by_genotype:
                for genotype in range(ng):
                    self.results[key+'_by_genotype'][genotype][idx] += count[genotype]
        for key,count in people.sex_flows.items():
            for sex in range(2):
                self.results[key][sex][idx] += count[sex]
        for key,count in people.age_flows.items():
            self.results[key+'_by_age'][:,idx] += count

        # Make stock updates every nth step, where n is the frequency of result output
        if t % self.resfreq == self.resfreq-1:

            # Number infectious/susceptible by age, for prevalence calculations
            f_inds = hpu.true(people['sex']==0)
            infinds = hpu.true(people['infectious'])
            f_infinds = np.intersect1d(f_inds, infinds)
            susinds = hpu.true(people['susceptible'])
            precininds = hpu.true(people['precin'])
            cininds = hpu.true(people['cin'])

            self.results['n_females_infectious_by_age'][:, idx] = np.histogram(people.age[f_infinds], bins=people.age_bin_edges, weights=people.scale[f_infinds])[0]
            self.results['n_infectious_by_age'][:, idx]  = np.histogram(people.age[infinds], bins=people.age_bin_edges, weights=people.scale[infinds])[0]
            self.results['n_susceptible_by_age'][:, idx] = np.histogram(people.age[susinds], bins=people.age_bin_edges, weights=people.scale[susinds])[0]
            self.results['n_precin_by_age'][:, idx] = np.histogram(people.age[precininds], bins=people.age_bin_edges, weights=people.scale[precininds])[0]
            self.results['n_cin_by_age'][:, idx] = np.histogram(people.age[cininds], bins=people.age_bin_edges, weights=people.scale[cininds])[0]

            # Create total stocks
            for key in self.people.meta.genotype_stock_keys:

                # Stocks by genotype
                for g in range(ng):
                    self.results[f'n_{key}_by_genotype'][g, idx] = people.count_by_genotype(key, g)

                # Total stocks
                if key not in ['susceptible']:
                    # For n_infectious etc, we get the total number where this state is true for at least one genotype
                    self.results[f'n_{key}'][idx] = people.count_any(key)
                elif key == 'susceptible':
                    # For n_total_susceptible, we get the total number of infections that could theoretically happen in the population, which can be greater than the population size
                    self.results[f'n_{key}'][idx] = people.count(key)

            # Create stocks of interventions
            for key in self.people.meta.intv_stock_keys:
                self.results[f'n_{key}'][idx] = people.count(key)

            # Update cancers and cancers by age
            cases_by_age = self.results['cancers_by_age'][:, idx]
            inds = people.alive * (self.people.sex==0) * ~people.cancerous.any(axis=0)
            vals = self.people.age[inds]
            bins = self.pars['standard_pop'][0,]
            weights = people.scale[inds]
            denom = np.histogram(vals, bins, weights=weights)[0]
            age_specific_incidence = sc.safedivide(cases_by_age, denom)*100e3
            age_specific_mortality = sc.safedivide(self.results['cancer_deaths_by_age'][:, idx], denom)*100e3
            standard_pop = self.pars['standard_pop'][1, :-1]
            self.results['asr_cancer_incidence'][idx] = np.dot(age_specific_incidence,standard_pop)
            self.results['asr_cancer_mortality'][idx] = np.dot(age_specific_mortality,standard_pop)

            # Save number alive
            alive_inds = hpu.true(people.alive)
            alive_female_inds = hpu.true(people.alive*people.is_female)
            self.results['n_alive'][idx] = people.scale_flows(alive_inds)
            self.results['n_alive_by_sex'][0,idx] = people.scale_flows((people.alive*people.is_female).nonzero()[0])
            self.results['n_alive_by_sex'][1,idx] = people.scale_flows((people.alive*people.is_male).nonzero()[0])
            self.results['n_alive_by_age'][:,idx] = np.histogram(people.age[alive_inds], bins=people.age_bin_edges, weights=people.scale[alive_inds])[0]
            self.results['n_females_alive_by_age'][:,idx] = np.histogram(people.age[alive_female_inds], bins=people.age_bin_edges, weights=people.scale[alive_female_inds])[0]

        # Apply analyzers
        for i,analyzer in enumerate(self.analyzers):
            analyzer(self)

        # Tidy up
        self.t += 1
        if self.t == self.npts:
            self.complete = True

        return


    def run(self, do_plot=False, until=None, restore_pars=True, reset_seed=True, verbose=None, **kwargs):
        ''' Run the model once '''
        # Initialization steps -- start the timer, initialize the sim and the seed, and check that the sim hasn't been run
        T = sc.timer()

        if not self.initialized:
            self.initialize(**kwargs)
            self._orig_pars = sc.dcp(self.pars) # Create a copy of the parameters, to restore after the run, in case they are dynamically modified

        if verbose is None:
            verbose = self['verbose']

        if reset_seed:
            # Reset the RNG. The primary use case (and why it defaults to True) is to ensure that
            #
            # >>> sim0.initialize()
            # >>> sim0.run()
            # >>> sim1.initialize()
            # >>> sim1.run()
            #
            # produces the same output as
            #
            # >>> sim0.initialize()
            # >>> sim1.initialize()
            # >>> sim0.run()
            # >>> sim1.run()
            #
            # The seed is offset by 1 to avoid drawing the same random numbers as those used for population generation, otherwise
            # the first set of random numbers in the model (e.g., deaths) will be correlated with the first set of random numbers
            # drawn in population generation (e.g., sex)
            hpu.set_seed(self['rand_seed']+1)

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
                string = f'  Running {simlabel}{self.yearvec[self.t]:0.1f} ({self.t:2.0f}/{self.npts}) ({elapsed:0.2f} s) '
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

        # Finalize analyzers and interventions
        self.finalize_analyzers()
        # self.finalize_interventions() #TODO: why is this commented out?

        if self['model_hiv']:
            self.hivsim.finalize(self)

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
        def safedivide(num,denom):
            ''' Define a variation on sc.safedivide that respects shape of numerator '''
            answer = np.zeros_like(num)
            fill_inds = (denom!=0).nonzero()
            if len(num.shape)==len(denom.shape):
                answer[fill_inds] = num[fill_inds] / denom[fill_inds]
            else:
                answer[:, fill_inds] = num[:, fill_inds] / denom[fill_inds]
            return answer
        ng = self.pars['n_genotypes']
        self.results['hpv_incidence'][:]                = safedivide(res['infections'][:], ng*res['n_susceptible'][:])
        self.results['hpv_incidence_by_genotype'][:]    = safedivide(res['infections_by_genotype'][:], res['n_susceptible_by_genotype'][:])
        self.results['hpv_incidence_by_age'][:]         = safedivide(res['infections_by_age'][:], res['n_susceptible_by_age'][:])
        self.results['hpv_prevalence'][:]               = safedivide(res['n_infectious'][:], res['n_alive'][:])
        self.results['hpv_prevalence_by_genotype'][:]   = safedivide(res['n_infectious_by_genotype'][:], res['n_alive'][:])
        self.results['hpv_prevalence_by_age'][:]        = safedivide(res['n_infectious_by_age'][:], res['n_alive_by_age'][:])

        alive_females = res['n_alive_by_sex'][0,:]

        self.results['female_hpv_prevalence_by_age'][:] = safedivide((res['n_females_infectious_by_age'][:]),
                                                                     res['n_females_alive_by_age'][:])

        self.results['precin_prevalence'][:] = safedivide(res['n_precin'][:], ng * alive_females)
        self.results['precin_prevalence_by_genotype'][:] = safedivide(res['n_precin_by_genotype'][:], alive_females)
        self.results['precin_prevalence_by_age'][:] = safedivide(res['n_precin_by_age'][:],
                                                               res['n_females_alive_by_age'][:])
        self.results['cin_prevalence'][:] = safedivide(res['n_cin'][:], alive_females)
        self.results['cin_prevalence_by_genotype'][:] = safedivide(res['n_cin_by_genotype'][:], alive_females)
        self.results['cin_prevalence_by_age'][:] = safedivide(res['n_cin_by_age'][:],
                                                               res['n_females_alive_by_age'][:])

        # Compute CIN and cancer incidence.
        demoninator = alive_females - res['n_cin'][:]
        self.results['cin_incidence'][:]             = res['cins'][:] / demoninator
        self.results['cin_incidence_by_genotype'][:] = res['cins_by_genotype'][:] / demoninator
        self.results['cin_incidence_by_age'][:]      = safedivide(res['cins_by_age'][:], res['n_females_alive_by_age'][:])

        at_risk_females = alive_females - res['n_cancerous'][:]
        scale_factor = 1e5  # Cancer incidence are displayed as rates per 100k women
        demoninator = at_risk_females / scale_factor
        self.results['cancer_incidence'][:]             = res['cancers'][:] / demoninator
        self.results['cancer_incidence_by_genotype'][:] = res['cancers_by_genotype'][:] / demoninator
        self.results['cancer_incidence_by_age'][:]      = safedivide(res['cancers_by_age'][:], res['n_females_alive_by_age'][:]/scale_factor)

        # Compute cancer mortality. Denominator is all women alive
        denominator = alive_females/scale_factor
        self.results['cancer_mortality'][:]         = res['cancer_deaths'][:]/denominator

        # Compute HPV type distribution by cytology
        for which in hpd.type_dist_keys:
            by_type = res[f'n_{which}_by_genotype'][:]
            totals = by_type.sum(axis=0)
            inds_to_fill = totals > 0
            res[which + '_genotype_dist'][:, inds_to_fill] = by_type[:, inds_to_fill] / totals[inds_to_fill]

        # Demographic results
        self.results['cdr'][:]  = self.results['other_deaths'][:] / (self.results['n_alive'][:])
        self.results['cbr'][:]  = self.results['births'][:] / (self.results['n_alive'][:])

        # Vaccination results
        self.results['cum_vaccinated'][:] = np.cumsum(self.results['new_vaccinated'][:], axis=0)
        self.results['cum_doses'][:] = np.cumsum(self.results['new_doses'][:])

        # Therapeutic vaccination results
        self.results['cum_tx_vaccinated'][:] = np.cumsum(self.results['new_tx_vaccinated'][:], axis=0)
        self.results['cum_txvx_doses'][:] = np.cumsum(self.results['new_txvx_doses'][:])

        # Screen & treat results
        self.results['cum_screens'][:] = np.cumsum(self.results['new_screens'][:], axis=0)
        self.results['cum_screened'][:] = np.cumsum(self.results['new_screened'][:], axis=0)
        self.results['cum_cin_treatments'][:] = np.cumsum(self.results['new_cin_treatments'][:], axis=0)
        self.results['cum_cin_treated'][:] = np.cumsum(self.results['new_cin_treated'][:], axis=0)
        self.results['cum_cancer_treatments'][:] = np.cumsum(self.results['new_cancer_treatments'][:], axis=0)
        self.results['cum_cancer_treated'][:] = np.cumsum(self.results['new_cancer_treatments'][:], axis=0)

        return
    
    
    def compute_age_mean(self, reskey, t=None):
        if t is None:
            t = -1
        assert 'by_age' in reskey, 'This method can only be used for age results'
        res = self.results[reskey][:,t]
        edges = self['age_bin_edges']
        mean_edges = edges[:-1] + np.diff(edges)/2
        age_mean = (sc.safedivide(res,res.sum())*mean_edges).sum()
        return age_mean
        

    def compute_summary(self, t=None, update=True, output=False, full=False, require_run=False):
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

        s = sc.odict()
        s['total HPV infections'] = self.results['infections'].sum()
        s['total cancers'] = self.results['cancers'].sum()
        s['total cancer deaths'] = self.results['cancer_deaths'].sum()
        s['mean HPV prevalence (%)'] = self.results['hpv_prevalence'].mean()*100
        s['mean cancer incidence (per 100k)'] = self.results['cancer_incidence'].mean()
        s['mean age of infection (years)'] = self.compute_age_mean('infections_by_age', t=t)
        s['mean age of cancer (years)'] = self.compute_age_mean('cancers_by_age', t=t)
        s['mean age of cancer death (years)'] = self.compute_age_mean('cancer_deaths_by_age', t=t)
        if self['model_hiv']:
            s['mean age of hiv mortality (years)'] = self.compute_age_mean('hiv_deaths_by_age', t=t)
        
        summary = sc.objdict()
        for key in self.result_keys('total'):
            summary[key] = self.results[key][t]

        # Update the stored state
        if update:
            self.short_summary = s
            self.summary = summary

        # Optionally return
        if output:
            if full:
                return summary
            else:
                return s
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

            sim = hpv.Sim(label='Example sim', verbose=0) # Set to run silently
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
        for label,val in summary.items():
            if 'total' in label:
                printval = f'   {val:13,.0f} '
            elif 'mean' in label:
                printval = f'   {val:13,.2f} '
            else:
                raise NotImplementedError
            string += printval + label + '\n'

        # Print or return string
        if not output:
            print(string)
        else:
            return string


    def plot(self, *args, **kwargs):
        '''
        Plot the outputs of the model
        '''
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
    :py:func:`Sim.run` and not taking any timesteps, would be an inadvertent error.
    '''
    pass


def diff_sims(sim1, sim2, skip_key_diffs=False, skip=None, full=False, output=False, die=False):
    '''
    Compute the difference of the summaries of two simulations, and print any
    values which differ.

    Args:
        sim1 (sim/dict): either a simulation object or the sim.summary dictionary
        sim2 (sim/dict): ditto
        skip_key_diffs (bool): whether to skip keys that don't match between sims
        skip (list): a list of values to skip
        full (bool): whether to use the full summary (else, brief)
        output (bool): whether to return the output as a string (otherwise print)
        die (bool): whether to raise an exception if the sims don't match
        require_run (bool): require that the simulations have been run

    **Example**::

        s1 = hpv.Sim(rand_seed=1).run()
        s2 = hpv.Sim(rand_seed=2).run()
        hpv.diff_sims(s1, s2)
    '''

    if isinstance(sim1, Sim):
        sim1 = sim1.compute_summary(update=False, output=True, require_run=True, full=full)
    if isinstance(sim2, Sim):
        sim2 = sim2.compute_summary(update=False, output=True, require_run=True, full=full)
    for sim in [sim1, sim2]:
        if not isinstance(sim, dict): # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a sim or a sim.summary dict'
            raise TypeError(errormsg)

    # Compare keys
    keymatchmsg = ''
    sim1_keys = set(sim1.keys())
    sim2_keys = set(sim2.keys())
    if sim1_keys != sim2_keys and not skip_key_diffs: # pragma: no cover
        keymatchmsg = "Keys don't match!\n"
        missing = list(sim1_keys - sim2_keys)
        extra   = list(sim2_keys - sim1_keys)
        if missing:
            keymatchmsg += f'  Missing sim1 keys: {missing}\ns'
        if extra:
            keymatchmsg += f'  Extra sim2 keys: {extra}\n'

    # Compare values
    valmatchmsg = ''
    mismatches = {}
    skip = sc.tolist(skip)
    for key in sim2.keys(): # To ensure order
        if key in sim1_keys and key not in skip: # If a key is missing, don't count it as a mismatch
            sim1_val = sim1[key] if key in sim1 else 'not present'
            sim2_val = sim2[key] if key in sim2 else 'not present'
            if not np.isclose(sim1_val, sim2_val, equal_nan=True):
                mismatches[key] = {'sim1': sim1_val, 'sim2': sim2_val}

    if len(mismatches):
        valmatchmsg = '\nThe following values differ between the two simulations:\n'
        df = pd.DataFrame.from_dict(mismatches).transpose()
        diff   = []
        ratio  = []
        change = []
        small_change = 1e-3 # Define a small change, e.g. a rounding error
        for mdict in mismatches.values():
            old = mdict['sim1']
            new = mdict['sim2']
            numeric = sc.isnumber(sim1_val) and sc.isnumber(sim2_val)
            if numeric and old>0:
                this_diff  = new - old
                this_ratio = new/old
                abs_ratio  = max(this_ratio, 1.0/this_ratio)

                # Set the character to use
                if abs_ratio<small_change:
                    change_char = '≈'
                elif new > old:
                    change_char = '↑'
                elif new < old:
                    change_char = '↓'
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio >= 1.1:
                    repeats = 2
                if abs_ratio >= 2:
                    repeats = 3
                if abs_ratio >= 10:
                    repeats = 4

                this_change = change_char*repeats
            else: # pragma: no cover
                this_diff   = np.nan
                this_ratio  = np.nan
                this_change = 'N/A'

            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)

        df['diff'] = diff
        df['ratio'] = ratio
        for col in ['sim1', 'sim2', 'diff', 'ratio']:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
        valmatchmsg += str(df)

    # Raise an error if mismatches were found
    mismatchmsg = keymatchmsg + valmatchmsg
    if mismatchmsg: # pragma: no cover
        if die:
            raise ValueError(mismatchmsg)
        elif output:
            return mismatchmsg
        else:
            print(mismatchmsg)
    else:
        if not output:
            print('Sims match')
    return
