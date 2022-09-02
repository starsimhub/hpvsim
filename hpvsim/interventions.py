'''
Specify the core interventions. Other interventions can be
defined by the user by inheriting from these classes.
'''

import numpy as np
import sciris as sc
import pylab as pl
import inspect
from . import defaults as hpd
from . import parameters as hppar
from . import utils as hpu
from . import immunity as hpi
from collections import defaultdict


#%% Helper functions

def find_day(arr, t=None, interv=None, sim=None, which='first'):
    '''
    Helper function to find if the current simulation time matches any day in the
    intervention. Although usually never more than one index is returned, it is
    returned as a list for the sake of easy iteration.

    Args:
        arr (list/function): list of timepoints in the intervention, or a boolean array; or a function that returns these
        t (int): current simulation time (can be None if a boolean array is used)
        which (str): what to return: 'first', 'last', or 'all' indices
        interv (intervention): the intervention object (usually self); only used if arr is callable
        sim (sim): the simulation object; only used if arr is callable

    Returns:
        inds (list): list of matching timepoints; length zero or one unless which is 'all'

    New in version 2.1.2: arr can be a function with arguments interv and sim.
    '''
    if callable(arr):
        arr = arr(interv, sim)
        arr = sc.promotetoarray(arr)
    all_inds = sc.findinds(arr=arr, val=t)
    if len(all_inds) == 0 or which == 'all':
        inds = all_inds
    elif which == 'first':
        inds = [all_inds[0]]
    elif which == 'last':
        inds = [all_inds[-1]]
    else: # pragma: no cover
        errormsg = f'Argument "which" must be "first", "last", or "all", not "{which}"'
        raise ValueError(errormsg)
    return inds


def get_subtargets(subtarget, sim):
    '''
    A small helper function to see if subtargeting is a list of indices to use,
    or a function that needs to be called. If a function, it must take a single
    argument, a sim object, and return a list of indices. Also validates the values.
    Currently designed for use with testing interventions, but could be generalized
    to other interventions. Not typically called directly by the user.

    Args:
        subtarget (dict): dict with keys 'inds' and 'vals'; see test_num() for examples of a valid subtarget dictionary
        sim (Sim): the simulation object
    '''

    # Validation
    if callable(subtarget):
        subtarget = subtarget(sim)

    if 'inds' not in subtarget: # pragma: no cover
        errormsg = f'The subtarget dict must have keys "inds" and "vals", but you supplied {subtarget}'
        raise ValueError(errormsg)

    # Handle the two options of type
    if callable(subtarget['inds']): # A function has been provided
        subtarget_inds = subtarget['inds'](sim) # Call the function to get the indices
    else:
        subtarget_inds = subtarget['inds'] # The indices are supplied directly

    # Validate the values
    if callable(subtarget['vals']): # A function has been provided
        subtarget_vals = subtarget['vals'](sim) # Call the function to get the indices
    else:
        subtarget_vals = subtarget['vals'] # The indices are supplied directly
    if sc.isiterable(subtarget_vals):
        if len(subtarget_vals) != len(subtarget_inds): # pragma: no cover
            errormsg = f'Length of subtargeting indices ({len(subtarget_inds)}) does not match length of values ({len(subtarget_vals)})'
            raise ValueError(errormsg)

    return subtarget_inds, subtarget_vals

#%% Generic intervention classes

__all__ = ['Intervention']

class Intervention:
    '''
    Base class for interventions.

    Args:
        label       (str): a label for the intervention (used for plotting, and for ease of identification)
        show_label (bool): whether or not to include the label in the legend
        do_plot    (bool): whether or not to plot the intervention
        line_args  (dict): arguments passed to pl.axvline() when plotting
    '''
    def __init__(self, label=None, show_label=False, do_plot=None, line_args=None, **kwargs):
        super().__init__(**kwargs)
        self._store_args() # Store the input arguments so the intervention can be recreated
        if label is None: label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Screen"
        self.show_label = show_label # Do not show the label by default
        self.do_plot = do_plot if do_plot is not None else True # Plot the intervention, including if None
        self.line_args = sc.mergedicts(dict(linestyle='--', c='#aaa', lw=1.0), line_args) # Do not set alpha by default due to the issue of overlapping interventions
        self.timepoints = [] # The start and end timepoints of the intervention
        self.initialized = False # Whether or not it has been initialized
        self.finalized = False # Whether or not it has been initialized
        return


    def __repr__(self, jsonify=False):
        ''' Return a JSON-friendly output if possible, else revert to short repr '''

        if self.__class__.__name__ in __all__ or jsonify:
            try:
                json = self.to_json()
                which = json['which']
                pars = json['pars']
                parstr = ', '.join([f'{k}={v}' for k,v in pars.items()])
                output = f"cv.{which}({parstr})"
            except Exception as E:
                output = f'{type(self)} (error: {str(E)})' # If that fails, print why
            return output
        else:
            return f'{self.__module__}.{self.__class__.__name__}()'


    def __call__(self, *args, **kwargs):
        # Makes Intervention(sim) equivalent to Intervention.apply(sim)
        if not self.initialized:  # pragma: no cover
            errormsg = f'Intervention (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)


    def disp(self):
        ''' Print a detailed representation of the intervention '''
        return sc.pr(self)


    def _store_args(self):
        ''' Store the user-supplied arguments for later use in to_json '''
        f0 = inspect.currentframe() # This "frame", i.e. Intervention.__init__()
        f1 = inspect.getouterframes(f0) # The list of outer frames
        parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
        _,_,_,values = inspect.getargvalues(parent) # Get the values of the arguments
        if values:
            self.input_args = {}
            for key,value in values.items():
                if key == 'kwargs': # Store additional kwargs directly
                    for k2,v2 in value.items(): # pragma: no cover
                        self.input_args[k2] = v2 # These are already a dict
                elif key not in ['self', '__class__']: # Everything else, but skip these
                    self.input_args[key] = value
        return


    def initialize(self, sim=None):
        '''
        Initialize intervention -- this is used to make modifications to the intervention
        that can't be done until after the sim is created.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize intervention

        This method is run once as part of `sim.finalize()` enabling the intervention to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized: # pragma: no cover
            raise RuntimeError('Intervention already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply the intervention. This is the core method which each derived intervention
        class must implement. This method gets called at each timestep and can make
        arbitrary changes to the Sim object, as well as storing or modifying the
        state of the intervention.

        Args:
            sim: the Sim instance

        Returns:
            None
        '''
        raise NotImplementedError


    def shrink(self, in_place=False):
        '''
        Remove any excess stored data from the intervention; for use with sim.shrink().

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place: # pragma: no cover
            return self
        else:
            return sc.dcp(self)


    def plot_intervention(self, sim, ax=None, **kwargs):
        '''
        Plot the intervention

        This can be used to do things like add vertical lines at timepoints when
        interventions take place. Can be disabled by setting self.do_plot=False.

        Note 1: you can modify the plotting style via the ``line_args`` argument when
        creating the intervention.

        Note 2: By default, the intervention is plotted at the timepoints stored in self.timepoints.
        However, if there is a self.plot_timepoints attribute, this will be used instead.

        Args:
            sim: the Sim instance
            ax: the axis instance
            kwargs: passed to ax.axvline()

        Returns:
            None
        '''
        line_args = sc.mergedicts(self.line_args, kwargs)
        if self.do_plot or self.do_plot is None:
            if ax is None:
                ax = pl.gca()
            if hasattr(self, 'plot_timepoints'):
                timepoints = self.plot_timepoints
            else:
                timepoints = self.timepoints
            if sc.isiterable(timepoints):
                label_shown = False # Don't show the label more than once
                for timepoint in timepoints:
                    if sc.isnumber(timepoint):
                        if self.show_label and not label_shown: # Choose whether to include the label in the legend
                            label = self.label
                            label_shown = True
                        else:
                            label = None
                        date = sim.yearvec[timepoint]
                        ax.axvline(date, label=label, **line_args)
        return


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those.

        Note that simply printing an intervention will usually return a representation
        that can be used to recreate it.

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = sc.jsonify(self.input_args)
        output = dict(which=which, pars=pars)
        return output



#%% Behavior change interventions
__all__ += ['dynamic_pars']

class dynamic_pars(Intervention):
    '''
    A generic intervention that modifies a set of parameters at specified points
    in time.

    The intervention takes a single argument, pars, which is a dictionary of which
    parameters to change, with following structure: keys are the parameters to change,
    then subkeys 'days' and 'vals' are either a scalar or list of when the change(s)
    should take effect and what the new value should be, respectively.

    You can also pass parameters to change directly as keyword arguments.

    Args:
        pars (dict): described above
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = hp.dynamic_pars(condoms=dict(timepoints=10, vals={'c':0.9})) # Increase condom use amount casual partners to 90%
        interv = hp.dynamic_pars({'beta':{'timepoints':[10, 15], 'vals':[0.005, 0.015]}, # At timepoint 10, reduce beta, then increase it again
                                  'debut':{'timepoints':10, 'vals':dict(f=dict(dist='normal', par1=20, par2=2.1), m=dict(dist='normal', par1=19.6, par2=1.8))}}) # Increase mean age of sexual debut
    '''

    def __init__(self, pars=None, **kwargs):

        # Find valid sim parameters and move matching keyword arguments to the pars dict
        pars = sc.mergedicts(pars) # Ensure it's a dictionary
        sim_par_keys = list(hppar.make_pars().keys()) # Get valid sim parameters
        kwarg_keys = [k for k in kwargs.keys() if k in sim_par_keys]
        for kkey in kwarg_keys:
            pars[kkey] = kwargs.pop(kkey)

        # Do standard initialization
        super().__init__(**kwargs) # Initialize the Intervention object

        # Handle the rest of the initialization
        subkeys = ['timepoints', 'vals']
        for parkey in pars.keys():
            for subkey in subkeys:
                if subkey not in pars[parkey].keys(): # pragma: no cover
                    errormsg = f'Parameter {parkey} is missing subkey {subkey}'
                    raise sc.KeyNotFoundError(errormsg)
                if sc.isnumber(pars[parkey][subkey]):
                    pars[parkey][subkey] = sc.promotetoarray(pars[parkey][subkey])
                else:
                    pars[parkey][subkey] = sc.promotetolist(pars[parkey][subkey])
            # timepoints = pars[parkey]['timepoints']
            # vals = pars[parkey]['vals']
            # if sc.isiterable(timepoints):
            #     len_timepoints = len(timepoints)
            #     len_vals = len(vals)
            #     if len_timepoints != len_vals:
            #         raise ValueError(f'Length of timepoints ({len_timepoints}) does not match length of values ({len_vals}) for parameter {parkey}')
        self.pars = pars

        return

    def initialize(self, sim):
        ''' Initialize with a sim '''
        for parkey in self.pars.keys():
            try: # First try to interpret the timepoints as dates
                tps = sim.get_t(self.pars[parkey]['timepoints'])  # Translate input to timepoints
            except:
                tps = []
                # See if it's in the time vector
                for tp in self.pars[parkey]['timepoints']:
                    if tp in sim.tvec:
                        tps.append(tp)
                    else: # Give up
                        errormsg = f'Could not parse timepoints provided for {parkey}.'
                        raise ValueError(errormsg)
            self.pars[parkey]['processed_timepoints'] = sc.promotetoarray(tps)
        self.initialized = True
        return


    def apply(self, sim):
        ''' Loop over the parameters, and then loop over the timepoints, applying them if any are found '''
        t = sim.t
        for parkey,parval in self.pars.items():
            if t in parval['processed_timepoints']: # TODO: make this more robust
                self.timepoints.append(t)
                ind = sc.findinds(parval['processed_timepoints'], t)[0]
                val = parval['vals'][ind]
                if isinstance(val, dict):
                    sim[parkey].update(val) # Set the parameter if a nested dict
                else:
                    sim[parkey] = val # Set the parameter if not a dict
        return


#%% Vaccination
__all__ += ['BaseVaccination', 'vaccinate_prob', 'vaccinate_routine', 'vaccinate_num']

class BaseVaccination(Intervention):
    '''
    Apply a vaccine to a subset of the population.

    This base class implements the mechanism of vaccinating people to modify their immunity.
    It does not implement allocation of the vaccines, which is implemented by derived classes
    such as `hpv.vaccinate_num`.

    Some quantities are tracked during execution for reporting after running the simulation.
    These are:

        - ``doses``:             the number of vaccine doses per person

    Args:
        vaccine (dict/str) : which vaccine to use; see below for dict parameters
        label   (str)      : if vaccine is supplied as a dict, the name of the vaccine
        kwargs  (dict)     : passed to Intervention()

    If ``vaccine`` is supplied as a dictionary, it must have the following parameters:

        - ``imm_init``:  the initial immunity level (higher = more protection)

    See :py:mod:`parameters` for additional examples of these parameters.

    '''

    def __init__(self, vaccine, label=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.index = None # Index of the vaccine in the sim; set later
        self.label = label # Vaccine label (used as a dict key)
        self.p     = None # Vaccine parameters
        self.immunity = None # Record the immunity conferred by this vaccine to each of the genotypes in the sim
        self.immunity_inds = None # Record the indices of genotypes that are targeted by this vaccine
        self._parse_vaccine_pars(vaccine=vaccine) # Populate
        return


    def _parse_vaccine_pars(self, vaccine=None):
        ''' Unpack vaccine information, which may be given as a string or dict '''

        # Option 1: vaccines can be chosen from a list of pre-defined vaccines
        if isinstance(vaccine, str):

            choices, mapping = hppar.get_vaccine_choices()
            genotype_pars = hppar.get_vaccine_genotype_pars()
            dose_pars = hppar.get_vaccine_dose_pars()

            label = vaccine.lower()
            for txt in ['.', ' ', '&', '-', 'vaccine']:
                label = label.replace(txt, '')

            if label in mapping:
                label = mapping[label]
                vaccine_pars = sc.mergedicts(genotype_pars[label], dose_pars[label])

            else: # pragma: no cover
                errormsg = f'The selected vaccine "{vaccine}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

            if self.label is None:
                self.label = label

        # Option 2: vaccines can be specified as a dict of pars
        elif isinstance(vaccine, dict):

            # Parse label
            vaccine_pars = vaccine
            label = vaccine_pars.pop('label', None) # Allow including the label in the parameters
            if self.label is None: # pragma: no cover
                if label is None:
                    self.label = 'custom'
                else:
                    self.label = label

        else: # pragma: no cover
            errormsg = f'Could not understand {type(vaccine)}, please specify as a string indexing a predefined vaccine or a dict.'
            raise ValueError(errormsg)

        # Set label and parameters
        self.p = sc.objdict(vaccine_pars)

        return


    def initialize(self, sim):
        super().initialize()

        # Populate any missing keys -- must be here, after genotypes are initialized
        default_genotype_pars   = hppar.get_vaccine_genotype_pars(default=True)
        default_dose_pars       = hppar.get_vaccine_dose_pars(default=True)
        genotype_labels         = list(sim['genotype_pars'].keys())
        dose_keys               = list(default_dose_pars.keys())

        # Handle dose keys
        for key in dose_keys:
            if key not in self.p:
                self.p[key] = default_dose_pars[key]

        # Set immunity to each genotype in the sim
        self.immunity = np.array([self.p[k] for k in genotype_labels])
        self.immunity_inds = hpu.true(self.immunity)

        for key in genotype_labels:
            if key not in self.p:
                if key in default_genotype_pars:
                    val = default_genotype_pars[key]
                else: # pragma: no cover
                    val = 1.0
                    if sim['verbose']: print(f'Note: No cross-immunity specified for vaccine {self.label} and genotype {key}, setting to 1.0')
                self.p[key] = val

        sim['vaccine_pars'][self.label] = self.p # Store the parameters
        self.index = list(sim['vaccine_pars'].keys()).index(self.label) # Find where we are in the list
        sim['vaccine_map'][self.index]  = self.label # Use that to populate the reverse mapping

        # Prepare to update sim['immunity']
        n_vax = self.index+1
        n_imm_sources = n_vax + len(sim['genotype_map'])
        immunity = sim['immunity']

        # add this vaccine to the immunity map
        sim['immunity_map'][n_imm_sources-1] = 'vaccine'
        if n_imm_sources > len(immunity): # need to add this vaccine, otherwise it's a duplicate
            vacc_mapping = [self.p[label] for label in sim['genotype_map'].values()]
            for _ in range(n_vax):
                vacc_mapping.append(1)
            vacc_mapping = np.reshape(vacc_mapping, (n_imm_sources, 1)).astype(hpd.default_float)
            immunity = np.hstack((immunity, vacc_mapping[0:len(immunity),]))
            immunity = np.vstack((immunity, np.transpose(vacc_mapping)))
            sim['immunity'] = immunity
            sc.promotetolist(sim['imm_boost']).append(sc.promotetolist(self.p['imm_boost'])) # This line happens in-place
            # sim.people.set_pars(sim.pars)

        return


    def finalize(self, sim):
        ''' Ensure variables with large memory footprints get erased '''
        super().finalize()
        self.subtarget = None # Reset to save memory
        return


    def select_people(self, sim):
        """
        Return an array of indices of people to vaccinate
        Derived classes must implement this function to determine who to vaccinate at each timestep
        Args:
            sim: A cv.Sim instance
        Returns: Array of person indices
        """
        raise NotImplementedError


    def vaccinate(self, sim, vacc_inds):
        '''
        Vaccinate people

        This method applies the vaccine to the requested people indices. The indices of people vaccinated
        is returned. These may be different to the requested indices, because anyone that is dead will be
        skipped, as well as anyone already fully vaccinated (if booster=False). This could
        occur if a derived class does not filter out such people in its `select_people` method.

        Args:
            sim: A cv.Sim instance
            vacc_inds: An array of person indices to vaccinate

        Returns: An array of person indices of people vaccinated
        '''


        # Perform checks
        vacc_inds = vacc_inds[sim.people.alive[vacc_inds]] # Skip anyone that is dead
        # Skip anyone that has already had all the doses of *this* vaccine (not counting boosters).
        # Otherwise, they will receive the 2nd dose boost cumulatively for every subsequent dose.
        # Note, this does not preclude someone from getting additional doses of another vaccine (e.g. a booster)
        vacc_inds = vacc_inds[sim.people.doses[vacc_inds] < self.p['doses']]
        first_vacc_inds = vacc_inds[~sim.people.vaccinated[vacc_inds]]

        if len(vacc_inds):
            sim.people.vaccinated[first_vacc_inds] = True #
            sim.people.vaccine_source[first_vacc_inds] = self.index
            sim.people.doses[vacc_inds] += 1
            sim.people.date_vaccinated[vacc_inds] = sim.t
            imm_source = len(sim['genotype_map']) + self.index
            hpi.update_peak_immunity(sim.people, vacc_inds, self.p, imm_source, infection=False)

            idx = int(sim.t / sim.resfreq)
            sim.results['new_vaccinated'][self.immunity_inds, idx] += len(first_vacc_inds)
            sim.results['new_total_vaccinated'][idx] += len(first_vacc_inds)
            sim.results['new_doses'][idx] += len(vacc_inds)

        return vacc_inds


    def apply(self, sim):
        ''' Perform vaccination each timestep '''
        inds = self.select_people(sim)
        if len(inds):
            inds = self.vaccinate(sim, inds)
        return inds


    def shrink(self, in_place=True):
        ''' Shrink vaccination intervention '''
        obj = super().shrink(in_place=in_place)
        obj.vaccinated = None
        obj.doses = None
        if hasattr(obj, 'second_dose_days'):
            obj.second_dose_days = None
        return obj


def check_doses(doses, interval, imm_boost):
    ''' Check that doses, intervals, and boost factors are supplied in correct formats '''

    # First check types
    if interval is not None:
        if sc.checktype(interval, 'num'):
            interval = sc.promotetolist(interval)
        if sc.checktype(imm_boost, 'num'):
            imm_boost = sc.promotetolist(imm_boost)

    if not sc.checktype(doses, int):
        raise ValueError(f'Doses must be an integer or array/list of integers, not {doses}.')

    # Now check that they're compatible
    if doses == 1 and ((interval is not None) or (imm_boost is not None)):
        raise ValueError("Can't use dosing intervals or boosting factors for vaccines with only one dose.")
    elif doses > 1:
        if interval is None or imm_boost is None:
            raise ValueError('Must specify a dosing interval and boosting factor if using a vaccine with more than one dose.')
        elif (len(interval) != doses-1) or (len(interval) != len(imm_boost)):
            raise ValueError(f'Dosing interval and imm_boost must both be length {doses-1}, not {len(interval)} and {len(imm_boost)}.')

    return doses, interval, imm_boost


class vaccinate_prob(BaseVaccination):
    '''
    Probability-based vaccination

    This vaccine intervention allocates vaccines parametrized by the daily probability
    of being vaccinated.

    Args:
        vaccine (dict/str): which vaccine to use; see below for dict parameters
        label        (str): if vaccine is supplied as a dict, the name of the vaccine
        timepoints   (int/arr): the year or array of timepoints to apply the interventions
        prob       (float): probability of being vaccinated (i.e., fraction of the population)
        subtarget   (dict): subtarget intervention to people with particular indices
        kwargs      (dict): passed to Intervention()

    **Example**::

        bivalent = hpv.vaccinate_prob(vaccine='bivalent', timepoints='2020', prob=0.7)
        hpv.Sim(interventions=bivalent).run().plot()
    '''
    def __init__(self, vaccine, timepoints, label=None, prob=None, subtarget=None, **kwargs) -> object:
        super().__init__(vaccine,label=label,**kwargs) # Initialize the Intervention object
        if prob is None: # Populate default value of probability: 1 if no subtargeting, 0 if subtargeting
            prob = 1.0 if subtarget is None else 0.0
        self.prob      = prob
        self.subtarget = subtarget
        self.timepoints = timepoints
        self.dates = None  # Representations in terms of years, e.g. 2020.4, set during initialization
        self.second_dose_days = None  # Track scheduled second doses
        return


    def initialize(self, sim):
        super().initialize(sim)
        self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format
        self.second_dose_timepoints = [None]*sim.npts # People who get second dose (if relevant)
        self.third_dose_timepoints  = [None]*sim.npts # People who get second dose (if relevant)
        self.p['doses'], self.p['interval'], self.p['imm_boost'] = check_doses(self.p['doses'], self.p['interval'], self.p['imm_boost'])
        return


    def select_people(self, sim):

        vacc_inds = np.array([], dtype=int)  # Initialize in case no one gets their first dose

        if sim.t >= np.min(self.timepoints):

            # Vaccinate people with their first dose
            for _ in find_day(self.timepoints, sim.t, interv=self, sim=sim):

                vacc_probs = np.zeros(len(sim.people))

                # Find eligible people
                vacc_probs[hpu.true(~sim.people.alive)] *= 0.0  # Do not vaccinate dead people
                eligible_inds = sc.findinds(~sim.people.vaccinated)
                vacc_probs[eligible_inds] = self.prob  # Assign equal vaccination probability to everyone

                # Apply any subtargeting
                if self.subtarget is not None:
                    subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
                    vacc_probs[subtarget_inds] = subtarget_vals  # People being explicitly subtargeted

                vacc_inds = hpu.true(hpu.binomial_arr(vacc_probs))  # Calculate who actually gets vaccinated

                if len(vacc_inds):
                    if self.p.interval is not None:
                        # Schedule the doses
                        second_dose_timepoints = sim.t + int(self.p.interval[0]/sim['dt'])
                        if second_dose_timepoints < sim.npts:
                            self.second_dose_timepoints[second_dose_timepoints] = vacc_inds
                        if self.p.doses==3:
                            third_dose_timepoints = sim.t + int(self.p.interval[1] / sim['dt'])
                            if third_dose_timepoints < sim.npts:
                                self.third_dose_timepoints[third_dose_timepoints] = vacc_inds

            # Also, if appropriate, vaccinate people with their second and third doses
            vacc_inds_dose2 = self.second_dose_timepoints[sim.t]
            vacc_inds_dose3 = self.third_dose_timepoints[sim.t]
            if vacc_inds_dose2 is not None:
                vacc_inds = np.concatenate((vacc_inds, vacc_inds_dose2), axis=None)
            if vacc_inds_dose3 is not None:
                vacc_inds = np.concatenate((vacc_inds, vacc_inds_dose3), axis=None)

        return vacc_inds


class vaccinate_routine(vaccinate_prob):

    def __init__(self, *args, age_range, coverage, sex=0, **kwargs):
        super().__init__(*args, **kwargs, subtarget=self.subtarget_function)
        self.age_range = age_range
        self.coverage = sc.promotetoarray(coverage)
        if len(self.coverage) == 1:
            self.coverage = self.coverage * np.ones_like(self.timepoints)

        # Deal with sex
        if sc.checktype(sex,'listlike'):
            if sc.checktype(sex[0],'str'): # If provided as 'f'/'m', convert to 0/1
                self.sex = np.array([0,1])
        else:
            self.sex = sc.promotetoarray(sex)


    def subtarget_function(self, sim):
        conditions = (sim.people.age >= self.age_range[0]) & (sim.people.age <self.age_range[1])
        if len(self.sex)==1: conditions = conditions & (sim.people.sex == self.sex[0])
        inds = sc.findinds(conditions)
        coverage = self.coverage[self.timepoints==sim.t][0]
        return {'vals': coverage*np.ones_like(inds), 'inds': inds}


class vaccinate_num(BaseVaccination):
    '''
    This vaccine intervention allocates vaccines in a pre-computed order of
    distribution, at a specified rate of doses per day.

    Args:
        vaccine (dict/str): which vaccine to use; see below for dict parameters
        label        (str): if vaccine is supplied as a dict, the name of the vaccine
        subtarget  (dict): subtarget intervention to people with particular indices
        num_doses: Specify the number of doses per timepoint. This can take three forms

            - A scalar number of doses per timepoint
            - A dict keyed by year/date with the number of doses e.g. ``{2010:10000, '2021-05-01':20000}``.
              Any dates are converted to simulation days in `initialize()` which will also copy the
              dictionary passed in.
            - A callable that takes in a ``hpv.Sim`` and returns a scalar number of doses. For example,
              ``def doses(sim): return 100 if sim.t > 10 else 0`` would be suitable

        **kwargs: Additional arguments passed to ``hpv.BaseVaccination``

    **Example**::

        bivalent = hpv.vaccinate_num(vaccine='bivalent', num_doses=1e6, timepoints=2020)
        hpv.Sim(interventions=bivalent).run().plot()
    '''

    def __init__(self, vaccine, num_doses, timepoints=None, dates=None, subtarget=None, spread_doses=False, **kwargs):
        super().__init__(vaccine, **kwargs)  # Initialize the Intervention object
        self.num_doses = num_doses
        self.timepoints = timepoints
        self.dates = dates
        self.subtarget = subtarget
        self.spread_doses = spread_doses
        self._scheduled_second_doses = defaultdict(set)
        self._scheduled_third_doses = defaultdict(set)
        return


    def initialize(self, sim):

        super().initialize(sim)

        # Firstly, translate the timepoints and dates to consistent formats
        if self.timepoints is None:
            if self.dates is not None: # Try to get timepoints from dates, if supplied
                self.timepoints, self.dates = sim.get_t(self.dates, return_date_format='str')
            else: # Otherwise, use all timepoints in the sim
                self.timepoints = sim.tvec
                self.dates = np.array([str(date) for date in sim.yearvec])
        else: # If timepoints have been supplied, use them
            self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format

        # Spread doses over the year
        if self.spread_doses:
            # Calculate these, but don't set them yet
            full_timepoints = [tp + dt for tp in self.timepoints for dt in range(int(1 / sim['dt']))]
            full_dates = [str(date) for date in sim.yearvec[full_timepoints]]
        else:
            full_timepoints = self.timepoints
            full_dates = self.dates

        # Check consistency of doses and timepoints
        if sc.checktype(self.num_doses, 'num'):
            if not sc.checktype(self.timepoints, int) and len(self.timepoints)>1:
                # If doses is a single entry, assume it applies to all timepoints
                if self.spread_doses:   num_doses = self.num_doses*sim['dt']
                else:                   num_doses = self.num_doses
                new_num_doses = {float(date):num_doses for date in full_dates}
                self.num_doses = new_num_doses

        elif sc.checktype(self.num_doses, 'listlike'):
            if not sc.checktype(self.timepoints, int) and len(self.timepoints)>1:
                if len(self.timepoints) != len(self.num_doses): # Check consistency
                    raise ValueError(f'Inconsistent lengths of num_doses and timepoints: {len(self.num_doses)} vs. {len(self.timepoints)}.')
                else:
                    if self.spread_doses:
                        num_doses = self.num_doses * sim['dt']
                        new_num_doses = {float(date): num_doses for date in full_dates}
                        self.num_doses = new_num_doses

        self.timepoints = np.array(full_timepoints)
        self.dates = np.array(full_dates)

        # Perform checks and process inputs
        if isinstance(self.num_doses, dict):  # Convert any dates to simulation days
            self.num_doses = {sim.get_t(k)[0]: v for k, v in self.num_doses.items()}
        self.p['doses'], self.p['interval'], self.p['imm_boost'] = check_doses(self.p['doses'], self.p['interval'], self.p['imm_boost'])

        return


    def select_people(self, sim):

        # Work out how many people to vaccinate today
        if sim.t in self.num_doses: num_people = self.num_doses[sim.t]
        else:                       num_people = 0

        if num_people == 0:
            self._scheduled_third_doses[sim.t + 1].update(self._scheduled_third_doses[sim.t])  # Defer any extras til the next timestep
            self._scheduled_second_doses[sim.t + 1].update(self._scheduled_second_doses[sim.t])  # Defer any extras til the next timestep
            return np.array([])

        num_agents = sc.randround(num_people / sim['pop_scale'])

        # First, see how many scheduled second/third doses we are going to deliver
        if self._scheduled_third_doses[sim.t]:
            scheduled_third = np.fromiter(self._scheduled_third_doses[sim.t], dtype=hpd.default_int)  # Everyone scheduled today
            still_alive = ~sim.people.dead_other[scheduled_third] & ~sim.people.dead_cancer[:, scheduled_third].sum(axis=0).astype(bool)
            scheduled_third = scheduled_third[(sim.people.doses[scheduled_third] == 2) & still_alive]  # Remove anyone who's already had all doses of this vaccine, also dead people

            # If there are more people due for a second/third dose than there are doses, vaccinate as many
            # as possible, and add the remainder to the next time step's doses.
            if len(scheduled_third) > num_agents:
                np.random.shuffle(scheduled_third)  # Randomly pick who to defer
                self._scheduled_third_doses[sim.t + 1].update(scheduled_third[num_agents:])  # Defer any extras
                return scheduled_third[:num_agents]
        else:
            scheduled_third = np.array([], dtype=hpd.default_int)

        if self._scheduled_second_doses[sim.t]:
            scheduled_second = np.fromiter(self._scheduled_second_doses[sim.t], dtype=hpd.default_int)  # Everyone scheduled today
            still_alive = ~sim.people.dead_other[scheduled_second] & ~sim.people.dead_cancer[scheduled_second].sum(axis=0).astype(bool)
            scheduled_second = scheduled_second[(sim.people.doses[scheduled_second] == 1) & still_alive]  # Remove anyone who's already had all doses of this vaccine, also dead people

            # If there are more people due for a second/third dose than there are doses, vaccinate as many
            # as possible, and add the remainder to the next time step's doses.
            if (len(scheduled_second)+len(scheduled_third)) > num_agents:
                scheduled_second = scheduled_second[:(num_agents - len(scheduled_third))]
                self._scheduled_second_doses[sim.t + 1].update(scheduled_second[(num_agents - len(scheduled_third)):])  # Defer any extras
                scheduled = np.concatenate([scheduled_third, scheduled_second[:(num_agents - len(scheduled_third))]])
                return scheduled
            else:
                scheduled = np.concatenate([scheduled_third, scheduled_second])
        else:
            scheduled = np.array([], dtype=hpd.default_int)

        # Next, work out who is eligible for their first dose
        vacc_probs = np.ones(sim.n)  # Begin by assigning equal weight (converted to a probability) to everyone
        vacc_probs[~sim.people.alive] = 0.0  # Dead people are not eligible

        # Apply any subtargeting for this vaccination
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            vacc_probs[subtarget_inds] = vacc_probs[subtarget_inds] * subtarget_vals

        # Exclude vaccinated people
        vacc_probs[sim.people.vaccinated] = 0.0  # Anyone who's received at least one dose is counted as vaccinated

        # All remaining people can be vaccinated, although anyone who has received half of a multi-dose
        # vaccine would have had subsequent doses scheduled and therefore should not be selected here
        first_dose_eligible = hpu.binomial_arr(vacc_probs)
        first_dose_eligible_inds = hpu.true(first_dose_eligible)

        if len(first_dose_eligible_inds) == 0:
            return scheduled  # Just return anyone that is scheduled

        elif len(first_dose_eligible_inds) > num_agents:
            # Truncate it to the number of agents for performance when checking whether anyone scheduled overlaps with first doses to allocate
            first_dose_eligible_inds = first_dose_eligible_inds[:num_agents]  # This is the maximum number of people we could vaccinate this timestep, if there are no second doses allocated

        # It's *possible* that someone has been *scheduled* for a first dose by some other mechanism externally
        # Therefore, we need to check and remove them from the first dose list, otherwise they could be vaccinated
        # twice here (which would amount to wasting a dose)
        first_dose_eligible_inds = first_dose_eligible_inds[~np.in1d(first_dose_eligible_inds, scheduled)]

        if (len(first_dose_eligible_inds) + len(scheduled)) > num_agents:
            first_dose_inds = first_dose_eligible_inds[:(num_agents - len(scheduled))]
        else:
            first_dose_inds = first_dose_eligible_inds

        # Schedule subsequent doses
        if self.p['doses'] > 1:
            self._scheduled_second_doses[int(sim.t + np.ceil(self.p.interval[0]/sim['dt']))].update(first_dose_inds)
        if self.p['doses'] > 2:
            self._scheduled_third_doses[int(sim.t + np.ceil(self.p.interval[1]/sim['dt']))].update(first_dose_inds)

        vacc_inds = np.concatenate([scheduled, first_dose_inds])

        return vacc_inds


#%% Screening
__all__ += ['Screening']

class Screening(Intervention):
    '''
    Screen/triage a subset of the population.

    This base class implements the mechanism of screening people to identify pre-cancerous lesions.
    Screening involves a series of standard operations to modify the trajectories of `hpv.People`. Screening algorithms
    can vary in complexity along the dimensions of primary screening modalities, triage modalities,
    interval between screens and follow-up protocol, loss-to-follow-up, test characteristics, and efficacies.

    Args:
         primary_screen_test (dict/str)  : the screening test to use as a primary filtering method
         triage_screen_test  (dict/str)  : the screening test to use as a triage (or None)
         treatment_pathway   (Product)   : optionally specify a treatment pathway to administer treatment
         screen_start_age    (int)       : age to start screening
         screen_interval     (int)       : interval between screens
         screen_stop_age     (int)       : age to stop screening
         screen_start_year   (str)       : the year to start screening intervention
         screen_end_year     (str)       : the year to end screening intervention (if None, assume continues until end of simulation)
         screen_compliance   (list of floats)     : probability of being screened (per screen) over time
         triage_compliance   (list of floats)     : probability of coming back for triage over time
         label               (str)       : the name of screening strategy
         kwargs (dict)      : passed to Intervention()

    If ``primary_screen_test`` and/or ``triage_screen_test`` is supplied as a dictionary, it must have the following parameters:

        - ``test_positivity``   : dictionary of probability of testing positive given each stage (i.e., NONE, CIN1, CIN2)

    '''

    def __init__(self, primary_screen_test, screen_start_age, screen_interval, screen_stop_age,
                 screen_start_year, screen_end_year=None, screen_compliance=None, triage_compliance=None,
                 triage_screen_test=None, screen_fu_neg_triage=None, treatment_pathway=None,
                 label=None, screen_states=None, verbose=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.label = label  # Screening label (used as a dict key)
        self.verbose = verbose
        self.p = None  # Screening parameters
        self.treatment = treatment_pathway
        self.screen_start_year = screen_start_year
        self.screen_end_year = screen_end_year
        if screen_compliance is None: # Populate default value of probability: 1
            screen_compliance = 1
        self.screen_compliance = sc.promotetolist(screen_compliance)
        if triage_compliance is None: # Populate default value of compliance: 1
            triage_compliance = 1
        self.triage_compliance = sc.promotetolist(triage_compliance)
        if screen_fu_neg_triage is None: # Populate default value of follow up after -ve triage: 1 year
            screen_fu_neg_triage = 1
        self.screen_fu_neg_triage = sc.promotetolist(screen_fu_neg_triage)
        self.screen_start_age = screen_start_age
        self.screen_interval = screen_interval
        self.screen_stop_age = screen_stop_age

        # States that will return a positive screen results
        if screen_states is None:
            screen_states = ['precin', 'cin1', 'cin2', 'cin3', 'cancerous']
        self.screen_states = screen_states

        # Parse the screening parameters, which can be provided in different formats
        self._parse_screening_pars(screen=primary_screen_test)  # Populate
        self._parse_screening_pars(screen=triage_screen_test, triage=True)  # Populate

        return


    def _parse_screening_pars(self, screen, triage=False):
        ''' Unpack screening information, which may be given as a string or dict '''

        # Option 1: screening can be chosen from a list of pre-defined screening strategies
        if isinstance(screen, str):

            choices, mapping = hppar.get_screen_choices()
            screen_pars = hppar.get_screen_pars()

            label = screen.lower()
            for txt in ['.', ' ', '&', '-', 'screen']:
                label = label.replace(txt, '')

            if label in mapping:
                label = mapping[label]
                screen_pars = screen_pars[label]
            else: # pragma: no cover
                errormsg = f'The selected screening method "{screen}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

            if self.label is None:
                self.label = label

        # Option 2: screening can be specified as a dict of pars
        elif isinstance(screen, dict):

            # Parse label
            screen_pars = screen
            label = screen_pars.pop('label', None) # Allow including the label in the parameters
            if self.label is None: # pragma: no cover
                if label is None:
                    self.label = 'custom'
                else:
                    self.label = label

        # Option 3: we are in triage and no triage is defined
        elif screen is None:
            screen_pars = None

        else: # pragma: no cover
            errormsg = f'Could not understand {type(screen)}, please specify as a string indexing a predefined vaccine or a dict.'
            raise ValueError(errormsg)

        if triage:
            if screen is None:
                self.p = sc.mergedicts(self.p, {'triage': None})
            else:
                self.p = sc.mergedicts(self.p, {'triage': sc.objdict(screen_pars)})
        else:
            # Set label and parameters
            self.p = {'primary': sc.objdict(screen_pars)}

        return


    def initialize(self, sim):
        super().initialize()

        if self.screen_end_year is None:
            self.screen_end_year = str(sim['end'])

        start_day, start_date = sim.get_t(self.screen_start_year, return_date_format='str')
        end_day, end_date = sim.get_t(self.screen_end_year, return_date_format='str')
        self.timepoints = np.arange(start_day, end_day)

        n_timepoints = len(self.timepoints)

        for compliance in ['screen_compliance', 'triage_compliance', 'screen_fu_neg_triage']:
            if len(getattr(self, compliance)) != n_timepoints:
                print(f'{n_timepoints} timepoints provided but only {len(getattr(self, compliance))} values for {compliance}. Assuming constant over time.')
                setattr(self, compliance, getattr(self, compliance)*n_timepoints)

        sim['screen_pars'][self.label] = self.p  # Store the parameters
        if self.treatment.cancer_product is not None:
            sim['treat_pars']['cancer_treatment'] = {'dur':self.treatment.cancer_product.dur}
        return


    def apply(self, sim):
        '''
        This method performs the entire screen-and-treat algorithm, using the following steps:

            1. Select people to screen and screen them using a defined primary screening algorithm
            2. Optionally triage anyone who screens positive to find those eligible for treatment
            3. Select those who will be treated, accounting for compliance
            4. Treat those who agree to treatment with defined treatment types

        Args:
            sim: hpv.Sim instance

        Returns:
            TBC
        '''

        # parameters that will be used below
        primary_screen_pars = self.p['primary']
        triage_screen_pars = self.p['triage']

        # 1. Select people to screen and screen them
        to_screen_inds = self.select_people_screen(sim)
        if len(to_screen_inds): # Screen people
            screen_pos_inds = self.screen(to_screen_inds, primary_screen_pars, sim, self.screen_states) # Determine who is eligible for triage

            # 2. Optionally triage anyone who has screened positive
            if len(screen_pos_inds):
                if triage_screen_pars is not None:
                    triage_probs = np.full(len(screen_pos_inds), self.triage_compliance[self.where_in_timepoints], dtype=hpd.default_float)
                    to_triage = hpu.binomial_arr(triage_probs)
                    triage_inds = screen_pos_inds[to_triage]  # Indices of those who get treated
                    treat_eligible_inds = self.screen(triage_inds, triage_screen_pars, sim, self.screen_states, triage=True) # Determine who is eligible for treatment

                else:
                    treat_eligible_inds = screen_pos_inds

                if self.treatment is not None:
                    self.treatment.administer(sim.people, treat_eligible_inds)

        return


    def select_people_screen(self, sim):
        """
        Return an array of indices of people to screen
        Args:
            sim: A hpv.Sim instance
        Returns: Array of person indices
        """

        screen_inds = np.array([], dtype=int)  # Initialize in case no one gets screened

        for i in find_day(self.timepoints, sim.t, interv=self, sim=sim):
            self.where_in_timepoints = i
            screen_probs = np.zeros(len(sim.people), dtype=hpd.default_float)

            # Find people eligible for screening based on age
            eligible_ages = (sim.people.age >= self.screen_start_age) &\
                            (sim.people.age <= self.screen_stop_age)

            # Assign screening probabilities
            screen_probs[eligible_ages & (sim.people.screens == 0)] = self.screen_compliance[self.where_in_timepoints] # First screen
            screen_probs[eligible_ages & (sim.t == sim.people.date_next_screen)] = self.screen_compliance[self.where_in_timepoints] # Due for next screen

            # Remove males and dead people
            screen_probs[~sim.people.alive]     *= 0.0  # Do not screen dead people
            screen_probs[ sim.people.is_male]   *= 0.0  # Do not screen men
            screen_probs[~sim.people.is_active] *= 0.0  # Corner case, avoid screening anyone not yet sexually active

            # Calculate who actually gets screened
            screen_inds = hpu.true(hpu.binomial_arr(screen_probs))

            # Set screening states and dates
            sim.people.intv_flows['screens'] += len(screen_inds)
            # sim.people.intv_flows['screened'] += len(hpu.true(sim.people[screen_inds].screens == 0))
            sim.people.screened[screen_inds] = True
            sim.people.screens[screen_inds] += 1
            sim.people.date_screened[screen_inds] = sim.t
            sim.people.date_next_screen[screen_inds] = sim.t + self.screen_interval/sim['dt']

        return screen_inds


    def screen(self, screen_inds, pars, sim, states, triage=False):
        ''' Screen or triage people '''
        screen_pos = []
        for state in states:
            screen_probs = np.zeros(len(screen_inds))
            if pars['by_genotype']:
                for g in range(sim['n_genotypes']):
                    tp_inds = hpu.true(sim.people[state][g, screen_inds])
                    screen_probs[tp_inds] = pars['test_positivity'][state][sim['genotype_map'][g]]
                    screen_pos_inds = hpu.true(hpu.binomial_arr(screen_probs))
                    screen_pos += list(screen_pos_inds)
                screen_pos = list(set(screen_pos)) # If anyone has screened positive for >1 genotype, only include them once

            else:
                tp_inds = hpu.true(sim.people[state][:, screen_inds].any(axis=0))
                screen_probs[tp_inds] = pars['test_positivity'][state]
                screen_pos_inds = hpu.true(hpu.binomial_arr(screen_probs))
                screen_pos += list(screen_pos_inds)

        screen_pos = np.array(screen_pos)
        if len(screen_pos)>0:
            screen_pos = screen_inds[screen_pos]

        if triage:
            screen_neg = np.setdiff1d(screen_inds, screen_pos)
            sim.people.date_next_screen[screen_neg] = sim.t + self.screen_fu_neg_triage[self.where_in_timepoints] / sim['dt'] # primary +ve/ triage -ve follow up sooner
        return screen_pos


    def shrink(self, in_place=True):
        ''' Shrink vaccination intervention '''
        obj = super().shrink(in_place=in_place)
        return obj


#%% Treatment
__all__ += ['StandardTreatmentPathway', 'RadiationTherapy', 'PrecancerTreatment', 'ExcisionTreatment', 'AblativeTreatment']

class Product():
    """
    Generic product implementation
    """

    # Could potentially track other product related things like costs there too?!

    def administer(self, people, inds):
        """
        Change something about the People based on them recieving this product
        """
        raise NotImplementedError


class PrecancerTreatment(Product):
    def __init__(self):
        self.efficacy=dict(
            precin=0,
            cin1=0.936,
            cin2=0.936,
            cin3=0.936,
        )
        self.treat_states = ['precin', 'cin1', 'cin2', 'cin3']

    def administer(self, people, inds):
        # Loop over treatment states to determine those who (a) are successfully treated and (b) clear infection

        # nb. this will record treated=True reflecting delivery of the treatment, even if the treatment fails?
        people.treated[inds] = True
        people.date_treated[inds] = people.t

        successfully_treated = []
        for state in self.treat_states:
            people_in_state = people[state].any(axis=0)
            treat_state_inds = inds[people_in_state[inds]]

            # Determine whether treatment is successful
            eff_probs = np.full(len(treat_state_inds), self.efficacy[state],
                                dtype=hpd.default_float)  # Assign probabilities of treatment success
            to_eff_treat = hpu.binomial_arr(eff_probs)  # Determine who will have effective treatment
            eff_treat_inds = treat_state_inds[to_eff_treat]
            successfully_treated += list(eff_treat_inds)
            people[state][:, eff_treat_inds] = False  # People who get treated have their CINs removed
            people[f'date_{state}'][:, eff_treat_inds] = np.nan

            # Clear infection for women who clear
            for g in range(people.pars['n_genotypes']):
                people['infectious'][g, eff_treat_inds] = False  # People whose HPV clears
                people.dur_disease[g, eff_treat_inds] = (people.t - people.date_infectious[g, eff_treat_inds]) * people.pars['dt']
                hpi.update_peak_immunity(people, eff_treat_inds, imm_pars=people.pars, imm_source=g)

        people.treated[inds] = True
        people.date_treated[inds] = people.t

        return successfully_treated


class ExcisionTreatment(PrecancerTreatment):
    def __init__(self):
        super().__init__()
        self.efficacy = dict(
            precin=0,
            cin1=0.936,
            cin2=0.936,
            cin3=0.936,
        )


class AblativeTreatment(PrecancerTreatment):
    def __init__(self):
        super().__init__()
        self.efficacy = dict(
            precin=0,
            cin1=0.81,
            cin2=0.81,
            cin3=0.81,
        )


# class TherapeuticVaccine(Product):
#     def __init__(self, timepoints=None, doses=None, interval=None, efficacy=None):
#         self.timepoints = timepoints or '2030'
#         self.doses = doses or 2
#         self.interval = interval or 0.5 # Interval between doses in years
#         self.treat_states = ['precin', 'cin1', 'cin2', 'cin3']
#         self.efficacy = efficacy or dict( # default efficacy decreases as dysplasia increases
#             precin=dict(
#                 hpv16=[0.1, 0.9],
#                 hpv18=[0.1, 0.9],
#                 hpv31=[0.01, 0.1],
#                 hpv33=[0.01, 0.1],
#                 hpv35=[0.01, 0.1],
#                 hpv45=[0.01, 0.1],
#                 hpv51=[0.01, 0.1],
#                 hpv52=[0.01, 0.1],
#                 hpv56=[0.01, 0.1],
#                 hpv58=[0.01, 0.1],
#                 hpv6=[0.01, 0.1],
#                 hpv11=[0.01, 0.1],
#             ),
#             cin1=dict(
#                 hpv16=[0.1, 0.7],
#                 hpv18=[0.1, 0.7],
#                 hpv31=[0.01, 0.1],
#                 hpv33=[0.01, 0.1],
#                 hpv35=[0.01, 0.1],
#                 hpv45=[0.01, 0.1],
#                 hpv51=[0.01, 0.1],
#                 hpv52=[0.01, 0.1],
#                 hpv56=[0.01, 0.1],
#                 hpv58=[0.01, 0.1],
#                 hpv6=[0.01, 0.1],
#                 hpv11=[0.01, 0.1],
#             ),
#             cin2=dict(
#                 hpv16=[0.1, 0.5],
#                 hpv18=[0.1, 0.5],
#                 hpv31=[0.01, 0.1],
#                 hpv33=[0.01, 0.1],
#                 hpv35=[0.01, 0.1],
#                 hpv45=[0.01, 0.1],
#                 hpv51=[0.01, 0.1],
#                 hpv52=[0.01, 0.1],
#                 hpv56=[0.01, 0.1],
#                 hpv58=[0.01, 0.1],
#                 hpv6=[0.01, 0.1],
#                 hpv11=[0.01, 0.1],
#             ),
#             cin3=dict(
#                 hpv16=[0.1, 0.4],
#                 hpv18=[0.1, 0.4],
#                 hpv31=[0.01, 0.1],
#                 hpv33=[0.01, 0.1],
#                 hpv35=[0.01, 0.1],
#                 hpv45=[0.01, 0.1],
#                 hpv51=[0.01, 0.1],
#                 hpv52=[0.01, 0.1],
#                 hpv56=[0.01, 0.1],
#                 hpv58=[0.01, 0.1],
#                 hpv6=[0.01, 0.1],
#                 hpv11=[0.01, 0.1],
#             ),
#         )
#
#     def initialize(self, sim):
#         self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str')  # Ensure timepoints and dates are in the right format
#
#         self.second_dose_timepoints = [None] * sim.npts  # People who get second dose (if relevant)
#
#
#     def administer(self, people, inds):
#
#         #Extract parameters that will be used below
#
#         ng = people.pars['n_genotypes']
#         genotype_map = people.pars['genotype_map']
#
#         # Find those who are getting first dose
#         people_not_vaccinated = hpu.false(people.tx_vaccinated)
#         first_dose_inds = inds[people_not_vaccinated[inds]]
#         people.tx_vaccinated[first_dose_inds] = True
#
#         # Schedule next dose
#         second_dose_timepoints = people.t + int(self.interval / people.pars['dt'])
#         if second_dose_timepoints < people.npts:
#             self.second_dose_timepoints[second_dose_timepoints] = first_dose_inds
#
#         people.txvx_doses[inds] += 1
#
#         # Find those who are getting second dose today
#         second_dose_inds = np.setdiff1d(inds, first_dose_inds)
#
#         # Deliver vaccine and update prognoses
#         for inds_to_treat, dose in zip([first_dose_inds, second_dose_inds], [0,1]):
#             for state in self.treat_states:
#                 for g in range(ng):
#                     people_in_state = hpu.true(people[g,state])
#                     treat_state_inds = inds_to_treat[people_in_state[inds_to_treat]]
#
#                     # Determine whether treatment is successful
#                     eff_probs = np.full(len(treat_state_inds), self.efficacy[state][genotype_map[g]][dose],
#                                         dtype=hpd.default_float)  # Assign probabilities of treatment success
#                     to_eff_treat = hpu.binomial_arr(eff_probs)  # Determine who will have effective treatment
#                     eff_treat_inds = treat_state_inds[to_eff_treat]
#                     people[state][g, eff_treat_inds] = False  # People who are successfully treated
#                     people[f'date_{state}'][g, eff_treat_inds] = np.nan
#                     hpi.update_peak_immunity(people, eff_treat_inds, imm_pars=people.pars, imm_source=g) # Get natural immune memory


class RadiationTherapy(Product):
    # Cancer treatment product
    def __init__(self, dur=None):
        self.dur = dur or dict(dist='normal', par1=18.0, par2=2.) # whatever the default duration should be

    def administer(self, people, inds):
        new_dur_cancer = hpu.sample(**self.dur, size=len(inds))
        people.date_dead_cancer[inds] += np.ceil(new_dur_cancer / people.pars['dt'])
        people.treated[inds] = True
        people.date_treated[inds] = people.t
        return inds


class StandardTreatmentPathway(Product):
    # A standard treatment pathway - kind of a meta-product that represents the normal algorithm and dispatches treatments to specific products

    def __init__(self, ablation_compliance, excision_compliance, cancer_compliance, cancer_product=None, ablation_product=None, excision_product=None):
        self.ablation_compliance = ablation_compliance # probability of coming back for ablation
        self.excision_compliance = excision_compliance # probability of coming back for excision
        self.cancer_compliance = cancer_compliance # probability of coming back for cancer treatment
        self.cancer_product = cancer_product or RadiationTherapy()
        self.ablation_product = ablation_product or AblativeTreatment()
        self.excision_product = excision_product or ExcisionTreatment()
        self.test_positivity = {'precin': 0.98, 'cin1': 0.97, 'cin2': 0.89,'cin3': 0.79, 'cancerous': 0.4} # eligibility for ablation vs excision
        self.treat_states = ['precin', 'cin1', 'cin2', 'cin3']

    def administer(self, people, inds):
        ''' Determine what kind of treatment they should receive and administer it '''

        # TREAT CANCER
        cancerous_inds = hpu.true(people.cancerous.any(axis=0)) # Find indices of people with cancer
        diagnosed_inds = np.intersect1d(inds, cancerous_inds) # Indices of those who will be diagnosed with cancer
        if len(diagnosed_inds)>0:
            # Treat cancers
            ca_treat_probs = np.full(len(diagnosed_inds), self.cancer_compliance, dtype=hpd.default_float)
            to_treat_ca = hpu.binomial_arr(ca_treat_probs)  # Determine who actually gets treated, after accounting for compliance
            ca_treat_inds = diagnosed_inds[to_treat_ca]  # Indices of those who get treated
            ca_LTFU_inds = diagnosed_inds[~to_treat_ca] # Indices of those lost to follow up
            people.date_next_screen[ca_LTFU_inds] = np.nan # Remove any future screening
            self.cancer_product.administer(people, ca_treat_inds)
        else:
            ca_treat_inds = np.array([], dtype=hpd.default_int)

        # Everyone remaining is eligible for precancer treatment
        preca_treat_eligible_inds = np.setdiff1d(inds, ca_treat_inds) # Indices of those eligible for precancer treatment

        # Determine who is eligible for ablative vs excisional treatment
        if len(preca_treat_eligible_inds)>0:
            ablation_eligible_inds = []
            for state in self.treat_states:
                ablate_probs = np.zeros(len(preca_treat_eligible_inds))
                ablate_inds = hpu.true(people[state][:, preca_treat_eligible_inds].any(axis=0))
                ablate_probs[ablate_inds] = self.test_positivity[state]
                ablate_inds = hpu.true(hpu.binomial_arr(ablate_probs))
                ablation_eligible_inds += list(ablate_inds)

            ablation_eligible_inds = np.array(ablation_eligible_inds)
            if len(ablation_eligible_inds)>0:
                ablation_eligible_inds = preca_treat_eligible_inds[ablation_eligible_inds]
            excision_eligible_inds = np.setdiff1d(preca_treat_eligible_inds, ablation_eligible_inds)

            # Apply LTFU for both
            if len(ablation_eligible_inds)>0:
                ablate_treat_probs = np.full(len(ablation_eligible_inds), self.ablation_compliance, dtype=hpd.default_float)
                to_ablate = hpu.binomial_arr(ablate_treat_probs)
                ablation_inds = ablation_eligible_inds[to_ablate]  # Indices of those who get treated
                ablate_LTFU_inds = ablation_eligible_inds[~to_ablate]
                self.ablation_product.administer(people, ablation_inds)
                people.date_next_screen[ablate_LTFU_inds] = np.nan

            if len(excision_eligible_inds)>0:
                excision_treat_probs = np.full(len(excision_eligible_inds), self.excision_compliance, dtype=hpd.default_float)
                to_excise = hpu.binomial_arr(excision_treat_probs)
                excision_inds = excision_eligible_inds[to_excise]  # Indices of those who get treated
                excision_LTFU_inds = excision_eligible_inds[~to_excise]
                self.excision_product.administer(people, excision_inds)
                people.date_next_screen[excision_LTFU_inds] = np.nan

        return


__all__ += ['TherapeuticVaccination', 'routine_therapeutic']


class TherapeuticVaccination(Intervention, Product):
    '''
        Base class to apply a therapeutic vaccine to a subset of the population. Can be implemented as
        a campaign-style or routine administration within S&T.

        This class implements the mechanism of delivering a therapeutic vaccine.

        '''

    def __init__(self, timepoints, prob=None, LTFU=None,  doses=None, interval=None, efficacy=None, subtarget=None,
                 proph=False, vaccine='bivalent_1dose', **kwargs):
        super().__init__(**kwargs)  # Initialize the Intervention object
        self.subtarget = subtarget
        if prob is None: # Populate default value of probability: 1 if no subtargeting, 0 if subtargeting
            prob = 1.0 if subtarget is None else 0.0
        self.prob      = prob
        self.LTFU = LTFU
        self.timepoints = timepoints
        self.doses = doses or 2
        self.interval = interval or 0.5  # Interval between doses in years
        self.prophylactic = proph # whether to deliver a single-dose prophylactic vaccine at first dose
        self.vaccine = vaccine # which vaccine to deliver
        self.treat_states = ['precin', 'latent', 'cin1', 'cin2', 'cin3']
        self.efficacy = efficacy or dict(  # default efficacy decreases as dysplasia increases
            precin=dict(
                hpv16=[0.1, 0.9],
                hpv18=[0.1, 0.9],
                hpv31=[0.01, 0.1],
                hpv33=[0.01, 0.1],
                hpv35=[0.01, 0.1],
                hpv45=[0.01, 0.1],
                hpv51=[0.01, 0.1],
                hpv52=[0.01, 0.1],
                hpv56=[0.01, 0.1],
                hpv58=[0.01, 0.1],
                hpv6=[0.01, 0.1],
                hpv11=[0.01, 0.1],
            ),
            latent=dict(
                hpv16=[0.1, 0.9],
                hpv18=[0.1, 0.9],
                hpv31=[0.01, 0.1],
                hpv33=[0.01, 0.1],
                hpv35=[0.01, 0.1],
                hpv45=[0.01, 0.1],
                hpv51=[0.01, 0.1],
                hpv52=[0.01, 0.1],
                hpv56=[0.01, 0.1],
                hpv58=[0.01, 0.1],
                hpv6=[0.01, 0.1],
                hpv11=[0.01, 0.1],
            ),
            cin1=dict(
                hpv16=[0.1, 0.7],
                hpv18=[0.1, 0.7],
                hpv31=[0.01, 0.1],
                hpv33=[0.01, 0.1],
                hpv35=[0.01, 0.1],
                hpv45=[0.01, 0.1],
                hpv51=[0.01, 0.1],
                hpv52=[0.01, 0.1],
                hpv56=[0.01, 0.1],
                hpv58=[0.01, 0.1],
                hpv6=[0.01, 0.1],
                hpv11=[0.01, 0.1],
            ),
            cin2=dict(
                hpv16=[0.1, 0.6],
                hpv18=[0.1, 0.6],
                hpv31=[0.01, 0.1],
                hpv33=[0.01, 0.1],
                hpv35=[0.01, 0.1],
                hpv45=[0.01, 0.1],
                hpv51=[0.01, 0.1],
                hpv52=[0.01, 0.1],
                hpv56=[0.01, 0.1],
                hpv58=[0.01, 0.1],
                hpv6=[0.01, 0.1],
                hpv11=[0.01, 0.1],
            ),
            cin3=dict(
                hpv16=[0.1, 0.5],
                hpv18=[0.1, 0.5],
                hpv31=[0.01, 0.1],
                hpv33=[0.01, 0.1],
                hpv35=[0.01, 0.1],
                hpv45=[0.01, 0.1],
                hpv51=[0.01, 0.1],
                hpv52=[0.01, 0.1],
                hpv56=[0.01, 0.1],
                hpv58=[0.01, 0.1],
                hpv6=[0.01, 0.1],
                hpv11=[0.01, 0.1],
            ),
        )
        return

    def initialize(self, sim):
        super().initialize()
        self.timepoints, self.dates = sim.get_t(self.timepoints,
                                                return_date_format='str')  # Ensure timepoints and dates are in the right format
        self.second_dose_timepoints = [None] * sim.npts  # People who get second dose (if relevant)
        if self.prophylactic:
            # Initialize a prophylactic vaccine intervention to reference later
            vx = vaccinate_prob(vaccine=self.vaccine, prob=0, timepoints=self.dates[0])
            vx.initialize(sim)
            self.prophylactic_vaccine = vx
        return

    def administer(self, people, inds):

        #Extract parameters that will be used below
        ng = people.pars['n_genotypes']
        genotype_map = people.pars['genotype_map']

        # Find those who are getting first dose
        people_not_vaccinated = hpu.false(people.tx_vaccinated)
        first_dose_inds = np.intersect1d(people_not_vaccinated, inds)
        people.tx_vaccinated[first_dose_inds] = True

        people.txvx_doses[inds] += 1

        # Find those who are getting second dose today
        second_dose_inds = np.setdiff1d(inds, first_dose_inds)

        # Deliver vaccine and update prognoses TODO: immune response in those without infection/lesion
        for inds_to_treat, dose in zip([first_dose_inds, second_dose_inds], [0,1]):
            for state in self.treat_states:
                for g in range(ng):
                    people_in_state = hpu.true(people[state][g,inds_to_treat])
                    treat_state_inds = inds_to_treat[people_in_state]

                    # Determine whether treatment is successful
                    eff_probs = np.full(len(treat_state_inds), self.efficacy[state][genotype_map[g]][dose],
                                        dtype=hpd.default_float)  # Assign probabilities of treatment success
                    to_eff_treat = hpu.binomial_arr(eff_probs)  # Determine who will have effective treatment
                    eff_treat_inds = treat_state_inds[to_eff_treat]
                    people[state][g, eff_treat_inds] = False  # People who are successfully treated
                    people[f'date_{state}'][g, eff_treat_inds] = np.nan
                    hpi.update_peak_immunity(people, eff_treat_inds, imm_pars=people.pars, imm_source=g) # Get natural immune memory

        return

    def select_people(self, sim):

        vacc_inds = np.array([], dtype=int)  # Initialize in case no one gets their first dose

        if sim.t >= np.min(self.timepoints):

            # Vaccinate people with their first dose
            for _ in find_day(self.timepoints, sim.t, interv=self, sim=sim):

                vacc_probs = np.zeros(len(sim.people))

                # Find eligible people
                vacc_probs[hpu.true(~sim.people.alive)] *= 0.0  # Do not vaccinate dead people
                eligible_inds = sc.findinds(~sim.people.tx_vaccinated)
                vacc_probs[eligible_inds] = self.prob  # Assign equal vaccination probability to everyone

                # Apply any subtargeting
                if self.subtarget is not None:
                    subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
                    vacc_probs[subtarget_inds] = subtarget_vals  # People being explicitly subtargeted

                vacc_inds = hpu.true(hpu.binomial_arr(vacc_probs))  # Calculate who actually gets vaccinated

                if len(vacc_inds):
                    if self.interval is not None:
                        # Schedule the doses
                        second_dose_timepoints = sim.t + int(self.interval/sim['dt'])
                        if second_dose_timepoints < sim.npts:
                            self.second_dose_timepoints[second_dose_timepoints] = vacc_inds

            idx = int(sim.t / sim.resfreq)
            sim.results['new_txvx_vaccinated'][idx] += len(vacc_inds)

            # Also, if appropriate, vaccinate people with their second doses
            vacc_inds_dose2 = self.second_dose_timepoints[sim.t]
            if vacc_inds_dose2 is not None:
                if self.LTFU is not None:
                    vacc_probs = np.full(len(vacc_inds_dose2), (1-self.LTFU))
                    vacc_inds_dose2 = vacc_inds_dose2[hpu.true(hpu.binomial_arr(vacc_probs))]
                vacc_inds = np.concatenate((vacc_inds, vacc_inds_dose2), axis=None)

            sim.results['new_txvx_doses'][idx] += len(vacc_inds)


        return vacc_inds

    def apply(self, sim):
        ''' Perform vaccination each timestep '''
        inds = self.select_people(sim)
        if len(inds):
            self.administer(sim.people, inds)
            if self.prophylactic:
                inds_to_vax = inds[hpu.false(sim.people.vaccinated[inds])]
                self.prophylactic_vaccine.vaccinate(sim, inds_to_vax)
        return inds


class routine_therapeutic(TherapeuticVaccination):

    def __init__(self, *args, age_range, coverage, **kwargs):
        super().__init__(*args, **kwargs, subtarget=self.subtarget_function)
        self.age_range = age_range
        self.coverage = sc.promotetoarray(coverage)
        if len(self.coverage) == 1:
            self.coverage = self.coverage * np.ones_like(self.timepoints)

    def subtarget_function(self, sim):
        inds = sc.findinds((sim.people.age >= self.age_range[0]) & (sim.people.age < self.age_range[1]) & (sim.people.is_female))
        coverage = self.coverage[self.timepoints == sim.t][0]
        return {'vals': coverage * np.ones_like(inds), 'inds': inds}
