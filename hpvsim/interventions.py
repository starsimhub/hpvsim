'''
Specify the core interventions. Other interventions can be
defined by the user by inheriting from these classes.
'''

import numpy as np
import sciris as sc
import pylab as pl
import pandas as pd
import inspect
from . import defaults as hpd
from . import parameters as hppar
from . import utils as hpu
from . import immunity as hpi
from . import base as hpb
from collections import defaultdict


#%% Helper functions

def find_timepoint(arr, t=None, interv=None, sim=None, which='first'):
    '''
    Helper function to find if the current simulation time matches any timepoint in the
    intervention. Although usually never more than one index is returned, it is
    returned as a list for the sake of easy iteration.

    Args:
        arr (list/function): list of timepoints in the intervention, or a boolean array; or a function that returns these
        t (int): current simulation time (can be None if a boolean array is used)
        interv (intervention): the intervention object (usually self); only used if arr is callable
        sim (sim): the simulation object; only used if arr is callable
        which (str): what to return: 'first', 'last', or 'all' indices

    Returns:
        inds (list): list of matching timepoints; length zero or one unless which is 'all'
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
__all__ += ['DynamicPars', 'EventSchedule', 'set_intervention_attributes']

class DynamicPars(Intervention):
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

        interv = hp.DynamicPars(condoms=dict(timepoints=10, vals={'c':0.9})) # Increase condom use amount casual partners to 90%
        interv = hp.DynamicPars({'beta':{'timepoints':[10, 15], 'vals':[0.005, 0.015]}, # At timepoint 10, reduce beta, then increase it again
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
            if t in parval['processed_timepoints']:
                self.timepoints.append(t)
                ind = sc.findinds(parval['processed_timepoints'], t)[0]
                val = parval['vals'][ind]
                if isinstance(val, dict):
                    sim[parkey].update(val) # Set the parameter if a nested dict
                else:
                    sim[parkey] = val # Set the parameter if not a dict
        return


class EventSchedule(Intervention):
    """
    Run functions on different days

    This intervention is a a kind of generalization of `dynamic_pars` to allow more
    flexibility in triggering multiple, arbitrary operations and to more easily assemble
    multiple changes at different times. This intervention can be used to implement scale-up
    or other changes to interventions without needing to implement time-dependency in the
    intervention itself.

    To use the intervention, simply index the intervention by `t` or by date, and then
    Example:

    >>> iv = EventSchedule()
    >>> iv[1] = lambda sim: print(sim.t)
    >>> iv['2020-04-02'] = lambda sim: print('foo')

    """

    def __init__(self):
        super().__init__()
        self.schedule = defaultdict(list)

    def __getitem__(self, day):
        return self.schedule[day]

    def __setitem__(self, day, fcn):
        if day in self.schedule:
            raise Exception("Use a list instead to assign multiple functions - or to really overwrite, delete the function for this day first i.e. `del schedule[day]` before performing `schedule[day]=...`")
        self.schedule[day] = fcn

    def __delitem__(self, key):
        del self.schedule[key]

    def initialize(self, sim):
        super().initialize(sim)

        # First convert all values into lists (i.e., wrap any standalone functions into lists)
        for k, v in list(self.schedule.items()):
            self.schedule[k] = [v] if not isinstance(self.schedule[k], list) else v

        # Then convert any dates into time indices
        for k, v in list(self.schedule.items()):
            t = sim.get_t(k)[0]
            if t != k:
                self.schedule[t] += v
                del self.schedule[k]

    def apply(self, sim):
        if sim.t in self.schedule:
            for fcn in self.schedule[sim.t]:
                fcn(sim)


def set_intervention_attributes(sim, intervention_name, **kwargs):
    # This is a helper method that can be used to set arbitrary intervention attributes
    # It's a separately defined function so that it can be pickled properly
    iv = sim.get_intervention(intervention_name)
    for attr, value in kwargs.items():
        assert hasattr(iv, attr), "set_intervention_attributes() should only be used to change existing attributes"  # avoid silent errors if the attr is misspelled
        setattr(iv, attr, value)


#%% Vaccination
__all__ += ['BaseVaccination', 'Vaccination', 'RoutineVaccination', 'NumberVaccination']

class BaseVaccination(Intervention):
    '''
    Apply a vaccine to a subset of the population.

    This base class implements the mechanism of vaccinating people to modify their immunity.
    It does not implement allocation of the vaccines, which is implemented by derived classes
    such as `hpv.VaccinateProb`.

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

        # Store the number of doses administered by this intervention
        self.doses = hpb.Result(name=f'Doses {self.label}', npts=sim.res_npts, scale=True)

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
            self.doses[idx] += len(vacc_inds)

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


class Vaccination(BaseVaccination):
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

        bivalent = hpv.Vaccination(vaccine='bivalent', timepoints='2020', prob=0.7)
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
            for _ in find_timepoint(self.timepoints, sim.t, interv=self, sim=sim):

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


class RoutineVaccination(Vaccination):

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


class NumberVaccination(BaseVaccination):
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

        bivalent = hpv.NumberVaccination(vaccine='bivalent', num_doses=1e6, timepoints=2020)
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


#%% Screening and triage
__all__ += ['BaseScreening', 'RoutineScreening', 'CampaignScreening', 'Triage']

# Helper function
def select_people(inds, prob=None):
    '''
    Return an array of indices of people to who accept a service being offered
    Args:
        inds: array of indices of people offered a service (e.g. screening, triage, treatment)
        prob: acceptance probability
    Returns: Array of indices of people who accept triage
    '''
    accept_probs    = np.full_like(inds, fill_value=prob, dtype=hpd.default_float)
    accept_inds     = hpu.true(hpu.binomial_arr(accept_probs))
    return accept_inds


class BaseScreening(Intervention):
    '''
    Base screening class. Different logic applies depending on whether it's a routine
    or campaign screening intervention, so these are separated into different Interventions.
    This base class contains the common functionality for both.
    Args:
         product            (str/Product)   : the screening test to use
         screen_prob        (float/arr) : annual probability of eligible women getting screened
         eligibility        (inds/fn/state): indices OR string corresponding to valid state or people OR callable
         age_range          (list)      : age range for screening
         results_to_store   (list)      : which results to store in the intervention (to avoid storing all results)
         store_by_time      (bool)      : whether to store the results keyed by time - default false, so the indices of people are stored in a single pool
         label              (str)       : the name of screening strategy
         kwargs             (dict)      : passed to Intervention()
≈    '''

    def __init__(self, product, screen_prob, eligibility=None,
                 age_range=None, store_states=None,
                 label=None, verbose=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.product        = product # The test product being used to screen people
        self.screen_prob    = sc.promotetoarray(screen_prob) # Annual probability of being screened
        self.eligibility    = eligibility
        self.age_range      = age_range or [30,50] # This is later filtered to exclude people not yet sexually active
        self.label          = label  # Screening label (used as a dict key)
        self.verbose        = verbose
        self.store_states   = store_states

        # # Parse the screening product(s), which can be provided in different formats
        # self._parse_products()

        return


    def _parse_products(self):
        ''' Unpack screening information, which may be given as a string or Product'''
        return


    def initialize(self, sim):
        super().initialize()
        self.npts = sim.res_npts
        self.n_products_used = hpb.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)
        self.outcomes = {k:[] for k in self.product.hierarchy}
        return


    def apply(self, sim):
        ''' TBC '''
        if sim.t in self.timepoints:

            # Select people for screening and then record the number of screens
            ti = sc.findinds(self.timepoints, sim.t)[0]
            screen_prob     = self.screen_prob[ti] # Get the proportion of people who screen on this timestep
            eligible_inds   = self.check_eligibility(sim) # Check eligibility
            screen_inds     = select_people(eligible_inds, prob=screen_prob)

            if len(screen_inds):
                idx = int(sim.t / sim.resfreq)
                sim.people.screened[screen_inds] = True
                sim.people.screens[screen_inds] += 1
                sim.people.date_screened[screen_inds] = sim.t
                self.n_products_used[idx] += len(screen_inds)

                # Step 2: screen people
                self.outcomes = self.product.administer(sim, screen_inds)

        return


    def check_eligibility(self, sim):
        ''' Return boolean array specifying who's eligible for screening at time t '''
        active_females  = sim.people.is_female & sim.people.is_active
        in_age_range    = (sim.people.age >= self.age_range[0]) & (sim.people.age <= self.age_range[1])
        screen_due      = self.eligibility(sim)
        return active_females & in_age_range & screen_due


class RoutineScreening(BaseScreening):
    '''
    Routine screening.
    Example:
        screen1 = hpv.RoutineScreening('hpv', 0.02) # Screen 2% of the eligible population every year
        screen2 = hpv.RoutineScreening('hpv', 0.02, start_year=2020) # Screen 2% every year starting in 2020
        screen3 = hpv.RoutineScreening('hpv', np.linspace(0.005,0.025,5), years=np.arange(2020,2025)) # Scale up screening over 5 years starting in 2020
    '''
    def __init__(self, years=None, start_year=None, end_year=None, interval=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.years      = years
        self.start_year = start_year
        self.end_year   = end_year
        self.interval   = interval or 100 # by default, people only have one lifetime screen
        return

    def initialize(self, sim):
        super().initialize(sim)

        # Handle time/date inputs
        if (self.years is not None) and (self.start_year is not None or self.end_year is not None):
            errormsg = 'Provide either a list of years or a start year, not both.'
            raise ValueError(errormsg)

        if self.years is None:
            if self.start_year is None: self.start_year = sim.res_yearvec[0]
            if self.end_year is None:   self.end_year   = sim.res_yearvec[-1]
        else:
            self.start_year = self.years[0]
            self.end_year   = self.years[-1]

        if (self.start_year not in sim.yearvec) or (self.end_year not in sim.yearvec):
            errormsg = 'Years for screening must be within simulation start and end dates.'
            raise ValueError(errormsg)

        self.start_point    = sc.findinds(sim.yearvec, self.start_year)[0]
        self.end_point      = sc.findinds(sim.yearvec, self.end_year)[0]
        self.years          = np.arange(self.start_year, self.end_year)
        self.timepoints     = np.arange(self.start_point, self.end_point)

        if len(self.years) != len(self.screen_prob):
            if len(self.screen_prob)==1:
                self.screen_prob = np.array([self.screen_prob[0]]*len(self.timepoints))
            else:
                errormsg = f'Length of screening years incompatible with length of screening probabilities: {len(self.years)} vs {len(self.screen_prob)}'
                raise ValueError(errormsg)
        else:
            self.screen_prob = sc.smoothinterp(self.timepoints, self.years, self.screen_prob, smoothness=0)

        self.screen_prob *= sim['dt']

        return

class CampaignScreening(BaseScreening):
    '''
    Campaign screening.
    Example:
        campaign_screen = hpv.CampaignScreening('hpv', 0.2, years=2020) # Screen 20% of the eligible population in 2020 and 2025
        sim = hpv.Sim(interventions=campaign_screen)
    '''
    def __init__(self, years, interpolate=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.years = sc.promotetoarray(years)
        self.interpolate = interpolate # Whether to space the intervention over the year (if true) or do them all at once (if false)
        return

    def initialize(self, sim):
        super().initialize(sim)

        if self.interpolate:
            yearpoints = []
            for yi, year in enumerate(self.years):
                yearpoints += [year+(i*sim['dt']) for i in range(int(1 / sim['dt']))]
            self.timepoints = np.array([sc.findinds(sim.yearvec,yp)[0] for yp in yearpoints])
        else:
            self.timepoints = np.array([sc.findinds(sim.yearvec,year)[0] for year in self.years])

        if len(self.years) != len(self.screen_prob):
            if len(self.screen_prob) == 1:
                self.screen_prob = np.array([self.screen_prob[0]] * len(self.timepoints))
            else:
                errormsg = f'Length of screening years incompatible with length of screening probabilities: {len(self.years)} vs {len(self.screen_prob)}'
                raise ValueError(errormsg)

        if self.interpolate: self.screen_prob *= sim['dt']

        return


class Triage(Intervention):
    '''
    Triage
    Args:
         product            (str/Product)   : the screening test to use
         triage_prob        (float/arr) : annual probability of eligible women getting screened
         eligibility        (callable/arr): array of indices of people who are eligible for triage OR callable that returns such indices
≈    '''

    def __init__(self, product, triage_prob, eligibility=None,
                 label=None, verbose=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.product        = product # The test product being used to screen people
        self.triage_prob    = sc.promotetoarray(triage_prob) # Proportion of those eligible for triage who accept. Applied each timestep to everyone who becomes eligible on that timestep (as determined by eligibility)
        self.eligibility    = eligibility # Function or indices that determine who is eligible for the intervention
        self.label          = label  # label (used as a dict key)
        self.verbose        = verbose
        return


    def initialize(self, sim):
        super().initialize()
        self.npts = sim.res_npts
        self.n_products_used = hpb.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)
        self.outcomes = {k:[] for k in self.product.hierarchy}
        return


    def apply(self, sim):
        eligible_inds = self.eligibility(sim) # Determine who's eligible
        idx = int(sim.t / sim.resfreq) # Get the result time index
        accept_inds = select_people(eligible_inds, prob=self.triage_prob[0]) # Select people who accept
        self.n_products_used[idx] += len(accept_inds) # Count the number of products used
        self.outcomes = self.product.administer(sim, accept_inds) # Administer the product
        return



#%% Treatment interventions
__all__ += ['BaseTreatment', 'NumTreat', 'DelayTreat']

class BaseTreatment(Intervention):
    def __init__(self, product, eligibility, treat_prob, **kwargs):
        super().__init__(**kwargs)
        self.eligibility = eligibility
        self.product = product
        self.treat_prob = sc.promotetoarray(treat_prob)

    def initialize(self, sim):
        super().initialize()
        self.n_products_used = hpb.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)

    def recheck_eligibility(self, inds, sim):
        ''' Recheck people's eligibility for treatment - it may have expired if they've been waiting awhile '''
        conditions = sim.people.alive[inds] # TODO: what else should go here? can't use sim.people.cin because precins may also get treatment
        return conditions

    def get_accept_inds(self, sim):
        ''' Add new indices to the queue of people awaiting treatment '''
        accept_inds     = np.array([], dtype=hpd.default_int)
        eligible_inds   = self.eligibility(sim) # Apply eligiblity
        if len(eligible_inds):
            accept_inds     = select_people(eligible_inds, prob=self.treat_prob[0])  # Select people who accept
        return accept_inds

    def apply(self, sim):
        treat_candidates = self.get_candidates(sim)
        still_eligible = self.recheck_eligibility(treat_candidates, sim)
        treat_inds = treat_candidates[still_eligible]
        self.product.administer(sim.people, treat_inds)
        idx = int(sim.t / sim.resfreq)
        self.n_products_used[idx] += len(treat_inds)


class NumTreat(BaseTreatment):
    def __init__(self, max_capacity=None, **kwargs):
        super().__init__(**kwargs)
        self.queue = []
        self.max_capacity = max_capacity
        return

    def add_to_queue(self, sim):
        accept_inds = self.get_accept_inds(sim)
        if len(accept_inds): self.queue += accept_inds.tolist()  # Add people who are willing to accept treatment to the queue

    def get_candidates(self, sim):
        treat_candidates = np.array([], dtype=hpd.default_int)
        if len(self.queue):
            if self.max_capacity is None or (self.max_capacity>len(self.queue)):
                treat_candidates = self.queue[:]
            else:
                treat_candidates = self.queue[:self.max_capacity]
        return treat_candidates

    def apply(self, sim):
        self.add_to_queue(sim)
        super().apply(sim)
        self.queue = [e for e in self.queue if e not in treat_inds] # Recreate the queue, removing people who were treated
        return


class DelayTreat(BaseTreatment):
    ''' Treat people after a fixed delay '''
    def __init__(self, delay=None, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay or 0
        self.scheduler = defaultdict(list)

    def add_to_schedule(self, sim):
        accept_inds = self.get_accept_inds(sim)
        if len(accept_inds): self.scheduler[sim.t] = accept_inds  # Add people who are willing to accept treatment to the queue

    def get_candidates(self, sim):
        ''' Get the indices of people who are candidates for treatment '''
        due_time = sim.t-self.delay/sim['dt']
        treat_candidates = np.array([], dtype=hpd.default_int)
        if len(self.scheduler[due_time]):
            treat_candidates = self.scheduler[due_time]
        return treat_candidates

    def apply(self, sim):
        self.add_to_schedule(sim)
        super().apply(sim)


#%% Products
__all__ += ['Test', 'Treatment']

class Product(hpb.FlexPretty):
    ''' Generic product implementation '''
    def administer(self, people, inds):
        ''' Adminster a Product - implemented by derived classes '''
        raise NotImplementedError


class Test(Product):
    '''
    Testing products are used within screening and triage. Their fundamental proprty is that they classify people
    into exactly one result state. They do not change anything about the People.
    '''
    def __init__(self, df, hierarchy):
        self.df = df
        self.states = df.state.unique()
        self.genotypes = df.genotype.unique()
        self.ng = len(self.genotypes)
        self.hierarchy = hierarchy # or ['high', 'medium', 'low', 'not detected'], or other

    @property
    def default_value(self):
        return len(self.hierarchy)-1

    def administer(self, sim, inds, return_format='dict'):
        '''
        Administer a testing product.
        Returns:
             if return_format=='array': an array of length len(inds) with integer entries that map each person to one of the result_states
             if return_format=='dict': a dictionary keyed by result_states with values containing the indices of people classified into this state
        '''

        # Pre-fill with the default value, which is set to be the last value in the hierarchy
        results = np.full_like(inds, fill_value=self.default_value, dtype=hpd.default_int)
        people = sim.people

        for state in self.states:
            for g,genotype in sim['genotype_map'].items():

                # gind = g if self.ng>1 else Ellipsis
                theseinds = hpu.true(people[state][g, inds])

                # Filter the dataframe to extract test results for people in this state
                df_filter = (self.df.state == state) # filter by state
                if self.ng>1: df_filter = df_filter & (self.df.genotype == genotype) # also filter by genotype, if this test is by genotype
                thisdf = self.df[df_filter] # apply filter to get the results for this state & genotype
                probs = [thisdf[thisdf.result==result].probability.values[0] for result in self.hierarchy] # Pull out the result probabilities in the order specified by the result hierarchy

                # Sort people into one of the possible result states and then update their overall results (aggregating over genotypes)
                this_gtype_results = hpu.n_multinomial(probs, len(theseinds))
                results[theseinds] = np.minimum(this_gtype_results, results[theseinds])

        if return_format=='dict':
            output = {self.hierarchy[i]:inds[results==i] for i in range(len(self.hierarchy))}
        elif return_format=='array':
            output = results

        return output


class Treatment(Product):
    '''
    Treatment products include anything used to treat cancer or precancer, as well as therapeutic vaccination.
    They change fundamental properties about People, including their prognoses and infectiousness.
    '''
    def __init__(self, efficacy):
        self.efficacy = efficacy
        self.treat_states = list(self.efficacy.keys())

    def administer(self, people, inds):
        # Loop over treatment states to determine those who (a) are successfully treated and (b) clear infection

        successfully_treated = []
        for state in self.treat_states:
            people_in_state = people[state].any(axis=0)
            treat_state_inds = inds[people_in_state[inds]]

            # Determine whether treatment is successful
            eff_probs = np.full(len(treat_state_inds), self.efficacy[state], dtype=hpd.default_float)  # Assign probabilities of treatment success
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

        return successfully_treated


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



#%% Therapeutic vaccination

__all__ += ['TherapeuticVaccination', 'RoutineTherapeutic']


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
            vx = Vaccination(vaccine=self.vaccine, prob=0, timepoints=self.dates[0])
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
            for _ in find_timepoint(self.timepoints, sim.t, interv=self, sim=sim):

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


class RoutineTherapeutic(TherapeuticVaccination):
    '''
    Routine therapeutic vaccination
    '''

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




