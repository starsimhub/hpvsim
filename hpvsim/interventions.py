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


#%% Template classes for routine and campaign delivery
__all__ = ['RoutineDelivery', 'CampaignDelivery']

class RoutineDelivery(Intervention):
    ''' Routine delivery '''
    def __init__(self, years=None, start_year=None, end_year=None):
        self.years      = years
        self.start_year = start_year
        self.end_year   = end_year
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

        if len(self.years) != len(self.prob):
            if len(self.prob)==1:
                self.prob = np.array([self.prob[0]]*len(self.timepoints))
            else:
                errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
                raise ValueError(errormsg)
        else:
            self.prob = sc.smoothinterp(np.arange(len(self.timepoints)), np.arange(len(self.years)), self.prob, smoothness=0)

        self.prob *= sim['dt']

        return

class CampaignDelivery(Intervention):
    ''' Campaign delivery. '''
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

        if len(self.prob) == len(self.years) and self.interpolate:
            self.prob = sc.smoothinterp(np.arange(len(self.timepoints)), np.arange(len(self.years)), self.prob, smoothness=0)*sim['dt']
        elif len(self.prob) == 1:
            self.prob = np.array([self.prob[0]] * len(self.timepoints))
        else:
            errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
            raise ValueError(errormsg)

        return


#%% Behavior change interventions
__all__ += ['dynamic_pars', 'EventSchedule', 'set_intervention_attributes']

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
__all__ += ['BaseVaccination', 'routine_vx', 'campaign_vx']


class BaseVaccination(Intervention):
    def __init__(self, product=None, prob=None, age_range=None, sex=None, eligibility=None, label=None, **kwargs):
        super().__init__(**kwargs)
        self.product = product
        self.prob = sc.promotetoarray(prob)
        self.age_range = age_range
        self.label = label
        self.eligibility = eligibility

        # Deal with sex
        if sc.checktype(sex,'listlike'):
            if sc.checktype(sex[0],'str'): # If provided as 'f'/'m', convert to 0/1
                self.sex = np.array([0,1])
        else:
            self.sex = sc.promotetoarray(sex)


    def initialize(self, sim):
        super().initialize(sim)
        self.npts = sim.res_npts
        self.n_products_used = hpb.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)
        return


    def check_eligibility(self, sim):
        conditions = np.full(len(sim.people), True, dtype=bool)
        if len(self.sex)==1:
            conditions = conditions & (sim.people.sex == self.sex[0])
        if self.age_range is not None:
            conditions = conditions & ((sim.people.age >= self.age_range[0]) & (sim.people.age <self.age_range[1]))
        if self.eligibility is not None:
            other_eligible  = self.eligibility(sim)
            conditions      = conditions & other_eligible
        return conditions


    def apply(self, sim):
        ''' Perform vaccination '''
        if sim.t in self.timepoints:

            # Select people for screening and then record the number of screens
            ti = sc.findinds(self.timepoints, sim.t)[0]
            prob            = self.prob[ti] # Get the proportion of people who screen on this timestep
            eligible_inds   = self.check_eligibility(sim) # Check eligibility
            inds            = select_people(eligible_inds, prob=prob)

            if len(inds):
                inds = self.product.administer(sim.people, inds) # Actually change people's immunity
                # Update people's state and dates, as well as results and doses
                sim.people.vaccinated[inds] = True
                sim.people.date_vaccinated[inds] = sim.t
                sim.people.doses[inds] += 1
                idx = int(sim.t / sim.resfreq)
                sim.results['new_vaccinated'][:,idx] += len(inds)
                sim.results['new_doses'][idx] += len(inds)
                self.n_products_used[idx] += len(inds)

        return


    def shrink(self, in_place=True):
        ''' Shrink vaccination intervention '''
        obj = super().shrink(in_place=in_place)
        obj.vaccinated = None
        return obj


class routine_vx(BaseVaccination, RoutineDelivery):

    def __init__(self, product=None, prob=None, age_range=None, sex=0, eligibility=None,
                 start_year=None, end_year=None, years=None, **kwargs):

        super().__init__(product=product, prob=prob, age_range=age_range, sex=sex, eligibility=eligibility,
                 start_year=start_year, end_year=end_year, years=years, **kwargs)


class campaign_vx(BaseVaccination, CampaignDelivery):

    def __init__(self, product=None, prob=None, age_range=None, sex=0, eligibility=None,
                 years=None, interpolate=True, **kwargs):

        super().__init__(product=product, prob=prob, age_range=age_range, sex=sex, eligibility=eligibility,
                 years=years, interpolate=interpolate, **kwargs)



#%% Screening and triage
__all__ += ['BaseScreening', 'routine_screening', 'campaign_screening', 'triage']


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

    def __init__(self, product=None, prob=None, eligibility=None,
                 age_range=None, store_states=None,
                 label=None, verbose=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.product        = product # The test product being used to screen people
        self.prob           = sc.promotetoarray(prob) # Annual probability of being screened
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
        super().initialize(sim)
        self.npts = sim.res_npts
        self.n_products_used = hpb.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)
        self.outcomes = {k:[] for k in self.product.hierarchy}
        return


    def apply(self, sim):
        ''' TBC '''
        if sim.t in self.timepoints:

            # Select people for screening and then record the number of screens
            ti = sc.findinds(self.timepoints, sim.t)[0]
            prob            = self.prob[ti] # Get the proportion of people who screen on this timestep
            eligible_inds   = self.check_eligibility(sim) # Check eligibility
            screen_inds     = select_people(eligible_inds, prob=prob)

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
        conditions      = (active_females & in_age_range)
        if self.eligibility is not None:
            other_eligible  = self.eligibility(sim)
            conditions      = conditions & other_eligible
        return conditions


class routine_screening(BaseScreening, RoutineDelivery):
    '''
    Routine screening.
    Example:
        screen1 = hpv.routine_screening('hpv', 0.02) # Screen 2% of the eligible population every year
        screen2 = hpv.routine_screening('hpv', 0.02, start_year=2020) # Screen 2% every year starting in 2020
        screen3 = hpv.routine_screening('hpv', np.linspace(0.005,0.025,5), years=np.arange(2020,2025)) # Scale up screening over 5 years starting in 2020
    '''
    def __init__(self, product=None, prob=None, eligibility=None, age_range=None, label=None, verbose=False,
                         years=None, start_year=None, end_year=None, **kwargs):
        super().__init__(product=product, prob=prob, eligibility=eligibility, age_range=age_range, label=label, verbose=verbose,
                         years=years, start_year=start_year, end_year=end_year, **kwargs)


class campaign_screening(BaseScreening, CampaignDelivery):
    '''
    Campaign screening.
    Example:
        campaign_screen = hpv.CampaignScreening('hpv', 0.2, years=[2020, 2025]) # Screen 20% of the eligible population in 2020 and again in 2025
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class triage(Intervention):
    '''
    Triage.
    TODO: is it possible to get rid of this and leave it as a type of screening??
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
__all__ += ['BaseTreatment', 'treat_num', 'treat_delay']

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


class treat_num(BaseTreatment):
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


class treat_delay(BaseTreatment):
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
__all__ += ['dx', 'tx', 'vx', 'txvx']

class Product(hpb.FlexPretty):
    ''' Generic product implementation '''
    def administer(self, people, inds):
        ''' Adminster a Product - implemented by derived classes '''
        raise NotImplementedError


class dx(Product):
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


class tx(Product):
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


class vx(Product):
    ''' Vaccine product '''
    def __init__(self, genotype_pars=None, imm_init=None, imm_boost=None, prophylactic=True):
        self.genotype_pars = genotype_pars
        self.imm_init = imm_init
        self.imm_boost = imm_boost
        if (imm_init is None and imm_boost is None) or (imm_init is not None and imm_boost is not None):
            errormsg = 'Must provide either an initial immune effect (for first doses) or an immune boosting effect (for subsequent doses), not both/neither.'
            raise ValueError(errormsg)
        self.prophylactic = prophylactic # Whether this vaccine has a prophylactic effect


    def administer(self, people, inds):
        ''' Apply the vaccine to the requested people indices. '''
        inds = inds[people.alive[inds]]  # Skip anyone that is dead
        imm_source = 0 # WARNING TEMPPPPPP <<<<<<<<<<!!!!!!!!!!!
        print('warning, using placeholder imm_source')
        if self.imm_init is not None:
            people.peak_imm[imm_source, inds] = hpu.sample(**self.imm_init, size=len(inds))
        elif self.imm_boost is not None:
            people.peak_imm[imm_source, inds] *= self.imm_boost
        people.t_imm_event[imm_source, inds] = people.t
        return inds


# class txvx(Product):
#     ''' Therapeutic vaccine product '''
#
#     def __init__(self, pars, hierarchy):
#         self.df = df
#         self.hierarchy = hierarchy
#         self.states = df.state.unique()
#         self.genotypes = df.genotype.unique()
#         self.ng = len(self.genotypes)
#
#     @property
#     def default_value(self):
#         return len(self.hierarchy)-1
#
#     def administer(self, people, inds):
#         pass

# class RadiationTherapy(Product):
#     # Cancer treatment product
#     def __init__(self, dur=None):
#         self.dur = dur or dict(dist='normal', par1=18.0, par2=2.) # whatever the default duration should be
#
#     def administer(self, people, inds):
#         new_dur_cancer = hpu.sample(**self.dur, size=len(inds))
#         people.date_dead_cancer[inds] += np.ceil(new_dur_cancer / people.pars['dt'])
#         people.treated[inds] = True
#         people.date_treated[inds] = people.t
#         return inds
#


# #%% Therapeutic vaccination
#
# __all__ += ['TherapeuticVaccination', 'RoutineTherapeutic']
#
#
# class TherapeuticVaccination(Intervention, Product):
#     '''
#         Base class to apply a therapeutic vaccine to a subset of the population. Can be implemented as
#         a campaign-style or routine administration within S&T.
#
#         This class implements the mechanism of delivering a therapeutic vaccine.
#
#         '''
#
#     def __init__(self, timepoints, prob=None, LTFU=None,  doses=None, interval=None, efficacy=None, subtarget=None,
#                  proph=False, vaccine='bivalent_1dose', **kwargs):
#         super().__init__(**kwargs)  # Initialize the Intervention object
#         self.subtarget = subtarget
#         if prob is None: # Populate default value of probability: 1 if no subtargeting, 0 if subtargeting
#             prob = 1.0 if subtarget is None else 0.0
#         self.prob      = prob
#         self.LTFU = LTFU
#         self.timepoints = timepoints
#         self.doses = doses or 2
#         self.interval = interval or 0.5  # Interval between doses in years
#         self.prophylactic = proph # whether to deliver a single-dose prophylactic vaccine at first dose
#         self.vaccine = vaccine # which vaccine to deliver
#         self.treat_states = ['precin', 'latent', 'cin1', 'cin2', 'cin3']
#         self.efficacy = efficacy or dict(  # default efficacy decreases as dysplasia increases
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
#             latent=dict(
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
#                 hpv16=[0.1, 0.6],
#                 hpv18=[0.1, 0.6],
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
#         )
#         return
#
#     def initialize(self, sim):
#         super().initialize()
#         self.timepoints, self.dates = sim.get_t(self.timepoints,
#                                                 return_date_format='str')  # Ensure timepoints and dates are in the right format
#         self.second_dose_timepoints = [None] * sim.npts  # People who get second dose (if relevant)
#         if self.prophylactic:
#             # Initialize a prophylactic vaccine intervention to reference later
#             vx = Vaccination(vaccine=self.vaccine, prob=0, timepoints=self.dates[0])
#             vx.initialize(sim)
#             self.prophylactic_vaccine = vx
#         return
#
#     def administer(self, people, inds):
#
#         #Extract parameters that will be used below
#         ng = people.pars['n_genotypes']
#         genotype_map = people.pars['genotype_map']
#
#         # Find those who are getting first dose
#         people_not_vaccinated = hpu.false(people.tx_vaccinated)
#         first_dose_inds = np.intersect1d(people_not_vaccinated, inds)
#         people.tx_vaccinated[first_dose_inds] = True
#
#         people.txvx_doses[inds] += 1
#
#         # Find those who are getting second dose today
#         second_dose_inds = np.setdiff1d(inds, first_dose_inds)
#
#         # Deliver vaccine and update prognoses TODO: immune response in those without infection/lesion
#         for inds_to_treat, dose in zip([first_dose_inds, second_dose_inds], [0,1]):
#             for state in self.treat_states:
#                 for g in range(ng):
#                     people_in_state = hpu.true(people[state][g,inds_to_treat])
#                     treat_state_inds = inds_to_treat[people_in_state]
#
#                     # Determine whether treatment is successful
#                     eff_probs = np.full(len(treat_state_inds), self.efficacy[state][genotype_map[g]][dose],
#                                         dtype=hpd.default_float)  # Assign probabilities of treatment success
#                     to_eff_treat = hpu.binomial_arr(eff_probs)  # Determine who will have effective treatment
#                     eff_treat_inds = treat_state_inds[to_eff_treat]
#                     people[state][g, eff_treat_inds] = False  # People who are successfully treated
#                     people[f'date_{state}'][g, eff_treat_inds] = np.nan
#                     hpi.update_peak_immunity(people, eff_treat_inds, imm_pars=people.pars, imm_source=g) # Get natural immune memory
#
#         return
#
#     def select_people(self, sim):
#
#         vacc_inds = np.array([], dtype=int)  # Initialize in case no one gets their first dose
#
#         if sim.t >= np.min(self.timepoints):
#
#             # Vaccinate people with their first dose
#             for _ in find_timepoint(self.timepoints, sim.t, interv=self, sim=sim):
#
#                 vacc_probs = np.zeros(len(sim.people))
#
#                 # Find eligible people
#                 vacc_probs[hpu.true(~sim.people.alive)] *= 0.0  # Do not vaccinate dead people
#                 eligible_inds = sc.findinds(~sim.people.tx_vaccinated)
#                 vacc_probs[eligible_inds] = self.prob  # Assign equal vaccination probability to everyone
#
#                 # Apply any subtargeting
#                 if self.subtarget is not None:
#                     subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
#                     vacc_probs[subtarget_inds] = subtarget_vals  # People being explicitly subtargeted
#
#                 vacc_inds = hpu.true(hpu.binomial_arr(vacc_probs))  # Calculate who actually gets vaccinated
#
#                 if len(vacc_inds):
#                     if self.interval is not None:
#                         # Schedule the doses
#                         second_dose_timepoints = sim.t + int(self.interval/sim['dt'])
#                         if second_dose_timepoints < sim.npts:
#                             self.second_dose_timepoints[second_dose_timepoints] = vacc_inds
#
#             idx = int(sim.t / sim.resfreq)
#             sim.results['new_txvx_vaccinated'][idx] += len(vacc_inds)
#
#             # Also, if appropriate, vaccinate people with their second doses
#             vacc_inds_dose2 = self.second_dose_timepoints[sim.t]
#             if vacc_inds_dose2 is not None:
#                 if self.LTFU is not None:
#                     vacc_probs = np.full(len(vacc_inds_dose2), (1-self.LTFU))
#                     vacc_inds_dose2 = vacc_inds_dose2[hpu.true(hpu.binomial_arr(vacc_probs))]
#                 vacc_inds = np.concatenate((vacc_inds, vacc_inds_dose2), axis=None)
#
#             sim.results['new_txvx_doses'][idx] += len(vacc_inds)
#
#
#         return vacc_inds
#
#     def apply(self, sim):
#         ''' Perform vaccination each timestep '''
#         inds = self.select_people(sim)
#         if len(inds):
#             self.administer(sim.people, inds)
#             if self.prophylactic:
#                 inds_to_vax = inds[hpu.false(sim.people.vaccinated[inds])]
#                 self.prophylactic_vaccine.vaccinate(sim, inds_to_vax)
#         return inds
#
#
# class RoutineTherapeutic(TherapeuticVaccination):
#     '''
#     Routine therapeutic vaccination
#     '''
#
#     def __init__(self, *args, age_range, coverage, **kwargs):
#         super().__init__(*args, **kwargs, subtarget=self.subtarget_function)
#         self.age_range = age_range
#         self.coverage = sc.promotetoarray(coverage)
#         if len(self.coverage) == 1:
#             self.coverage = self.coverage * np.ones_like(self.timepoints)
#
#     def subtarget_function(self, sim):
#         inds = sc.findinds((sim.people.age >= self.age_range[0]) & (sim.people.age < self.age_range[1]) & (sim.people.is_female))
#         coverage = self.coverage[self.timepoints == sim.t][0]
#         return {'vals': coverage * np.ones_like(inds), 'inds': inds}
#
#
#
#
