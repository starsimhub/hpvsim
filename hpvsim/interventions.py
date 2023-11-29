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

#%% Define data files
datafiles = sc.objdict()
for key in ['dx', 'tx', 'vx', 'txvx']:
    datafiles[key] = hpd.datadir / f'products_{key}.csv'


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

    Returns: Array of indices of people who accept
    '''
    accept_probs    = np.full_like(inds, fill_value=prob, dtype=hpd.default_float)
    accept_inds     = hpu.true(hpu.binomial_arr(accept_probs))
    return inds[accept_inds]


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
        # super().__init__(**kwargs)
        self._store_args() # Store the input arguments so the intervention can be recreated
        if label is None: label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Screen"
        self.show_label = show_label # Do not show the label by default
        self.do_plot = do_plot if do_plot is not None else False # Plot the intervention, including if None
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
                output = f"hpv.{which}({parstr})"
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
        if self.__class__.__init__ is Intervention.__init__:
            parent = f1[1].frame  # parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
        else:
            parent = f1[2].frame  # parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
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

        This method is run once as part of ``sim.finalize()`` enabling the intervention to perform any
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
        attributes, then its ``to_json`` method will need to handle those.

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
__all__ += ['RoutineDelivery', 'CampaignDelivery']

class RoutineDelivery(Intervention):
    '''
    Base class for any intervention that uses routine delivery; handles interpolation of input years.
    '''
    def __init__(self, years=None, start_year=None, end_year=None, prob=None, annual_prob=True):
        self.years      = years
        self.start_year = start_year
        self.end_year   = end_year
        self.prob       = sc.promotetoarray(prob)
        self.annual_prob = annual_prob # Determines whether the probability is annual or per timestep
        return

    def initialize(self, sim):

        # Validate inputs
        if (self.years is not None) and (self.start_year is not None or self.end_year is not None):
            errormsg = 'Provide either a list of years or a start year, not both.'
            raise ValueError(errormsg)

        # If start_year and end_year are not provided, figure them out from the provided years or the sim
        if self.years is None:
            if self.start_year is None: self.start_year = sim['start']
            if self.end_year is None:   self.end_year   = sim['end']
        else:
            self.start_year = self.years[0]
            self.end_year   = self.years[-1]

        # More validation
        if (self.start_year not in sim.yearvec) or (self.end_year not in sim.yearvec):
            errormsg = 'Years must be within simulation start and end dates.'
            raise ValueError(errormsg)

        # Adjustment to get the right end point
        adj_factor = int(1/sim['dt'])-1 if sim['dt']<1 else 1

        # Determine the timepoints at which the intervention will be applied
        self.start_point    = sc.findinds(sim.yearvec, self.start_year)[0]
        self.end_point      = sc.findinds(sim.yearvec, self.end_year)[0] + adj_factor
        self.years          = sc.inclusiverange(self.start_year, self.end_year)
        self.timepoints     = sc.inclusiverange(self.start_point, self.end_point)
        self.yearvec        = np.arange(self.start_year, self.end_year+adj_factor, sim['dt'])

        # Get the probability input into a format compatible with timepoints
        if len(self.years) != len(self.prob):
            if len(self.prob)==1:
                self.prob = np.array([self.prob[0]]*len(self.timepoints))
            else:
                errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
                raise ValueError(errormsg)
        else:
            self.prob = sc.smoothinterp(self.yearvec, self.years, self.prob, smoothness=0)

        # Lastly, adjust the probability by the sim's timestep, if it's an annual probability
        if self.annual_prob: self.prob = 1-(1-self.prob)**sim['dt']

        return


class CampaignDelivery(Intervention):
    '''
    Base class for any intervention that uses campaign delivery; handles interpolation of input years.
    '''
    def __init__(self, years, interpolate=None, prob=None, annual_prob=True):
        self.years = sc.promotetoarray(years)
        self.interpolate = True if interpolate is None else interpolate
        self.prob = sc.promotetoarray(prob)
        self.annual_prob = annual_prob
        return

    def initialize(self, sim):
        # Decide whether to apply the intervention at every timepoint throughout the year, or just once.
        if self.interpolate:
            self.timepoints = hpu.true(np.isin(np.floor(sim.yearvec), np.floor(self.years)))
        else:
            self.timepoints = hpu.true(np.isin(sim.yearvec, self.years))

        # Get the probability input into a format compatible with timepoints
        if len(self.prob) == len(self.years) and self.interpolate:
            self.prob = sc.smoothinterp(np.arange(len(self.timepoints)), np.arange(len(self.years)), self.prob, smoothness=0)
        elif len(self.prob) == 1:
            self.prob = np.array([self.prob[0]] * len(self.timepoints))
        else:
            errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
            raise ValueError(errormsg)

        # Lastly, adjust the annual probability by the sim's timestep, if it's an annual probability
        if self.annual_prob: self.prob = 1-(1-self.prob)**sim['dt']

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

        interv = hpv.dynamic_pars(condoms=dict(timepoints=10, vals={'c':0.9})) # Increase condom use amount casual partners to 90%
        interv = hpv.dynamic_pars({'beta':{'timepoints':[10, 15], 'vals':[0.005, 0.015]}, # At timepoint 10, reduce beta, then increase it again
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

    This intervention is a a kind of generalization of ``dynamic_pars`` to allow more
    flexibility in triggering multiple, arbitrary operations and to more easily assemble
    multiple changes at different times. This intervention can be used to implement scale-up
    or other changes to interventions without needing to implement time-dependency in the
    intervention itself.

    To use the intervention, simply index the intervention by ``t`` or by date.

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
            raise Exception("Use a list instead to assign multiple functions - or to really overwrite, delete the function for this day first i.e. del schedule[day] before performing schedule[day]=...")
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
    '''
    Base vaccination class for determining who will receive a vaccine.

    Args:
         product        (str/Product)   : the vaccine to use
         prob           (float/arr)     : annual probability of eligible population getting vaccinated
         age_range      (list/tuple)    : age range to vaccinate
         sex            (int/str/list)  : sex to vaccinate - accepts 0/1 or 'f'/'m' or a list of both
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of vaccination strategy
         kwargs         (dict)          : passed to Intervention()
    '''
    def __init__(self, product=None, prob=None, age_range=None, sex=None, eligibility=None, label=None, **kwargs):
        Intervention.__init__(self, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.age_range = age_range
        self.label = label
        self.eligibility = eligibility
        self._parse_product(product)

        # Deal with sex
        if sc.checktype(sex,'listlike'):
            if sc.checktype(sex[0],'str'): # If provided as ['f','m'], convert to [0,1]
                sex_list = sc.autolist()
                for isex in sex:
                    if isex=='f':
                        sex_list += 0
                    elif isex=='m':
                        sex_list += 1
                    else:
                        errormsg = f'Sex "{isex}" not understood.'
                        raise ValueError(errormsg)
                self.sex = sc.promotetoarray(sex_list)
            else:
                self.sex=sc.promotetoarray(sex)
        elif sc.checktype(sex, 'str'):  # If provided as 'f' or 'm', convert to 0 or 1
            if sex=='f':
                self.sex = np.array([0])
            elif sex=='m':
                self.sex = np.array([1])
            else:
                errormsg = f'Sex "{sex}" not understood.'
                raise ValueError(errormsg)
        else:
            self.sex = sc.promotetoarray(sex)

    def _parse_product(self, product):
        '''
        Parse the product input
        '''
        if isinstance(product, Product): # No need to do anything
            self.product=product
        elif isinstance(product, str): # Try to find it in the list of defaults
            try:
                self.product = default_vx(prod_name=product)
            except:
                errormsg = f'Could not find product {product} in the standard list.'
                raise ValueError(errormsg)
        else:
            errormsg = f'Cannot understand format of product {product} - please provide it as either a Product or string matching a default product.'
            raise ValueError(errormsg)
        return

    def initialize(self, sim):
        Intervention.initialize(self)
        self.npts = sim.res_npts
        self.n_products_used = hpb.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)
        return


    def check_eligibility(self, sim):
        '''
        Determine who is eligible for vaccination
        '''
        conditions = sim.people.alive # Start by assuming everyone alive is eligible
        if len(self.sex)==1:
            conditions = conditions & (sim.people.sex == self.sex[0]) # Filter by sex
        if self.age_range is not None:
            conditions = conditions & ((sim.people.age >= self.age_range[0]) & (sim.people.age < self.age_range[1])) # Filter by age
        if self.eligibility is not None:
            other_eligible  = sc.promotetoarray(self.eligibility(sim)) # Apply any other user-defined eligibility
            conditions      = conditions & other_eligible
        return hpu.true(conditions)


    def apply(self, sim):
        '''
        Perform vaccination by finding who's eligible for vaccination, finding who accepts, and applying the vaccine product.
        '''
        # Determine whether to apply the intervention. Apply it if no timepoints have been given or
        # if the timepoint matches one of the requested timepoints.
        if len(self.timepoints)>0 and (sim.t not in self.timepoints): do_apply = False
        else: do_apply = True
        accept_inds = np.array([])

        if do_apply:

            # Select people for screening and then record the number of screens
            eligible_inds   = self.check_eligibility(sim) # Check eligibility
            if len(self.timepoints)==0: # No timepoints provided
                prob = self.prob
            else: # Get the proportion of people who screen on this timestep
                prob = self.prob[sc.findinds(self.timepoints, sim.t)[0]]
            accept_inds = select_people(eligible_inds, prob=prob)

            if len(accept_inds):
                self.product.administer(sim.people, accept_inds) # Administer the product

                # Update people's state and dates
                sim.people.vaccinated[accept_inds] = True
                sim.people.date_vaccinated[accept_inds] = sim.t
                sim.people.doses[accept_inds] += 1

                # Update results
                idx = int(sim.t / sim.resfreq)
                new_vx_inds = hpu.ifalsei(sim.people.vaccinated, accept_inds)  # Figure out people who are getting vaccinated for the first time
                n_new_doses = sim.people.scale_flows(accept_inds)  # Scale
                n_new_people = sim.people.scale_flows(new_vx_inds)  # Scale
                sim.results['new_vaccinated'][:,idx] += n_new_people
                sim.results['new_doses'][idx] += n_new_doses
                self.n_products_used[idx] += n_new_doses

        return accept_inds


    def shrink(self, in_place=True):
        ''' Shrink vaccination intervention '''
        obj = super().shrink(in_place=in_place)
        obj.vaccinated = None
        return obj


class routine_vx(BaseVaccination, RoutineDelivery):
    '''
    Routine vaccination - an instance of base vaccination combined with routine delivery.
    See base classes for a description of input arguments.

    **Examples**::

        vx1 = hpv.routine_vx(product='bivalent', age_range=[9,10], prob=0.9, start_year=2025) # Vaccinate 90% of girls aged 9-10 every year
        vx2 = hpv.routine_vx(product='bivalent', age_range=[9,10], prob=0.9, sex=[0,1], years=np.arange(2020,2025)) # Screen 90% of girls and boys aged 9-10 every year from 2020-2025
        vx3 = hpv.routine_vx(product='quadrivalent', prob=np.linspace(0.2,0.8,5), years=np.arange(2020,2025)) # Scale up vaccination over 5 years starting in 2020
    '''

    def __init__(self, product=None, prob=None, age_range=None, sex=0, eligibility=None,
                 start_year=None, end_year=None, years=None, **kwargs):

        BaseVaccination.__init__(self, product=product, age_range=age_range, sex=sex, eligibility=eligibility, **kwargs)
        RoutineDelivery.__init__(self, prob=prob, start_year=start_year, end_year=end_year, years=years)

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseVaccination.initialize(self, sim) # Initialize this next


class campaign_vx(BaseVaccination, CampaignDelivery):
    '''
    Campaign vaccination - an instance of base vaccination combined with campaign delivery.
    See base classes for a description of input arguments.
    '''

    def __init__(self, product=None, prob=None, age_range=None, sex=0, eligibility=None,
                 years=None, interpolate=True, **kwargs):

        BaseVaccination.__init__(self, product=product, age_range=age_range, sex=sex, eligibility=eligibility, **kwargs)
        CampaignDelivery.__init__(self, prob=prob, years=years, interpolate=interpolate)

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseVaccination.initialize(self, sim) # Initialize this next


#%% Screening and triage
__all__ += ['BaseTest', 'BaseScreening', 'routine_screening', 'campaign_screening', 'BaseTriage', 'routine_triage', 'campaign_triage']


class BaseTest(Intervention):
    '''
    Base class for screening and triage.

    Args:
         product        (str/Product)   : the diagnostic to use
         prob           (float/arr)     : annual probability of eligible women receiving the diagnostic
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of screening strategy
         kwargs         (dict)          : passed to Intervention()
    '''

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        Intervention.__init__(self, **kwargs)
        self.prob           = sc.promotetoarray(prob)
        self.eligibility    = eligibility
        self._parse_product(product)

    def _parse_product(self, product):
        '''
        Parse the product input
        '''
        if isinstance(product, Product): # No need to do anything
            self.product=product
        elif isinstance(product, str): # Try to find it in the list of defaults
            try:
                self.product = default_dx(prod_name=product)
            except:
                errormsg = f'Could not find product {product} in the standard list.'
                raise ValueError(errormsg)
        else:
            errormsg = f'Cannot understand format of product {product} - please provide it as either a Product or string matching a default product.'
            raise ValueError(errormsg)
        return

    def initialize(self, sim):
        Intervention.initialize(self)
        self.npts = sim.res_npts
        self.n_products_used = hpb.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)
        self.outcomes = {k:np.array([], dtype=hpd.default_int) for k in self.product.hierarchy}
        return

    def deliver(self, sim):
        '''
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        '''
        ti = sc.findinds(self.timepoints, sim.t)[0]
        prob = self.prob[ti] # Get the proportion of people who will be tested this timestep
        eligible_inds = self.check_eligibility(sim) # Check eligibility
        accept_inds = select_people(eligible_inds, prob=prob) # Find people who accept
        if len(accept_inds):
            idx = int(sim.t / sim.resfreq)
            self.n_products_used[idx] += sim.people.scale_flows(accept_inds)
            self.outcomes = self.product.administer(sim, accept_inds) # Actually administer the diagnostic, filtering people into outcome categories
        return accept_inds

    def check_eligibility(self, sim):
        raise NotImplementedError


class BaseScreening(BaseTest):
    '''
    Base class for screening.

    Args:
        age_range (list/tuple/arr)  : age range for screening, e.g. [30,50]
        kwargs    (dict)            : passed to BaseTest
    '''
    def __init__(self, age_range=None, **kwargs):
        BaseTest.__init__(self, **kwargs) # Initialize the BaseTest object
        self.age_range = age_range or [30,50] # This is later filtered to exclude people not yet sexually active

    def check_eligibility(self, sim):
        '''
        Return an array of indices of agents eligible for screening at time t, i.e. sexually active
        females in age range, plus any additional user-defined eligibility, which often includes
        the screening interval.
        '''
        adult_females   = sim.people.is_female_adult
        in_age_range    = (sim.people.age >= self.age_range[0]) * (sim.people.age <= self.age_range[1])
        conditions      = (adult_females * in_age_range).astype(bool)
        if self.eligibility is not None:
            other_eligible  = sc.promotetoarray(self.eligibility(sim))
            conditions      = conditions * other_eligible
        return hpu.true(conditions)

    def apply(self, sim):
        '''
        Perform screening by finding who's eligible, finding who accepts, and applying the product.
        '''
        self.outcomes = {k:np.array([], dtype=hpd.default_int) for k in self.product.hierarchy}
        accept_inds = np.array([])
        if sim.t in self.timepoints:
            accept_inds = self.deliver(sim)
            sim.people.screened[accept_inds] = True
            sim.people.screens[accept_inds] += 1
            sim.people.date_screened[accept_inds] = sim.t

            # Store results
            idx = int(sim.t / sim.resfreq)
            new_screen_inds = hpu.ifalsei(sim.people.screened, accept_inds)  # Figure out people who are getting screened for the first time
            n_new_people = sim.people.scale_flows(new_screen_inds)  # Scale
            n_new_screens = sim.people.scale_flows(accept_inds)  # Scale
            sim.results['new_screened'][idx] += n_new_people
            sim.results['new_screens'][idx] += n_new_screens

        return accept_inds


class BaseTriage(BaseTest):
    '''
    Base class for triage.

    Args:
        kwargs (dict): passed to BaseTest
    '''
    def __init__(self, **kwargs):
        BaseTest.__init__(self, **kwargs)

    def check_eligibility(self, sim):
        return sc.promotetoarray(self.eligibility(sim))

    def apply(self, sim):
        self.outcomes = {k:np.array([], dtype=hpd.default_int) for k in self.product.hierarchy}
        accept_inds = np.array([])
        if sim.t in self.timepoints: accept_inds = self.deliver(sim)
        return accept_inds


class routine_screening(BaseScreening, RoutineDelivery):
    '''
    Routine screening - an instance of base screening combined with routine delivery.
    See base classes for a description of input arguments.

    **Examples**::

        screen1 = hpv.routine_screening(product='hpv', prob=0.02) # Screen 2% of the eligible population every year
        screen2 = hpv.routine_screening(product='hpv', prob=0.02, start_year=2020) # Screen 2% every year starting in 2020
        screen3 = hpv.routine_screening(product='hpv', prob=np.linspace(0.005,0.025,5), years=np.arange(2020,2025)) # Scale up screening over 5 years starting in 2020
    '''
    def __init__(self, product=None, prob=None, eligibility=None, age_range=None,
                         years=None, start_year=None, end_year=None, **kwargs):
        BaseScreening.__init__(self, product=product, age_range=age_range, eligibility=eligibility, **kwargs)
        RoutineDelivery.__init__(self, prob=prob, start_year=start_year, end_year=end_year, years=years)

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseScreening.initialize(self, sim) # Initialize this next

class campaign_screening(BaseScreening, CampaignDelivery):
    '''
    Campaign screening - an instance of base screening combined with campaign delivery.
    See base classes for a description of input arguments.

    **Examples**::

        screen1 = hpv.campaign_screening(product='hpv', prob=0.2, years=2030) # Screen 20% of the eligible population in 2020
        screen2 = hpv.campaign_screening(product='hpv', prob=0.02, years=[2025,2030]) # Screen 20% of the eligible population in 2025 and again in 2030
    '''
    def __init__(self, product=None, age_range=None, sex=None, eligibility=None,
                 prob=None, years=None, interpolate=None, **kwargs):
        BaseScreening.__init__(self, product=product, age_range=age_range, sex=sex, eligibility=eligibility, **kwargs)
        CampaignDelivery.__init__(self, prob=prob, years=years, interpolate=interpolate)

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseScreening.initialize(self, sim) # Initialize this next


class routine_triage(BaseTriage, RoutineDelivery):
    '''
    Routine triage - an instance of base triage combined with routine delivery.
    See base classes for a description of input arguments.
    
    **Examples**::

        # Example 1: Triage 40% of the eligible population in all years
        triage1 = hpv.routine_triage(product='via_triage', prob=0.4)

        # Example 2: Triage positive screens into confirmatory testing or theapeutic vaccintion
        screened_pos = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage2 = hpv.routine_triage(product='pos_screen_assessment', eligibility=screen_pos, prob=0.9, start_year=2030)
    '''
    def __init__(self, product=None, prob=None, eligibility=None, age_range=None,
                         years=None, start_year=None, end_year=None, annual_prob=None, **kwargs):
        BaseTriage.__init__(self, product=product, age_range=age_range, eligibility=eligibility, **kwargs)
        RoutineDelivery.__init__(self, prob=prob, start_year=start_year, end_year=end_year, years=years, annual_prob=annual_prob)

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseTriage.initialize(self, sim) # Initialize this next


class campaign_triage(BaseTriage, CampaignDelivery):
    '''
    Campaign triage - an instance of base triage combined with campaign delivery.
    See base classes for a description of input arguments.
    
    **Examples**::

        # Example 1: In 2030, triage all positive screens into confirmatory testing or therapeutic vaccintion
        screened_pos = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage1 = hpv.campaign_triage(product='pos_screen_assessment', eligibility=screen_pos, prob=0.9, years=2030)
    '''
    def __init__(self, product=None, age_range=None, sex=None, eligibility=None,
                 prob=None, years=None, interpolate=None, annual_prob=None, **kwargs):
        BaseTriage.__init__(self, product=product, age_range=age_range, sex=sex, eligibility=eligibility, **kwargs)
        CampaignDelivery.__init__(self, prob=prob, years=years, interpolate=interpolate, annual_prob=annual_prob)

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseTriage.initialize(self, sim) # Initialize this next



#%% Treatment interventions
__all__ += ['BaseTreatment', 'treat_num', 'treat_delay', 'BaseTxVx', 'routine_txvx', 'campaign_txvx', 'linked_txvx',]

class BaseTreatment(Intervention):
    '''
    Base treatment class.

    Args:
         product        (str/Product)   : the treatment product to use
         accept_prob     (float/arr)    : acceptance rate of treatment - interpreted as the % of women eligble for treatment who accept
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of treatment strategy
         kwargs         (dict)          : passed to Intervention()
    '''
    def __init__(self, product=None, prob=None, eligibility=None, age_range=None, **kwargs):
        Intervention.__init__(self, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.age_range = age_range or [0,99] # By default, no restrictions on treatment age


    def _parse_product(self, product):
        '''
        Parse the product input
        '''
        if isinstance(product, Product): # No need to do anything
            self.product=product
        elif isinstance(product, str): # Try to find it in the list of defaults
            try:
                self.product = default_tx(prod_name=product)
            except:
                errormsg = f'Could not find product {product} in the standard list.'
                raise ValueError(errormsg)
        else:
            errormsg = f'Cannot understand format of product {product} - please provide it as either a Product or string matching a default product.'
            raise ValueError(errormsg)
        return

    def initialize(self, sim):
        Intervention.initialize(self)
        self.n_products_used = hpb.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)
        self.outcomes = {k: np.array([], dtype=hpd.default_int) for k in ['unsuccessful', 'successful']} # Store outcomes on each timestep

    def check_eligibility(self, sim):
        '''
        Check people's eligibility for treatment
        '''
        females         = sim.people.is_female
        in_age_range    = (sim.people.age >= self.age_range[0]) * (sim.people.age <= self.age_range[1])
        alive           = sim.people.alive
        nocancer        = ~sim.people.cancerous.any(axis=0)
        conditions      = (females * in_age_range * alive * nocancer)
        return conditions

    def get_accept_inds(self, sim):
        '''
        Get indices of people who will acccept treatment; these people are then added to a queue or scheduled for receiving treatment
        '''
        accept_inds     = np.array([], dtype=hpd.default_int)
        is_eligible     = self.check_eligibility(sim) # Apply eligiblity
        if len(self.eligibility(sim)):
            eligible_inds   = hpu.itruei(is_eligible, sc.promotetoarray(self.eligibility(sim)))
            if len(eligible_inds):
                accept_inds     = select_people(eligible_inds, prob=self.prob[0])  # Select people who accept
        return accept_inds

    def get_candidates(self, sim):
        '''
        Get candidates for treatment on this timestep. Implemented by derived classes.
        '''
        raise NotImplementedError

    def apply(self, sim):
        '''
        Perform treatment by getting candidates, checking their eligibility, and then treating them.
        '''
        # Get indices of who will get treated
        treat_candidates = self.get_candidates(sim) # NB, this needs to be implemented by derived classes
        still_eligible = self.check_eligibility(sim)
        treat_inds = hpu.itruei(still_eligible, treat_candidates)

        # Store treatment and dates
        sim.people.cin_treated[treat_inds] = True
        sim.people.cin_treatments[treat_inds] += 1
        sim.people.date_cin_treated[treat_inds] = sim.t

        # Store results
        idx = int(sim.t / sim.resfreq)
        new_treat_inds = hpu.ifalsei(sim.people.cin_treated, treat_inds)  # Figure out people who are getting radiation for the first time
        n_new_cin_treatments = sim.people.scale_flows(treat_inds)  # Scale
        n_new_people = sim.people.scale_flows(new_treat_inds)  # Scale
        sim.results['new_cin_treated'][idx] += n_new_people
        sim.results['new_cin_treatments'][idx] += n_new_cin_treatments

        # Administer treatment and store products used
        self.outcomes = self.product.administer(sim, treat_inds)
        self.n_products_used[idx] += sim.people.scale_flows(treat_inds)

        return treat_inds


class treat_num(BaseTreatment):
    '''
    Treat a fixed number of people each timestep.

    Args:
         max_capacity (int): maximum number who can be treated each timestep
    '''
    def __init__(self, max_capacity=None, **kwargs):
        BaseTreatment.__init__(self, **kwargs)
        self.queue = []
        self.max_capacity = max_capacity
        return

    def add_to_queue(self, sim):
        '''
        Add people who are willing to accept treatment to the queue
        '''
        accept_inds = self.get_accept_inds(sim)
        if len(accept_inds): self.queue += accept_inds.tolist()

    def get_candidates(self, sim):
        '''
        Get the indices of people who are candidates for treatment
        '''
        treat_candidates = np.array([], dtype=hpd.default_int)
        if len(self.queue):
            if self.max_capacity is None or (self.max_capacity>len(self.queue)):
                treat_candidates = self.queue[:]
            else:
                treat_candidates = self.queue[:self.max_capacity]
        return sc.promotetoarray(treat_candidates)

    def apply(self, sim):
        '''
        Apply treatment. On each timestep, this method will add eligible people who are willing to accept treatment to a
        queue, and then will treat as many people in the queue as there is capacity for.
        '''
        self.add_to_queue(sim)
        treat_inds = BaseTreatment.apply(self, sim) # Apply method from BaseTreatment class
        self.queue = [e for e in self.queue if e not in treat_inds] # Recreate the queue, removing people who were treated
        return treat_inds


class treat_delay(BaseTreatment):
    '''
    Treat people after a fixed delay

    Args:
         delay (int): years of delay between becoming eligible for treatment and receiving treatment.
    '''
    def __init__(self, delay=None, **kwargs):
        BaseTreatment.__init__(self, **kwargs)
        self.delay = delay or 0
        self.scheduler = defaultdict(list)

    def add_to_schedule(self, sim):
        '''
        Add people who are willing to accept treatment to the treatment scehduler
        '''
        accept_inds = self.get_accept_inds(sim)
        if len(accept_inds): self.scheduler[sim.t] = accept_inds

    def get_candidates(self, sim):
        '''
        Get the indices of people who are candidates for treatment
        '''
        due_time = sim.t-self.delay/sim['dt']
        treat_candidates = np.array([], dtype=hpd.default_int)
        if len(self.scheduler[due_time]):
            treat_candidates = self.scheduler[due_time]
        return treat_candidates

    def apply(self, sim):
        '''
        Apply treatment. On each timestep, this method will add eligible people who are willing to accept treatment to a
        scheduler, and then will treat anyone scheduled for treatment on this timestep.
        '''
        self.add_to_schedule(sim)
        treat_inds = BaseTreatment.apply(self, sim)
        return treat_inds


class BaseTxVx(BaseTreatment):
    '''
    Base class for therapeutic vaccination
    '''
    def __init__(self, **kwargs):
        BaseTreatment.__init__(self, **kwargs)


    def deliver(self, sim):
        '''
        Deliver the intervention. This applies on a single timestep, whereas apply() methods
        apply on every timestep and can selectively call this method.
        '''
        is_eligible = self.check_eligibility(sim) # Apply general eligiblity

        # Apply extra user-defined eligibility conditions, if given
        if self.eligibility is not None:
            extra_conditions = sc.promotetoarray(self.eligibility(sim))

            # Checking self.eligibility() can return either a boolean array of indices. Convert to indices.
            if (len(extra_conditions) == len(is_eligible)) & (len(extra_conditions) > 0):
                if sc.checktype(extra_conditions[0], 'bool'):
                    extra_conditions = hpu.true(extra_conditions)

            # Combine the extra conditions with general eligibility
            if len(extra_conditions)>0:
                eligible_inds = hpu.itruei(is_eligible, extra_conditions)  # First make sure they're generally eligible
            else:
                eligible_inds = np.array([])

        else:
            eligible_inds = hpu.true(is_eligible)

        # Get anyone eligible and apply acceptance rates
        accept_inds = np.array([])
        if len(eligible_inds): # If so, proceed
            accept_inds = select_people(eligible_inds, prob=self.prob[0]) # Select people who accept
            new_vx_inds = hpu.ifalsei(sim.people.tx_vaccinated, accept_inds) # Figure out people who are getting vaccinated for the first time
            n_new_doses  = sim.people.scale_flows(accept_inds) # Scale
            n_new_people = sim.people.scale_flows(new_vx_inds) # Scale

            if n_new_doses:
                self.outcomes = self.product.administer(sim, accept_inds) # Administer
                sim.people.tx_vaccinated[accept_inds] = True
                sim.people.date_tx_vaccinated[accept_inds] = sim.t
                sim.people.txvx_doses[accept_inds] += 1

                idx = int(sim.t / sim.resfreq)
                sim.results['new_tx_vaccinated'][idx] += n_new_people
                sim.results['new_txvx_doses'][idx] += n_new_doses
                self.n_products_used[idx] += n_new_doses

        return accept_inds


class routine_txvx(BaseTxVx, RoutineDelivery):
    '''
    Routine delivery of therapeutic vaccine - an instance of treat_num combined
     with routine delivery. See base classes for a description of input arguments.

    **Examples**::

        txvx1 = hpv.routine_txvx(product='txvx1', prob=0.9, age_range=[25,26], start_year=2030) # Vaccinate 90% of 25yo women every year starting 2025
        txvx2 = hpv.routine_txvx(product='txvx1', prob=np.linspace(0.2,0.8,5), age_range=[25,26], years=np.arange(2030,2035)) # Scale up vaccination over 5 years starting in 2020
    '''

    def __init__(self, product=None, prob=None, age_range=None, eligibility=None,
                 start_year=None, end_year=None, years=None, annual_prob=None, **kwargs):
        BaseTxVx.__init__(self, product=product, age_range=age_range, eligibility=eligibility, **kwargs)
        RoutineDelivery.__init__(self, prob=prob, start_year=start_year, end_year=end_year, years=years, annual_prob=annual_prob)

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseTxVx.initialize(self, sim) # Initialize this next - this actually calls BaseTreatment.initialize() and ensures products will be counted

    def apply(self, sim):
        accept_inds = np.array([])
        if sim.t in self.timepoints:
            accept_inds = self.deliver(sim)
        return accept_inds


class campaign_txvx(BaseTxVx, CampaignDelivery):
    '''
    Campaign delivery of therapeutic vaccine - an instance of treat_num combined
    with campaign delivery. See base classes for a description of input arguments.
    '''

    def __init__(self, product=None, prob=None, age_range=None, eligibility=None,
                 years=None, interpolate=True, annual_prob=None, **kwargs):
        BaseTxVx.__init__(self, product=product, age_range=age_range, eligibility=eligibility, **kwargs)
        CampaignDelivery.__init__(self, prob=prob, years=years, interpolate=interpolate, annual_prob=annual_prob)

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseTxVx.initialize(self, sim) # Initialize this next - this actually calls BaseTreatment.initialize() and ensures products will be counted

    def apply(self, sim):
        accept_inds = np.array([])
        if sim.t in self.timepoints:
            accept_inds = self.deliver(sim)
        return accept_inds


class linked_txvx(BaseTxVx):
    '''
    Deliver therapeutic vaccine. This intervention should be used if TxVx delivery
    is linked to another program that determines eligibility, e.g. a screening program.
    Handling of dates is assumed to be handled by the linked intervention.
    '''
    def __init__(self, **kwargs):
        BaseTxVx.__init__(self, **kwargs)

    def apply(self, sim):
        accept_inds = BaseTxVx.deliver(self, sim)
        return accept_inds


#%% Products
__all__ += ['dx', 'tx', 'vx', 'radiation', 'default_dx', 'default_tx', 'default_vx']

class Product(hpb.FlexPretty):
    ''' Generic product implementation '''
    def administer(self, people, inds):
        ''' Adminster a Product - implemented by derived classes '''
        raise NotImplementedError


class dx(Product):
    '''
    Testing products are used within screening and triage. Their fundamental property is that they classify people
    into exactly one result state. They do not change anything about the People.
    '''
    def __init__(self, df, hierarchy=None):
        self.df = df
        self.states = df.state.unique()
        self.genotypes = df.genotype.unique()
        self.ng = len(self.genotypes)

        if hierarchy is None:
            self.hierarchy = df.result.unique() # Hierarchy is drawn from the order in which the outcomes are specified. The last unique item to be specified is the default
        else:
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
            # First check if this is a genotype specific intervention or not
            if len(np.unique(self.df.genotype)) == 1 and np.unique(self.df.genotype)[0]== 'all':
                if state == 'susceptible':
                    theseinds = hpu.true(people[state][:, inds].all(axis=0)) # Must be susceptibile for all genotypes
                else:
                    theseinds = hpu.true(people[state][:, inds].any(axis=0)) # Only need to be truly inf/cin/cancerous for one genotype
                # Filter the dataframe to extract test results for people in this state
                df_filter = (self.df.state == state)  # filter by state
                thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype
                probs = [thisdf[thisdf.result == result].probability.values[0] for result in
                         self.hierarchy]  # Pull out the result probabilities in the order specified by the result hierarchy
                # Sort people into one of the possible result states and then update their overall results (aggregating over genotypes)
                this_result = hpu.n_multinomial(probs, len(theseinds))
                results[theseinds] = np.minimum(this_result, results[theseinds])

            else:
                for g,genotype in sim['genotype_map'].items():

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
    def __init__(self, df, clearance=0.8, genotype_pars=None, imm_init=None, imm_boost=None):
        self.df = df
        self.clearance = clearance
        self.name = df.name.unique()[0]
        self.genotype_pars=genotype_pars
        self.imm_init = imm_init
        self.imm_boost = imm_boost
        self.imm_source = None
        self.states = df.state.unique()
        self.genotypes = df.genotype.unique()
        self.ng = len(self.genotypes)

    def get_people_in_state(self, state, g, sim):
        '''
        Find people within a given state/genotype. Returns indices
        '''
        if self.ng==1:  theseinds = sim.people[state].any(axis=0)
        else:           theseinds = hpu.true(sim.people[state][g, :])


    def administer(self, sim, inds, return_format='dict'):
        '''
        Loop over treatment states to determine those who are successfully treated and clear infection
        '''

        tx_successful = [] # Initialize list of successfully treated individuals
        people = sim.people

        for state in self.states: # Loop over states
            for g,genotype in sim['genotype_map'].items(): # Loop over genotypes in the sim

                theseinds = inds[hpu.true(people[state][g, inds])] # Extract people for whom this state is true for this genotype

                if len(theseinds):

                    df_filter = (self.df.state == state)  # Filter by state
                    if self.ng>1: df_filter = df_filter & (self.df.genotype == genotype)
                    thisdf = self.df[df_filter] # apply filter to get the results for this state & genotype

                    # Determine whether treatment is successful
                    efficacy = thisdf.efficacy.values[0]
                    eff_probs = np.full(len(theseinds), efficacy, dtype=hpd.default_float)  # Assign probabilities of treatment success
                    to_eff_treat = hpu.binomial_arr(eff_probs)  # Determine who will have effective treatment
                    eff_treat_inds = theseinds[to_eff_treat]
                    if len(eff_treat_inds):
                        tx_successful += list(eff_treat_inds)
                        people[state][g, eff_treat_inds] = False  # People who get treated have their CINs removed
                        people['cin'][g, eff_treat_inds] = False  # People who get treated have their CINs removed
                        people[f'date_{state}'][g, eff_treat_inds] = np.nan
                        people[f'date_cancerous'][g, eff_treat_inds] = np.nan
                        people['date_clearance'][g, eff_treat_inds] = people.t + 1
                        # Determine whether women also clear infection
                        # clearance_probs = np.full(len(eff_treat_inds), self.clearance, dtype=hpd.default_float)
                        # to_clear = hpu.binomial_arr(clearance_probs)  # Determine who will have effective treatment
                        # clear_inds = eff_treat_inds[to_clear]
                        # if len(clear_inds):
                        #     # If so, set date of clearance of infection on next timestep
                        #
                        #     people.dur_infection[g, clear_inds] = (people.t - people.date_infectious[g, clear_inds]) * people.pars['dt']

        tx_successful = np.array(list(set(tx_successful)))
        tx_unsuccessful = np.setdiff1d(inds, tx_successful)
        if return_format=='dict':
            output = {'successful':tx_successful, 'unsuccessful': tx_unsuccessful}
        elif return_format=='array':
            output = tx_successful

        if self.imm_init is not None:
            people.cell_imm[self.imm_source, inds] = hpu.sample(**self.imm_init, size=len(inds))
            people.t_imm_event[self.imm_source, inds] = people.t
        elif self.imm_boost is not None:
            people.cell_imm[self.imm_source, inds] *= self.imm_boost
            people.t_imm_event[self.imm_source, inds] = people.t
        return output


class vx(Product):
    ''' Vaccine product '''
    def __init__(self, genotype_pars=None, imm_init=None, imm_boost=None):
        self.genotype_pars = genotype_pars
        self.imm_init = imm_init
        self.imm_boost = imm_boost
        self.imm_source = None # Set during immunity initialization. Warning, fragile!!!
        if (imm_init is None and imm_boost is None) or (imm_init is not None and imm_boost is not None):
            errormsg = 'Must provide either an initial immune effect (for first doses) or an immune boosting effect (for subsequent doses), not both/neither.'
            raise ValueError(errormsg)


    def administer(self, people, inds):
        ''' Apply the vaccine to the requested people indices. '''
        inds = inds[people.alive[inds]]  # Skip anyone that is dead
        if self.imm_init is not None:
            people.peak_imm[self.imm_source, inds] = hpu.sample(**self.imm_init, size=len(inds))
        elif self.imm_boost is not None:
            people.peak_imm[self.imm_source, inds] *= self.imm_boost
        people.t_imm_event[self.imm_source, inds] = people.t



class radiation(Product):
    # Cancer treatment product
    def __init__(self, dur=None):
        self.dur = dur or dict(dist='normal', par1=18.0, par2=2.) # whatever the default duration should be

    def administer(self, sim, inds):
        people = sim.people
        new_dur_cancer = hpu.sample(**self.dur, size=len(inds))
        people.date_dead_cancer[inds] += np.ceil(new_dur_cancer / people.pars['dt'])

        # Store treatment and dates
        sim.people.cancer_treated[inds] = True
        sim.people.cancer_treatments[inds] += 1
        sim.people.date_cancer_treated[inds] = sim.t

        # Store results
        idx = int(sim.t / sim.resfreq)
        new_cctreat_inds = hpu.ifalsei(sim.people.cancer_treated, inds)  # Figure out people who are getting radiation for the first time
        n_new_radiaitons = sim.people.scale_flows(inds)  # Scale
        n_new_people = sim.people.scale_flows(new_cctreat_inds)  # Scale
        sim.results['new_cancer_treated'][idx] += n_new_people
        sim.results['new_cancer_treatments'][idx] += n_new_radiaitons

        return inds


#%% Create default products

def default_dx(prod_name=None):
    '''
    Create default diagnostic products
    '''
    dfdx = pd.read_csv(datafiles.dx) # Read in dataframe with parameters
    dxprods = dict(
        # Default primary screening diagnostics
        via             = dx(dfdx[dfdx.name == 'via'],              hierarchy=['positive', 'inadequate', 'negative']),
        lbc             = dx(dfdx[dfdx.name == 'lbc'],              hierarchy=['abnormal', 'ascus', 'inadequate', 'normal']),
        pap             = dx(dfdx[dfdx.name == 'pap'],              hierarchy=['abnormal', 'ascus', 'inadequate', 'normal']),
        colposcopy      = dx(dfdx[dfdx.name == 'colposcopy'],       hierarchy=['cancer', 'hsil', 'lsil', 'ascus', 'normal']),
        hpv             = dx(dfdx[dfdx.name == 'hpv'],              hierarchy=['positive', 'inadequate', 'negative']),
        hpv1618         = dx(dfdx[dfdx.name == 'hpv1618'],          hierarchy=['positive', 'inadequate', 'negative']),
        hpv_type        = dx(dfdx[dfdx.name == 'hpv_type'],         hierarchy=['positive_1618', 'positive_ohr', 'inadequate', 'negative']),
        # Diagnostics used to determine of subsequent care pathways
        txvx_assigner   = dx(dfdx[dfdx.name == 'txvx_assigner'],    hierarchy=['triage', 'txvx', 'none']),
        tx_assigner     = dx(dfdx[dfdx.name == 'tx_assigner'],      hierarchy=['radiation', 'excision', 'ablation', 'none']),
    )
    if prod_name is not None:   return dxprods[prod_name]
    else:                       return dxprods


def default_tx(prod_name=None):
    '''
    Create default treatment products
    '''
    dftx = pd.read_csv(datafiles.tx) # Read in dataframe with parameters
    dftxvx = pd.read_csv(datafiles.txvx)
    txprods = dict()
    for name in dftx.name.unique():
        if name =='txvx1':
            txprods[name] = tx(dftx[dftx.name==name],
                               genotype_pars=dftxvx[dftxvx.name==name],
                               imm_init=dict(dist='beta_mean', par1=0.35, par2=0.025))
        elif name == 'txvx2':
            txprods[name] = tx(dftx[dftx.name==name],
                               genotype_pars=dftxvx[dftxvx.name==name],
                               imm_boost=1.5)
        else:
            txprods[name] = tx(dftx[dftx.name == name])
    if prod_name is not None:   return txprods[prod_name]
    else:                       return txprods


def default_vx(prod_name=None):
    '''
    Create default vaccine products
    '''
    dfvx = pd.read_csv(datafiles.vx) # Read in dataframe with parameters
    vxprods = dict()
    for name in dfvx.name.unique():
        vxprods[name]       = vx(genotype_pars=dfvx[dfvx.name==name], imm_init=dict(dist='beta', par1=30, par2=2))
        vxprods[name+'2']   = vx(genotype_pars=dfvx[dfvx.name==name], imm_boost=1.2) # 2nd dose
        vxprods[name+'3']   = vx(genotype_pars=dfvx[dfvx.name==name], imm_boost=1.1) # 3rd dose
    if prod_name is not None:   return vxprods[prod_name]
    else:                       return vxprods
