'''
Additional analysis functions that are not part of the core workflow,
but which are useful for particular investigations.
'''

# import os
import numpy as np
import pylab as pl
# import pandas as pd
import sciris as sc
# from . import utils as cvu
from . import misc as hpm
from . import interventions as hpi
# from . import plotting as cvpl
# from . import run as cvr
# from .settings import options as cvo # For setting global options


__all__ = ['Analyzer', 'snapshot', 'age_histogram']


class Analyzer(sc.prettyobj):
    '''
    Base class for analyzers. Based on the Intervention class. Analyzers are used
    to provide more detailed information about a simulation than is available by
    default -- for example, pulling states out of sim.people on a particular timestep
    before it gets updated in the next timestep.

    To retrieve a particular analyzer from a sim, use sim.get_analyzer().

    Args:
        label (str): a label for the Analyzer (used for ease of identification)
    '''

    def __init__(self, label=None):
        if label is None:
            label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Record ages"
        self.initialized = False
        self.finalized = False
        return


    def __call__(self, *args, **kwargs):
        # Makes Analyzer(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)


    def initialize(self, sim=None):
        '''
        Initialize the analyzer, e.g. convert date strings to integers.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize analyzer

        This method is run once as part of `sim.finalize()` enabling the analyzer to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized:
            raise RuntimeError('Analyzer already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply analyzer at each time point. The analyzer has full access to the
        sim object, and typically stores data/results in itself. This is the core
        method which each analyzer object needs to implement.

        Args:
            sim: the Sim instance
        '''
        raise NotImplementedError


    def shrink(self, in_place=False):
        '''
        Remove any excess stored data from the intervention; for use with sim.shrink().

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place:
            return self
        else:
            return sc.dcp(self)


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. This method will attempt to JSONify each attribute of the
        intervention, skipping any that fail.

        Returns:
            JSON-serializable representation
        '''
        # Set the name
        json = {}
        json['analyzer_name'] = self.label if hasattr(self, 'label') else None
        json['analyzer_class'] = self.__class__.__name__

        # Loop over the attributes and try to process
        attrs = self.__dict__.keys()
        for attr in attrs:
            try:
                data = getattr(self, attr)
                try:
                    attjson = sc.jsonify(data)
                    json[attr] = attjson
                except Exception as E:
                    json[attr] = f'Could not jsonify "{attr}" ({type(data)}): "{str(E)}"'
            except Exception as E2:
                json[attr] = f'Could not jsonify "{attr}": "{str(E2)}"'
        return json


def validate_recorded_dates(sim, requested_dates, recorded_dates, die=True):
    '''
    Helper method to ensure that dates recorded by an analyzer match the ones
    requested.
    '''
    requested_dates = sorted(list(requested_dates))
    recorded_dates = sorted(list(recorded_dates))
    if recorded_dates != requested_dates: # pragma: no cover
        errormsg = f'The dates {requested_dates} were requested but only {recorded_dates} were recorded: please check the dates fall between {sim["start_day"]} and {sim["start_day"]} and the sim was actually run'
        if die:
            raise RuntimeError(errormsg)
        else:
            print(errormsg)
    return



class snapshot(Analyzer):
    '''
    Analyzer that takes a "snapshot" of the sim.people array at specified points
    in time, and saves them to itself. To retrieve them, you can either access
    the dictionary directly, or use the get() method.

    Args:
        days   (list): list of ints/strings/date objects, the days on which to take the snapshot
        args   (list): additional day(s)
        die    (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs (dict): passed to Analyzer()


    **Example**::

        sim = cv.Sim(analyzers=cv.snapshot('2015.4', '2020'))
        sim.run()
        snapshot = sim['analyzers'][0]
        people = snapshot.snapshots[0]            # Option 1
        people = snapshot.snapshots['2020']       # Option 2
        people = snapshot.get('2020')             # Option 3
        people = snapshot.get(34)                 # Option 4
        people = snapshot.get()                   # Option 5
    '''

    def __init__(self, timepoints, *args, die=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        timepoints = sc.promotetolist(timepoints) # Combine multiple timepoints
        timepoints.extend(args) # Include additional arguments, if present (assume these are more timepoints)
        self.timepoints     = timepoints 
        self.die            = die  # Whether or not to raise an exception
        self.dates          = None # Representations in terms of years, e.g. 2020.4, set during initialization
        self.start          = None # Store the start year of the simulation
        self.snapshots      = sc.odict() # Store the actual snapshots
        return


    def initialize(self, sim):
        self.start = sim['start'] # Store the simulation start
        self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format
        self.initialized = True
        return


    def apply(self, sim):
        for ind in sc.findinds(self.timepoints, sim.t):
            date = self.dates[ind]
            self.snapshots[date] = sc.dcp(sim.people) # Take snapshot!


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.snapshots.keys(), die=self.die)
        return


    def get(self, key=None):
        ''' Retrieve a snapshot from the given key (int, str, or date) '''
        if key is None:
            key = self.dates[0]
        date = key # TODO: consider ways to make this more robust
        if date in self.snapshots:
            snapshot = self.snapshots[date]
        else: 
            dates = ', '.join(list(self.snapshots.keys()))
            errormsg = f'Could not find snapshot date {date}: choices are {self.dates}'
            raise sc.KeyNotFoundError(errormsg)
        return snapshot



