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



class age_histogram(Analyzer):
    '''
    Calculate statistics across age bins, including histogram plotting functionality.

    Args:
        timepoints (list): list of ints/strings/dates, representing the timepoints on which to calculate the histograms (default: last timepoint)
        states  (list): which states of people to record (default: alive)
        edges   (list): edges of age bins to use (default: 10 year bins from 0 to 100)
        datafile (str): the name of the data file to load in for comparison, or a dataframe of data (optional)
        sim      (Sim): only used if the analyzer is being used after a sim has already been run
        die     (bool): whether to raise an exception if dates are not found (default true)
        kwargs  (dict): passed to Analyzer()

    **Examples**::

        sim = hp.Sim(analyzers=hp.age_histogram())
        sim.run()

        agehist = sim.get_analyzer()
        agehist = cv.age_histogram(sim=sim) # Alternate method
        agehist.plot()
    '''

    def __init__(self, timepoints=None, states=None, edges=None, datafile=None, sim=None, die=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        self.timepoints= timepoints # To be converted to integer representations
        self.edges     = edges # Edges of age bins
        self.states    = states # States to save
        self.datafile  = datafile # Data file to load
        self.die       = die # Whether to raise an exception if dates are not found
        self.bins      = None # Age bins, calculated from edges
        self.dates     = None # String representations of dates
        self.start     = None # Store the start date of the simulation
        self.data      = None # Store the loaded data
        self.hists = sc.odict() # Store the actual snapshots
        self.window_hists = None # Store the histograms for individual windows -- populated by compute_windows()
        if sim is not None: # Process a supplied simulation
            self.from_sim(sim)
        return


    def from_sim(self, sim):
        ''' Create an age histogram from an already run sim '''
        if self.timepoints is not None: # pragma: no cover
            errormsg = 'If a simulation is being analyzed post-run, no day can be supplied: only the last day of the simulation is available'
            raise ValueError(errormsg)
        self.initialize(sim)
        self.apply(sim)
        return


    def initialize(self, sim):
        super().initialize()

        # Handle days
        self.start = sim['start'] # Store sim start
        self.end   = sim['end'] # Store sim end
        if self.timepoints is None:
            self.timepoints = -1 # If no timepoint is supplied, use the last timepoints
        self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format

        # Handle edges and age bins
        if self.edges is None: # Default age bins
            self.edges = np.linspace(0,100,11)
        self.bins = self.edges[:-1] # Don't include the last edge in the bins

        # Handle states
        if self.states is None:
            self.states = ['alive', 'other_dead']
        self.states = sc.promotetolist(self.states)
        for s,state in enumerate(self.states):
            self.states[s] = state.replace('date_', '') # Allow keys starting with date_ as input, but strip it off here

        # Handle the data file
        if self.datafile is not None:
            if sc.isstring(self.datafile):
                self.data = hpm.load_data(self.datafile, check_date=False)
            else:
                self.data = self.datafile # Use it directly
                self.datafile = None

        return


    def apply(self, sim):
        for ind in sc.findinds(self.timepoints, sim.t):
            date = self.dates[ind] # Find the date for this index
            self.hists[date] = sc.objdict() # Initialize the dictionary
            age    = sim.people.age # Get the age distribution,since used heavily
            self.hists[date]['bins'] = self.bins # Copy here for convenience
            for state in self.states: # Loop over each state
                inds = sim.people.defined(f'date_{state}') # Pull out people for which this state is defined
                self.hists[date][state] = np.histogram(age[inds], bins=self.edges)[0] # Actually count the people


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.hists.keys(), die=self.die)
        return


    def get(self, key=None):
        ''' Retrieve a specific histogram from the given key (int, str, or date) '''
        if key is None:
            key = self.days[0]
        date = key
        if date in self.hists:
            hists = self.hists[date]
        else: # pragma: no cover
            dates = ', '.join(list(self.hists.keys()))
            errormsg = f'Could not find histogram date {date}: choices are {dates}'
            raise sc.KeyNotFoundError(errormsg)
        return hists


    def compute_windows(self):
        ''' Convert cumulative histograms to windows '''
        if len(self.hists)<2:
            errormsg = 'You must have at least two dates specified to compute a window'
            raise ValueError(errormsg)

        self.window_hists = sc.objdict()
        for d,end,hists in self.hists.enumitems():
            if d==0: # Copy the first one
                start = self.start
                self.window_hists[f'{start} to {end}'] = self.hists[end]
            else:
                start = self.dates[d-1]
                datekey = f'{start} to {end}'
                self.window_hists[datekey] = sc.objdict() # Initialize the dictionary
                self.window_hists[datekey]['bins'] = self.hists[end]['bins']
                for state in self.states: # Loop over each state
                    self.window_hists[datekey][state] = self.hists[end][state] - self.hists[start][state]

        return


#     def plot(self, windows=False, width=0.8, color='#F8A493', fig_args=None, axis_args=None, data_args=None, **kwargs):
#         '''
#         Simple method for plotting the histograms.

#         Args:
#             windows (bool): whether to plot windows instead of cumulative counts
#             width (float): width of bars
#             color (hex or rgb): the color of the bars
#             fig_args (dict): passed to pl.figure()
#             axis_args (dict): passed to pl.subplots_adjust()
#             data_args (dict): 'width', 'color', and 'offset' arguments for the data
#             kwargs (dict): passed to ``cv.options.with_style()``; see that function for choices
#         '''

#         # Handle inputs
#         fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
#         axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
#         d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))

#         # Initialize
#         n_plots = len(self.states)
#         n_rows, n_cols = sc.get_rows_cols(n_plots)
#         figs = []

#         # Handle windows and what to plot
#         if windows:
#             if self.window_hists is None:
#                 self.compute_windows()
#             histsdict = self.window_hists
#         else:
#             histsdict = self.hists
#         if not len(histsdict): # pragma: no cover
#             errormsg = f'Cannot plot since no histograms were recorded (schuled days: {self.days})'
#             raise ValueError(errormsg)

#         # Make the figure(s)
#         with cvo.with_style(**kwargs):
#             for date,hists in histsdict.items():
#                 figs += [pl.figure(**fig_args)]
#                 pl.subplots_adjust(**axis_args)
#                 bins = hists['bins']
#                 barwidth = width*(bins[1] - bins[0]) # Assume uniform width
#                 for s,state in enumerate(self.states):
#                     ax = pl.subplot(n_rows, n_cols, s+1)
#                     ax.bar(bins, hists[state], width=barwidth, facecolor=color, label=f'Number {state}')
#                     if self.data and state in self.data:
#                         data = self.data[state]
#                         ax.bar(bins+d_args.offset, data, width=barwidth*d_args.width, facecolor=d_args.color, label='Data')
#                     ax.set_xlabel('Age')
#                     ax.set_ylabel('Count')
#                     ax.set_xticks(ticks=bins)
#                     ax.legend()
#                     preposition = 'from' if windows else 'by'
#                     ax.set_title(f'Number of people {state} {preposition} {date}')

#         return cvpl.handle_show_return(figs=figs)

