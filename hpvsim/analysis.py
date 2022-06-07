'''
Additional analysis functions that are not part of the core workflow,
but which are useful for particular investigations.
'''

# import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
# from . import utils as cvu
from . import misc as hpm
from . import interventions as hpi
from . import plotting as hppl
# from . import run as cvr
from .settings import options as hpo # For setting global options
import seaborn as sns


__all__ = ['Analyzer', 'snapshot', 'age_pyramid']


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
        timepoints  (list): list of ints/strings/date objects, the days on which to take the snapshot
        die         (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs      (dict): passed to Analyzer()


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



class age_pyramid(Analyzer):
    '''
    Constructs an age/sex pyramid at specified points within the sim. Can be used with data
    Args:
        timepoints  (list): list of ints/strings/date objects, the days on which to take the snapshot
        die         (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs      (dict): passed to Analyzer()


    **Example**::
        sim = cv.Sim(analyzers=hp.age_histogram('2015', '2020'))
        sim.run()
        age_pyramid = sim['analyzers'][0]
    '''

    def __init__(self, timepoints, edges=None, age_labels=None, datafile=None, die=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        timepoints          = sc.promotetolist(timepoints) # Combine multiple timepoints
        self.timepoints     = timepoints
        self.edges          = edges # Edges of bins
        self.datafile       = datafile # Data file to load
        self.bins           = None # Age bins, calculated from edges
        self.data           = None # Store the loaded data
        self.die            = die  # Whether or not to raise an exception
        self.dates          = None # Representations in terms of years, e.g. 2020.4, set during initialization
        self.start          = None # Store the start year of the simulation
        self.age_labels     = age_labels # Labels for the age bins - will be automatically generated if not provided
        self.age_pyramids   = sc.odict() # Store the age pyramids
        return


    def initialize(self, sim):
        super().initialize()

        # Handle timepoints and dates
        self.start = sim['start']   # Store the simulation start
        self.end   = sim['end']     # Store simulation end
        if self.timepoints is None:
            self.timepoints = self.end # If no day is supplied, use the last day
        self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format
        max_hist_time = self.timepoints[-1]
        max_sim_time = sim['end']
        if max_hist_time > max_sim_time:
            errormsg = f'Cannot create histogram for {self.dates[-1]} ({max_hist_time}) because the simulation ends on {self.end} ({max_sim_time})'
            raise ValueError(errormsg)

        # Handle edges, age bins, and labels
        if self.edges is None: # Default age bins
            self.edges = np.linspace(0,100,11)
        self.bins = self.edges[:-1] # Don't include the last edge in the bins
        if self.age_labels is None:
            self.age_labels = [f'{int(self.edges[i])}-{int(self.edges[i+1])}' for i in range(len(self.edges)-1)]
            self.age_labels.append(f'{int(self.edges[-1])}+')

        # Handle the data file
        if self.datafile is not None:
            if sc.isstring(self.datafile):
                self.data = hpm.load_data(self.datafile, check_date=False)
            else:
                self.data = self.datafile # Use it directly
                self.datafile = None

            # Validate the data. Currently we only allow the same timepoints and age brackets
            data_dates = {str(float(i)) for i in self.data.year}
            if len(set(self.dates)-data_dates) or len(data_dates-set(self.dates)):
                string = f'Dates provided in the age pyramid datafile ({data_dates}) are not the same as the age pyramid dates that were requested ({self.dates}).'
                if self.die:
                    raise ValueError(string)
                else:
                    string += '\nPlots will only show requested dates, not all dates in the datafile.'
                    print(string)
            self.data_dates = data_dates

            # Validate the edges - must be the same as requested edges from the model output
            data_edges = np.array(self.data.age.unique(), dtype=float)
            if not np.array_equal(np.sort(self.edges),np.sort(data_edges)):
                errormsg = f'Age bins provided in the age pyramid datafile ({data_edges}) are not the same as the age pyramid age bins that were requested ({self.edges}).'
                raise ValueError(errormsg)

        self.initialized = True

        return


    def apply(self, sim):
        for ind in sc.findinds(self.timepoints, sim.t):

            date = self.dates[ind]
            self.age_pyramids[date] = sc.objdict() # Initialize the dictionary
            scale = sim.rescale_vec[sim.t//sim.resfreq] # Determine current scale factor
            age = sim.people.age # Get the age distribution
            self.age_pyramids[date]['bins'] = self.bins # Copy here for convenience
            for sb,sex in enumerate(['m','f']): # Loop over each sex; sb stands for sex boolean, translating the labels to 0/1
                inds = (sim.people.alive*(sim.people.sex==sb)).nonzero()[0]
                self.age_pyramids[date][sex] = np.histogram(age[inds], bins=self.edges)[0]*scale  # Bin people


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.age_pyramids.keys(), die=self.die)
        return


    def plot(self, m_color='#4682b4', f_color='#ee7989', fig_args=None, axis_args=None, data_args=None, percentages=True, **kwargs):
        '''
        Plot the age pyramids

        Args:
            m_color (hex or rgb): the color of the bars for males
            f_color (hex or rgb): the color of the bars for females
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            percentages (bool): whether to plot the pyramid as percentages or numbers
            kwargs (dict): passed to ``hp.options.with_style()``; see that function for choices
        '''

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))

        # Initialize
        fig = pl.figure(**fig_args)
        labels = list(reversed(self.age_labels))

        # Set properties depending on data
        if self.data is None: # Simple case: just plot model output
            n_plots = len(self.timepoints)
            n_rows, n_cols = sc.get_rows_cols(n_plots)
        else: # Complex case: add data
            n_cols = 2 # Data plots go in the right column
            n_rows = len(self.timepoints) # We only show plots for requested timepoints

        # Handle windows and what to plot
        pyramidsdict = self.age_pyramids
        if not len(pyramidsdict):
            errormsg = f'Cannot plot since no age pyramids were recorded (scheduled timepoints: {self.timepoints})'
            raise ValueError(errormsg)

        # Make the figure(s)
        with hpo.with_style(**kwargs):
            count=1
            for date,pyramid in pyramidsdict.items():
                pl.subplots_adjust(**axis_args)
                bins = pyramid['bins']

                # Prepare data
                pydf = pd.DataFrame(pyramid)
                if percentages:
                    pydf['m'] = pydf['m'] / sum(pydf['m'])
                    pydf['f'] = pydf['f'] / sum(pydf['f'])
                pydf['f']=-pydf['f'] # Reverse values for females to get on same axis

                # Start making plot
                ax = pl.subplot(n_rows, n_cols, count)
                sns.barplot(x='m', y='bins', data=pydf, order=np.flip(bins), orient='h', ax=ax, color=m_color)
                sns.barplot(x='f', y='bins', data=pydf, order=np.flip(bins), orient='h', ax=ax, color=f_color)
                ax.set_xlabel('Population by sex')
                ax.set_ylabel('Age group')
                ax.set_yticklabels(labels[1:])
                xticks = ax.get_xticks()
                if percentages:
                    xlabels = [f'{abs(i):.2f}' for i in xticks]
                else:
                    xlabels = [f'{sc.sigfig(abs(i), sigfigs=2, SI=True)}' for i in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels)
                ax.set_title(f'{date}')
                count +=1

                if self.data is not None:

                    if date in self.data_dates:
                        datadf = self.data[self.data.year==float(date)]
                        # Consistent naming of males and females
                        datadf.columns = datadf.columns.str[0]
                        datadf.columns = datadf.columns.str.lower()
                        if percentages:
                            datadf = datadf.assign(m=datadf['m'] / sum(datadf['m']), f=datadf['f'] / sum(datadf['f']))
                        datadf = datadf.assign(f=-datadf['f'])

                        # Start making plot
                        ax = pl.subplot(n_rows, n_cols, count)
                        sns.barplot(x='m', y='a', data=datadf, order=np.flip(bins), orient='h', ax=ax, color=m_color)
                        sns.barplot(x='f', y='a', data=datadf, order=np.flip(bins), orient='h', ax=ax, color=f_color)
                        ax.set_xlabel('Population by sex')
                        ax.set_ylabel('Age group')
                        ax.set_yticklabels(labels[1:])
                        xticks = ax.get_xticks()
                        if percentages:
                            xlabels = [f'{abs(i):.2f}' for i in xticks]
                        else:
                            xlabels = [f'{sc.sigfig(abs(i), sigfigs=2, SI=True)}' for i in xticks]
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xlabels)
                        ax.set_title(f'{date} - data')

                    count += 1


        return hppl.handle_show_return(figs=fig)

