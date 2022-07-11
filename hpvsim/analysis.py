'''
Additional analysis functions that are not part of the core workflow,
but which are useful for particular investigations.
'''

# import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from . import utils as hpu
from . import misc as hpm
from . import interventions as hpi
from . import plotting as hppl
from . import defaults as hpd
# from . import run as cvr
from .settings import options as hpo # For setting global options
import seaborn as sns


__all__ = ['Analyzer', 'snapshot', 'age_pyramid', 'age_results']


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
        errormsg = f'The dates {requested_dates} were requested but only {recorded_dates} were recorded: please check the dates fall between {sim["start"]} and {sim["end"]} and the sim was actually run'
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
        sim = cv.Sim(analyzers=hp.age_pyramid('2015', '2020'))
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


    def plot(self, m_color='#4682b4', f_color='#ee7989', fig_args=None, axis_args=None, data_args=None,
             percentages=True, do_save=None, fig_path=None, do_show=True, **kwargs):
        '''
        Plot the age pyramids

        Args:
            m_color (hex or rgb): the color of the bars for males
            f_color (hex or rgb): the color of the bars for females
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            percentages (bool): whether to plot the pyramid as percentages or numbers
            do_save (bool): whether to save
            fig_path (str or filepath): filepath to save to
            do_show (bool): whether to show the figure
            kwargs (dict): passed to ``hp.options.with_style()``; see that function for choices
        '''

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))
        all_args = sc.mergedicts(fig_args, axis_args, d_args)

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
        xlabel = 'Share of population by sex' if percentages else 'Population by sex'
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
                ax.set_xlabel(xlabel)
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
                        ax.set_xlabel(xlabel)
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

        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)



class age_results(Analyzer):
    '''
    Constructs results by age at specified points within the sim. Can be used with data
    Args:
        timepoints  (list): list of ints/strings/date objects, timepoints at which to generate by-age results
        results     (list): list of strings, results to generate
        age_standardized (bool): whether or not to provide age-standardized results
        die         (bool): whether or not to raise an exception if errors are found
        kwargs      (dict): passed to Analyzer()

    **Example**::
        sim = hp.Sim(analyzers=hp.age_results(timepoints=['2015', '2020'], results=['hpv_incidence', 'total_cancers']))
        sim.run()
        age_results = sim['analyzers'][0]
    '''

    def __init__(self, timepoints, edges=None, result_keys=None, age_labels=None, age_standardized=False, datafile=None, die=False, **kwargs):
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
        self.age_standard   = None
        self.age_standardized = age_standardized # Whether or not to compute age-standardized results
        self.result_keys    = result_keys # Store the result keys
        self.results        = sc.odict() # Store the age results
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
            errormsg = f'Cannot create age results for {self.dates[-1]} ({max_hist_time}) because the simulation ends on {self.end} ({max_sim_time})'
            raise ValueError(errormsg)

        # Handle dt - if we're storing annual results we'll need to aggregate them over
        # several consecutive timesteps
        self.dt = sim['dt']
        self.calcpoints = []
        for tp in self.timepoints:
            self.calcpoints += [tp+i for i in range(int(1/self.dt))]

        # Handle edges, age bins, and labels
        if self.edges is None: # Default age bins
            self.edges = np.linspace(0,100,11)
        self.bins = self.edges[:-1] # Don't include the last edge in the bins
        if self.age_labels is None:
            self.age_labels = [f'{int(self.bins[i])}-{int(self.bins[i+1])}' for i in range(len(self.bins)-1)]
            self.age_labels.append(f'{int(self.bins[-1])}+')

        if self.age_standardized:
            self.age_standard = self.get_standard_population()
            self.edges = self.age_standard[0]
            self.bins = self.edges[:-1]  # Don't include the last edge in the bins
            self.age_labels = [f'{int(self.bins[i])}-{int(self.bins[i + 1])}' for i in range(len(self.bins) - 1)]
            self.age_labels.append(f'{int(self.bins[-1])}+')
        else:
            self.age_standard = np.array([self.edges, np.full(len(self.edges), 1)])

        # Handle result keys
        choices = sim.result_keys()
        if self.result_keys is None:
            self.result_keys = ['total_cancers'] # Defaults
        for rk in self.result_keys:
            if rk not in choices:
                strm = '\n'.join(choices)
                errormsg = f'Cannot compute age results for {rk}. Please enter one of the standard sim result_keys to the age_results analyzer; choices are {strm}.'
                raise ValueError(errormsg)

        # Store genotypes
        self.ng = sim['n_genotypes']
        self.glabels = [g.label for g in sim['genotypes']]

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
                string = f'Dates provided in the age result datafile ({data_dates}) are not the same as the age result dates that were requested ({self.dates}).'
                if self.die:
                    raise ValueError(string)
                else:
                    string += '\nPlots will only show requested dates, not all dates in the datafile.'
                    print(string)
            self.data_dates = data_dates

            # Validate the edges - must be the same as requested edges from the model output
            data_bins = np.array(self.data.age.unique(), dtype=float)
            if not np.array_equal(np.sort(self.bins), np.sort(data_bins)):
                errormsg = f'Age bins provided in the age result datafile ({data_bins}) are not the same as the age result age bins that were requested ({self.edges}).'
                raise ValueError(errormsg)

            # Validate the result keys - must be the same as requested edges from the model output
            data_result_keys_array = self.data.name.unique()
            data_result_keys = []
            for drk in data_result_keys_array:
                if drk not in self.result_keys:
                    string = f'The age result datafile contains {drk}, which is not one of the results were requested ({self.result_keys}).'
                    if self.die:
                        raise ValueError(string)
                    else:
                        string += '\nPlots will only show requested results, not all results in the datafile.'
                        print(string)
                else:
                    data_result_keys.append(drk)
            self.data_result_keys = data_result_keys

        # Handle variable names (TODO, should this be centralized somewhere?)
        self.mapping = {
            'infections': ['date_infectious', 'infectious'],
            'cin':  ['date_cin1', 'cin'], # Not a typo - the date the get a CIN is the same as the date they get a CIN1
            'cin1': ['date_cin1', 'cin1'],
            'cin2': ['date_cin2', 'cin2'],
            'cin3': ['date_cin3', 'cin3'],
            'cancers': ['date_cancerous', 'cancerous'],
        }

        # Store colors
        self.result_properties = sc.objdict()
        for rkey in self.result_keys:
            self.result_properties[rkey] = sc.objdict()
            self.result_properties[rkey].color = sim.results[rkey].color
            self.result_properties[rkey].name = sim.results[rkey].name

        self.initialized = True

        return


    def get_standard_population(self):
        '''
        Returns the WHO standard population for computation of age-standardized rates
        '''

        age_standard = np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                                [0.08860, 0.08690, 0.086, 0.0847, 0.0822, 0.0793, 0.0761, 0.0715, 0.0659, 0.0604,
                                 0.0537, 0.0455, 0.0372, 0.0296, 0.0221, 0.0152, 0.0091, 0.0044, 0.0015, 0.0004,
                                 0.00005]])
        return age_standard


    def apply(self, sim):
        ''' Calculate age results '''

        # Shorten variables that are used a lot
        na = len(self.bins)
        ng = self.ng

        # This is the a timepoint where age results were requested. Start by initializing storage
        # If the requested result is a stock, we can just calculate it for this timepoint
        if sim.t in self.timepoints:
            ind = sc.findinds(self.timepoints, sim.t)[0] # Get the index
            date = self.dates[ind] # Create the date which will be used to key the results
            self.results[date] = sc.objdict() # Initialize the dictionary for result storage
            self.results[date]['bins'] = self.bins # Copy here for convenience
            age = sim.people.age # Get the age distribution
            scale = sim.rescale_vec[sim.t//sim.resfreq] # Determine current scale factor


            for rkey in self.result_keys: # Loop over each result, but only stocks are calculated here

                # Initialize storage
                size = na if 'total' in rkey else (ng,na)
                self.results[date][rkey] = np.zeros(size)

                # Both annual stocks and prevalence require us to calculate the current stocks.
                # Unlike incidence, these don't have to be aggregated over multiple timepoints.
                if rkey[0] == 'n' or 'prevalence' in rkey:
                    attr = rkey.replace('total_','').replace('_prevalence','') # Name of the actual state
                    if attr[0] == 'n': attr = attr[2:]
                    if attr == 'hpv': attr = 'infectious' # People with HPV are referred to as infectious in the sim
                    if 'total' in rkey:
                        inds = sim.people[attr].any(axis=0).nonzero()  # Pull out people for which this state is true
                        self.results[date][rkey] = np.histogram(age[inds[-1]], bins=self.edges)[0] * scale  # Bin the people
                    else:
                        for g in range(ng):
                            inds = sim.people[attr][g,:].nonzero()
                            self.results[date][rkey][g,:] = np.histogram(age[inds[-1]], bins=self.edges)[0] * scale  # Bin the people

                    if 'prevalence' in rkey:
                        # Need to divide by the right denominator
                        if 'hpv' in rkey: # Denominator is whole population
                            denom = (np.histogram(age, bins=self.edges)[0] * scale)
                        else: # Denominator is females
                            denom = (np.histogram(age[sim.people.f_inds], bins=self.edges)[0] * scale)
                        if 'total' not in rkey: denom = denom[None,:]
                        self.results[date][rkey] = self.results[date][rkey] / denom

            self.date = date # Need to store the date for subsequent calcpoints

        # Both annual new cases and incidence require us to calculate the new cases over all
        # the timepoints that belong to the requested year.
        if sim.t in self.calcpoints:
            date = self.date # Stored just above for use here
            scale = sim.rescale_vec[sim.t//sim.resfreq] # Determine current scale factor
            age = sim.people.age # Get the age distribution
            age_standard = self.age_standard[1, :-1]

            for rkey in self.result_keys: # Loop over each result

                # Figure out if it's a flow or incidence
                if rkey.replace('total_', '') in hpd.flow_keys or 'incidence' in rkey:
                    attr = rkey.replace('total_','').replace('_incidence','') # Name of the actual state
                    if attr == 'hpv': attr = 'infections' # HPV is referred to as infections in the sim
                    if attr == 'cancer': attr = 'cancers' # cancer is referred to as cancers in the sim
                    attr1 = self.mapping[attr][0] # Messy way of turning 'total cancers' into 'date_cancerous' and 'cancerous' etc
                    attr2 = self.mapping[attr][1] # As above
                    if rkey[:5] == 'total': # Results across all genotypes
                        inds = ((sim.people[attr1]==sim.t)*(sim.people[attr2])).nonzero()
                        self.results[date][rkey] += np.histogram(age[inds[-1]], bins=self.edges)[0] * scale * age_standard  # Bin the people
                    else: # Results by genotype
                        for g in range(ng): # Loop over genotypes
                            inds = ((sim.people[attr1][g,:] == sim.t) * (sim.people[attr2][g,:])).nonzero()
                            self.results[date][rkey][g,:] += np.histogram(age[inds[-1]], bins=self.edges)[0] * scale * age_standard  # Bin the people

                    if 'incidence' in rkey:
                        # Need to divide by the right denominator
                        if 'hpv' in rkey: # Denominator is susceptible population
                            denom = (np.histogram(age[sim.people.sus_pool[-1]], bins=self.edges)[0] * scale)
                        else:  # Denominator is females
                            denom = (np.histogram(age[sim.people.f_inds], bins=self.edges)[0] * scale * 1/100000)  # scale to be per 100,000 for cancer and cin incidence
                        if 'total' not in rkey: denom = denom[None,:]
                        self.results[date][rkey] = self.results[date][rkey] / denom


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.results.keys(), die=self.die)
        return


    def plot(self, fig_args=None, axis_args=None, data_args=None, width=0.8,
             do_save=None, fig_path=None, do_show=True, **kwargs):
        '''
        Plot the age results

        Args:
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            width (float): width of the bars
            do_save (bool): whether to save
            fig_path (str or filepath): filepath to save to
            do_show (bool): whether to show the figure
            kwargs (dict): passed to ``hp.options.with_style()``; see that function for choices
        '''

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))
        all_args = sc.mergedicts(fig_args, axis_args, d_args)

        # Initialize
        fig = pl.figure(**fig_args)

        # Handle what to plot
        if not len(self.results):
            errormsg = f'Cannot plot since no age results were recorded (scheduled timepoints: {self.timepoints})'
            raise ValueError(errormsg)
        if len(self.timepoints)>1:
            n_cols = len(self.timepoints) # One column for each requested timepoint
            n_rows = len(self.result_keys) # One row for each requested result
        else: # If there's only one timepoint, automatically figure out rows and columns
            n_plots = len(self.result_keys)
            n_rows, n_cols = sc.get_rows_cols(n_plots)

        # Make the figure(s)
        with hpo.with_style(**kwargs):
            col_count=1
            for date,resdict in self.results.items():

                pl.subplots_adjust(**axis_args)
                row_count = 0

                for rkey in self.result_keys:

                    # Start making plot
                    barwidth = 1 * width
                    ax = pl.subplot(n_rows, n_cols, col_count+row_count)
                    x = np.arange(len(self.age_labels))  # the label locations
                    if self.data is not None:
                        thisdatadf = self.data[(self.data.year == float(date))&(self.data.name == rkey)]
                        if len(thisdatadf)>0:
                            barwidth /= 2 # Adjust width based on data

                    if 'total' not in rkey:
                        # Prepare plot settings
                        barwidth /= self.ng  # Adjust width based on number of genotypes (warning, this will be crowded)
                        if (self.ng % 2) == 0:  # Incredibly complex way of automatically generating bar offsets
                            xlocations = np.array([-(g + 1) for g in reversed(range(self.ng // 2))] + [(g + 1) for g in range(self.ng // 2)]) * .5 * barwidth
                        else:
                            xlocations = np.array([-2 * (g + 1) for g in reversed(range(self.ng // 2))] + [0] + [2 * (g + 1) for g in range(self.ng // 2)]) * .5 * barwidth

                        for g in range(self.ng):
                            glabel = self.glabels[g].upper()
                            ax.bar(x+xlocations[g]-barwidth, resdict[rkey][g,:], color=self.result_properties[rkey].color[g], label=f'Model - {glabel}', width=barwidth)
                            if len(thisdatadf)>0:
                                ydata = np.array(thisdatadf[thisdatadf.genotype==self.glabels[g].upper()].value)
                                ax.bar(x+xlocations[g]+barwidth, ydata, color=self.result_properties[rkey].color[g], hatch='/', label=f'Data - {glabel}', width=barwidth)

                    else:
                        if (self.data is not None) and (len(thisdatadf) > 0):
                            ax.bar(x-1/2*barwidth, resdict[rkey], color=self.result_properties[rkey].color, width=barwidth, label='Model')
                            ydata = np.array(thisdatadf.value)
                            ax.bar(x+1/2*barwidth, ydata, color=d_args.color, width=barwidth, label='Data')
                        else:
                            ax.bar(x, resdict[rkey], color=self.result_properties[rkey].color, width=barwidth, label='Model')
                    ax.set_xlabel('Age group')
                    # ax.set_ylabel('Frequency')
                    ax.set_title(self.result_properties[rkey].name+' - '+date)
                    ax.legend()
                    row_count += n_cols
                    ax.set_xticks(x, self.age_labels)

                col_count+=1


        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)

