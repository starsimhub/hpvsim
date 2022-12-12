'''
Additional analysis functions that are not part of the core workflow,
but which are useful for particular investigations.
'''

import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from . import utils as hpu
from . import misc as hpm
from . import plotting as hppl
from . import defaults as hpd
from . import parameters as hppar
from . import interventions as hpi
from .settings import options as hpo # For setting global options


__all__ = ['Analyzer', 'snapshot', 'age_pyramid', 'age_results', 'age_causal_infection',
           'cancer_detection', 'analyzer_map']


class Analyzer(sc.prettyobj):
    '''
    Base class for analyzers. Based on the Intervention class. Analyzers are used
    to provide more detailed information about a simulation than is available by
    default -- for example, pulling states out of sim.people on a particular timestep
    before it gets updated in the next timestep.

    To retrieve a particular analyzer from a sim, use ``sim.get_analyzer()``.

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

        This method is run once as part of ``sim.finalize()`` enabling the analyzer to perform any
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
        Remove any excess stored data from the intervention; for use with ``sim.shrink()``.

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place:
            return self
        else:
            return sc.dcp(self)

    @staticmethod
    def reduce(analyzers, use_mean=False):
        '''
        Create a reduced analyzer from a list of analyzers, using
        
        Args:
            analyzers: list of analyzers
            use_mean (bool): whether to use medians (the default) or means to create the reduced analyzer
        '''
        pass


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
    the dictionary directly, or use the ``get()`` method.

    Args:
        timepoints  (list): list of ints/strings/date objects, the days on which to take the snapshot
        die         (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs      (dict): passed to :py:class:`Analyzer`


    **Example**::

        sim = hpv.Sim(analyzers=hpv.snapshot('2015.4', '2020'))
        sim.run()
        snapshot = sim['analyzers'][0]
        people = snapshot.snapshots[0]            # Option 1
        people = snapshot.snapshots['2020']       # Option 2
        people = snapshot.get('2020')             # Option 3
        people = snapshot.get(34)                 # Option 4
        people = snapshot.get()                   # Option 5
    '''

    def __init__(self, timepoints=None, *args, die=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        self.timepoints     = timepoints
        self.die            = die  # Whether or not to raise an exception
        self.dates          = None # Representations in terms of years, e.g. 2020.4, set during initialization
        self.start          = None # Store the start year of the simulation
        self.snapshots      = sc.odict() # Store the actual snapshots
        return


    def initialize(self, sim):
        self.start = sim['start'] # Store the simulation start
        if self.timepoints is None:
            self.timepoints = [sim['end']]
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

        sim = hpv.Sim(analyzers=hpv.age_pyramid('2015', '2020'))
        sim.run()
        age_pyramid = sim['analyzers'][0]
    '''

    def __init__(self, timepoints=None, *args, edges=None, age_labels=None, datafile=None, die=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
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
            self.timepoints = [self.end] # If no day is supplied, use the last day
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
            ppl = sim.people
            self.age_pyramids[date]['bins'] = self.bins # Copy here for convenience
            for sb,sex in enumerate(['m','f']): # Loop over each sex; sb stands for sex boolean, translating the labels to 0/1
                inds = (sim.people.alive*(ppl.sex==sb)).nonzero()[0]
                self.age_pyramids[date][sex] = np.histogram(ppl.age[inds], bins=self.edges, weights=ppl.scale[inds])[0]  # Bin people


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.age_pyramids.keys(), die=self.die)
        return


    @staticmethod
    def reduce(analyzers, use_mean=False, bounds=None, quantiles=None):
        ''' Create an averaged age pyramid from a list of age pyramid analyzers '''

        # Process inputs for statistical calculations
        if use_mean:
            if bounds is None:
                bounds = 2
        else:
            if quantiles is None:
                quantiles = {'low':0.1, 'high':0.9}
            if not isinstance(quantiles, dict):
                try:
                    quantiles = {'low':float(quantiles[0]), 'high':float(quantiles[1])}
                except Exception as E:
                    errormsg = f'Could not figure out how to convert {quantiles} into a quantiles object: must be a dict with keys low, high or a 2-element array ({str(E)})'
                    raise ValueError(errormsg)

        # Check that a list of analyzers has been provided
        if not isinstance(analyzers, list):
            errormsg = 'age_pyramid.reduce() expects a list of age pyramid analyzers'
            raise TypeError(errormsg)

        # Check that everything in the list is an analyzer of the right type
        for analyzer in analyzers:
            if not isinstance(analyzer, age_pyramid):
                errormsg = 'All items in the list of analyzers provided to age_pyramid.reduce must be age pyramids'
                raise TypeError(errormsg)

        # Check that all the analyzers have the same timepoints and age bins
        base_analyzer = analyzers[0]
        for analyzer in analyzers:
            if not np.array_equal(analyzer.timepoints, base_analyzer.timepoints):
                errormsg = 'The list of analyzers provided to age_pyramid.reduce have different timepoints.'
                raise TypeError(errormsg)
            if not np.array_equal(analyzer.edges, base_analyzer.edges):
                errormsg = 'The list of analyzers provided to age_pyramid.reduce have different age bin edges.'
                raise TypeError(errormsg)

        # Initialize the reduced analyzer
        reduced_analyzer = sc.dcp(base_analyzer)
        reduced_analyzer.age_pyramids = sc.objdict() # Remove the age pyramids so we can rebuild them

        # Aggregate the list of analyzers
        raw = {}
        for date,tp in zip(base_analyzer.dates, base_analyzer.timepoints):
            raw[date] = {}
            # raw[date]['bins'] = analyzer.age_pyramids[date]['bins']
            reduced_analyzer.age_pyramids[date] = sc.objdict()
            reduced_analyzer.age_pyramids[date]['bins'] = analyzer.age_pyramids[date]['bins']
            for sk in ['f','m']:
                raw[date][sk] = np.zeros((len(base_analyzer.age_pyramids[date]['bins']), len(analyzers)))
                for a, analyzer in enumerate(analyzers):
                    vals = analyzer.age_pyramids[date][sk]
                    raw[date][sk][:, a] = vals

                # Summarizing the aggregated list
                reduced_analyzer.age_pyramids[date][sk] = sc.objdict()
                if use_mean:
                    r_mean = np.mean(raw[date][sk], axis=1)
                    r_std = np.std(raw[date][sk], axis=1)
                    reduced_analyzer.age_pyramids[date][sk].best    = r_mean
                    reduced_analyzer.age_pyramids[date][sk].low     = r_mean - bounds * r_std
                    reduced_analyzer.age_pyramids[date][sk].high    = r_mean + bounds * r_std
                else:
                    reduced_analyzer.age_pyramids[date][sk].best    = np.quantile(raw[date][sk], q=0.5, axis=1)
                    reduced_analyzer.age_pyramids[date][sk].low     = np.quantile(raw[date][sk], q=quantiles['low'], axis=1)
                    reduced_analyzer.age_pyramids[date][sk].high    = np.quantile(raw[date][sk], q=quantiles['high'], axis=1)

        return reduced_analyzer


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
            kwargs (dict): passed to ``hpv.options.with_style()``; see that function for choices
        '''
        import seaborn as sns # Import here since slow

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
                pl.xticks(xticks, xlabels)
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
                        pl.xticks(xticks, xlabels)
                        ax.set_title(f'{date} - data')

                    count += 1

        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)



class age_results(Analyzer):
    '''
    Constructs results by age at specified points within the sim. Can be used with data

    Args:
        result_args (dict): dict of results to generate and associated timepoints/age-bins to generate each result as well as whether to compute_fit
        result_keys (list): list of results to generate - used to construct result_args is result_args is not provided
        timepoints  (list): list of timepoints - used to construct result_args is result_args is not provided
        edges       (arr): list of edges of age bins - used to construct result_args is result_args is not provided
        die         (bool): whether or not to raise an exception if errors are found
        kwargs      (dict): passed to :py:class:`Analyzer`

    **Example**::
    # Construct your own result_args if you want different timepoints / age buckets for each one
        result_args=sc.objdict(
            hpv_prevalence=sc.objdict(
                timepoints=[1990],
                edges=np.array([0.,20.,25.,30.,40.,45.,50.,55.,65.,100.]),
            ),
            hpv_incidence=sc.objdict(
                timepoints=[1990, 2000],
                edges=np.array([0.,20.,30.,40.,50.,60.,70.,80.,100.])
            )
        sim = hpv.Sim(analyzers=hpv.age_results(result_args=result_args))
        sim.run()
        age_results = sim['analyzers'][0]

    # Alternatively, use standard timepoints and age buckets across all results
        sim = hpv.Sim(analyzers=hpv.age_results(result_keys=['cancers']))
    '''

    def __init__(self, result_keys=None, die=False, edges=None, timepoints=None, result_args=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        self.mismatch       = 0
        self.die            = die  # Whether or not to raise an exception
        self.start          = None # Store the start year of the simulation
        self.edges          = edges or np.array([0., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 100.])
        self.timepoints     = timepoints
        self.result_keys    = result_keys or ['infections', 'cancers']
        self.results        = sc.odict() # Store the age results
        self.result_args    = result_args
        return


    def initialize(self, sim):

        super().initialize()

        # Handle timepoints and dates
        self.start = sim['start']  # Store the simulation start
        self.end = sim['end']  # Store simulation end
        if self.timepoints is None:
            self.timepoints = [f'{sim["end"]}']
            self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format

        # Handle which results to make. Specification of the results to make is stored in result_args
        if self.result_args is None: # Make defaults if none are provided
            self.result_args = sc.objdict()
            for rkey in self.result_keys:
                self.result_args[rkey] = sc.objdict(timepoints=self.dates, edges=self.edges)
        elif sc.checktype(self.result_args, dict): # Ensure it's an object dict
            self.result_args = sc.objdict(self.result_args)
        else: # Raise an error
            errormsg = f'result_args must be a dict with keys for the timepoints and edges you want to compute, not {type(result_args)}.'
            raise TypeError(errormsg)

        # Handle dt - if we're storing annual results we'll need to aggregate them over several consecutive timesteps
        self.dt = sim['dt']
        self.resfreq = sim.resfreq

        self.validate_results(sim)

        # Store genotypes
        self.ng = sim['n_genotypes']
        self.glabels = [g.upper() for g in sim['genotype_map'].values()]

        # Store colors
        for rkey in self.result_args.keys():
            self.result_args[rkey].color = sim.results[rkey].color
            self.result_args[rkey].name = sim.results[rkey].name

        self.initialized = True

        return


    def validate_results(self, sim):
        choices = sim.result_keys('total')+[k for k in sim.result_keys('genotype')]
        for rk, rdict in self.result_args.items():
            if rk not in choices:
                strm = '\n'.join(choices)
                errormsg = f'Cannot compute age results for {rk}. Please enter one of the standard sim result_keys to the age_results analyzer; choices are {strm}.'
                raise ValueError(errormsg)
            else:
                self.results[rk] = sc.objdict()

            # Handle the data file
            # If data is provided, extract timepoints, edges, age bins and labels from that
            if 'datafile' in rdict.keys():
                if sc.isstring(rdict.datafile):
                    rdict.data = hpm.load_data(rdict.datafile, check_date=False)
                else:
                    rdict.data = rdict.datafile  # Use it directly
                    rdict.datafile = None

                # extract edges, age bins and labels from that
                # Handle edges, age bins, and labels
                rdict.timepoints = rdict.data.year.unique()
                rdict.age_labels = []
                rdict.edges = np.array(rdict.data.age.unique(), dtype=float)
                rdict.edges = np.append(rdict.edges, 100)
                rdict.bins = rdict.edges[:-1]  # Don't include the last edge in the bins
                self.results[rk]['bins'] = rdict.bins
                rdict.age_labels = [f'{int(rdict.bins[i])}-{int(rdict.bins[i + 1])}' for i in
                                        range(len(rdict.bins) - 1)]
                rdict.age_labels.append(f'{int(rdict.bins[-1])}+')

            else:
                # Handle edges, age bins, and labels
                rdict.age_labels = []
                if (not hasattr(rdict,'edges')) or rdict.edges is None:  # Default age bins
                    rdict.edges = np.linspace(0, 100, 11)
                rdict.bins = rdict.edges[:-1]  # Don't include the last edge in the bins
                self.results[rk]['bins'] = rdict.bins
                rdict.age_labels = [f'{int(rdict.bins[i])}-{int(rdict.bins[i + 1])}' for i in
                                    range(len(rdict.bins) - 1)]
                rdict.age_labels.append(f'{int(rdict.bins[-1])}+')
                if 'timepoints' not in rdict.keys():
                    errormsg = 'Did not provide timepoints for this age analyzer'
                    raise ValueError(errormsg)

            if not rdict.get('dates') or rdict.dates is None:
                rdict.timepoints, rdict.dates = sim.get_t(rdict.timepoints, return_date_format='str')  # Ensure timepoints and dates are in the right format
            max_hist_time = rdict.timepoints[-1]
            max_sim_time = sim['end']
            if max_hist_time > max_sim_time:
                errormsg = f'Cannot create age results for {rdict.dates[-1]} ({max_hist_time}) because the simulation ends on {self.end} ({max_sim_time})'
                raise ValueError(errormsg)

            rdict.calcpoints = []
            for tpi, tp in enumerate(rdict.timepoints):
                rdict.calcpoints += [tp + i for i in range(int(1 / self.dt))]

            if 'compute_fit' in rdict.keys() and rdict.compute_fit:
                if rdict.data is None:
                    errormsg = 'Cannot compute fit without data'
                    raise ValueError(errormsg)
                else:
                    if 'weights' in rdict.data.columns:
                        rdict.weights = rdict.data['weights'].values
                    else:
                        rdict.weights = np.ones(len(rdict.data))
                    rdict.mismatch = 0  # The final value


    def convert_rname_stocks(self, rname):
        ''' Helper function for converting stock result names to people attributes '''
        attr = rname.replace('_prevalence', '')  # Strip out terms that aren't stored in the people
        if attr[0] == 'n': attr = attr[2:] # Remove n, used to identify stocks
        if attr == 'hpv': attr = 'infectious'  # People with HPV are referred to as infectious in the sim
        if attr == 'cancer': attr = 'cancerous'
        return attr

    def convert_rname_flows(self, rname):
        ''' Helper function for converting flow result names to people attributes '''
        attr = rname.replace('_incidence', '')  # Name of the actual state
        if attr == 'hpv': attr = 'infections'  # HPV is referred to as infections in the sim
        if attr == 'cancer': attr = 'cancers'  # cancer is referred to as cancers in the sim
        if attr == 'cancer_mortality': attr = 'cancer_deaths'
        # Handle variable names
        mapping = {
            'infections': ['date_infectious', 'infectious'],
            'cin':  ['date_cin1', 'cin'], # Not a typo - the date the get a CIN is the same as the date they get a CIN1
            'cins':  ['date_cin1', 'cin'], # Not a typo - the date the get a CIN is the same as the date they get a CIN1
            'cin1': ['date_cin1', 'cin1'],
            'cin2': ['date_cin2', 'cin2'],
            'cin3': ['date_cin3', 'cin3'],
            'cancers': ['date_cancerous', 'cancerous'],
            'cancer': ['date_cancerous', 'cancerous'],
            'detected_cancer': ['date_detected_cancer', 'detected_cancer'],
            'detected_cancers': ['date_detected_cancer', 'detected_cancer'],
            'cancer_deaths': ['date_dead_cancer', 'dead_cancer'],
            'detected_cancer_deaths': ['date_dead_cancer', 'dead_cancer']
        }
        attr1 = mapping[attr][0]  # Messy way of turning 'total cancers' into 'date_cancerous' and 'cancerous' etc
        attr2 = mapping[attr][1]  # As above
        return attr1, attr2


    def apply(self, sim):
        ''' Calculate age results '''

        # Shorten variables that are used a lot
        ng = self.ng
        ppl = sim.people
        
        def bin_ages(inds=None, bins=None):
            return np.histogram(ppl.age[inds], bins=bins, weights=ppl.scale[inds])[0] # Bin the people

        # Go through each result key and determine if this is a timepoint where age results are requested
        for result, result_dict in self.result_args.items():

            # Establish initial quantities
            bins = result_dict.edges
            na = len(result_dict.bins)
            if 'genotype' in result: # Results by genotype
                result_name = result[9:]
                size = (na, ng)
                by_genotype = True
            else: # Total results
                result_name = result[:]
                size = na
                by_genotype = False

            # This section is completed for stocks
            if sim.t in result_dict.timepoints:

                ind = sc.findinds(result_dict.timepoints, sim.t)[0]  # Get the index
                date = result_dict.dates[ind]  # Create the date which will be used to key the results
                self.results[result][date] = np.zeros(size)

                if 'compute_fit' in result_dict.keys():
                    thisdatadf = result_dict.data[(result_dict.data.year == float(date)) & (result_dict.data.name == result)]
                    unique_genotypes = thisdatadf.genotype.unique()
                    ng = len(unique_genotypes)

                # Both annual stocks and prevalence require us to calculate the current stocks.
                # Unlike incidence, these don't have to be aggregated over multiple timepoints.
                if result_name[0] == 'n' or 'prevalence' in result_name:
                    attr = self.convert_rname_stocks(result_name) # Convert to a people attribute
                    if attr in ppl.keys():
                        if not by_genotype:
                            inds = ppl[attr].any(axis=0).nonzero()[-1]  # Pull out people for which this state is true
                            self.results[result][date] = bin_ages(inds, bins)
                        else:
                            for g in range(ng):
                                inds = ppl[attr][g, :].nonzero()[-1]
                                self.results[result][date][g, :] = bin_ages(inds, bins)  # Bin the people

                        if 'prevalence' in result:
                            # Need to divide by the right denominator
                            if 'hpv' in result:  # Denominator is whole population
                                denom = bin_ages(inds=None, bins=bins)
                            else:  # Denominator is females
                                denom = bin_ages(inds=ppl.f_inds, bins=bins)
                            if by_genotype: denom = denom[None, :]
                            self.results[result][date] = self.results[result][date] / denom

                self.date = date # Need to store the date for subsequent calcpoints
                self.timepoint = sim.t # Need to store the timepoints for subsequent calcpoints

            # Both annual new cases and incidence require us to calculate the new cases over all
            # the timepoints that belong to the requested year.
            if sim.t in result_dict.calcpoints:

                # Figure out if it's a flow or incidence
                if result_name in hpd.flow_keys or 'incidence' in result_name or 'mortality' in result_name:

                    date = self.date  # Stored just above for use here
                    attr1, attr2 = self.convert_rname_flows(result_name)
                    if not by_genotype:  # Results across all genotypes
                        if result_name == 'detected_cancer_deaths':
                            inds = ((ppl[attr1] == sim.t) * (ppl[attr2]) * (ppl['detected_cancer'])).nonzero()[-1]
                        else:
                            inds = ((ppl[attr1] == sim.t) * (ppl[attr2])).nonzero()[-1]
                        self.results[result][date] += bin_ages(inds, bins)  # Bin the people
                    else:  # Results by genotype
                        for g in range(ng):  # Loop over genotypes
                            inds = ((ppl[attr1][g, :] == sim.t) * (ppl[attr2][g, :])).nonzero()[-1]
                            self.results[result][date][g, :] += bin_ages(inds, bins)  # Bin the people

                    # Figure out if this is the last timepoint in the year we're calculating results for
                    if sim.t == self.timepoint+self.resfreq-1:
                        if 'incidence' in result:
                            # Need to divide by the right denominator
                            if 'hpv' in result:  # Denominator is susceptible population
                                denom = bin_ages(inds=hpu.true(ppl.sus_pool), bins=bins)
                            else:  # Denominator is females at risk for cancer
                                inds = sc.findinds(ppl.is_female_alive & ~ppl.cancerous.any(axis=0))
                                denom = bin_ages(inds, bins) / 1e5  # CIN and cancer are per 100,000 women
                            if 'total' not in result and 'cancer' not in result: denom = denom[None, :]
                            self.results[result][date] = self.results[result][date] / denom

                        if 'mortality' in result:
                            # Need to divide by the right denominator
                            # first need to find people who died of other causes today and add them back into denom
                            denom = bin_ages(inds=ppl.is_female_alive, bins=bins)
                            scale_factor =  1e5  # per 100,000 women
                            denom /= scale_factor
                            self.results[result][date] = self.results[result][date] / denom


    def finalize(self, sim):
        super().finalize()
        for rkey, rdict in self.result_args.items():
            validate_recorded_dates(sim, requested_dates=rdict.dates, recorded_dates=self.results[rkey].keys()[1:], die=self.die)
            if 'compute_fit' in rdict.keys():
                self.mismatch += self.compute(rkey)

        sim.fit = self.mismatch

        return


    @staticmethod
    def reduce(analyzers, use_mean=False, bounds=None, quantiles=None):
        ''' Create an averaged age result from a list of age result analyzers '''

        # Process inputs for statistical calculations
        if use_mean:
            if bounds is None:
                bounds = 2
        else:
            if quantiles is None:
                quantiles = {'low':0.1, 'high':0.9}
            if not isinstance(quantiles, dict):
                try:
                    quantiles = {'low':float(quantiles[0]), 'high':float(quantiles[1])}
                except Exception as E:
                    errormsg = f'Could not figure out how to convert {quantiles} into a quantiles object: must be a dict with keys low, high or a 2-element array ({str(E)})'
                    raise ValueError(errormsg)

        # Check that a list of analyzers has been provided
        if not isinstance(analyzers, list):
            errormsg = 'age_results.reduce() expects a list of age_results analyzers'
            raise TypeError(errormsg)

        # Check that everything in the list is an analyzer of the right type
        for analyzer in analyzers:
            if not isinstance(analyzer, age_results):
                errormsg = 'All items in the list of analyzers provided to age_results.reduce must be age_results instances'
                raise TypeError(errormsg)

        # Check that all the analyzers have the same timepoints and age bins
        base_analyzer = analyzers[0]
        for analyzer in analyzers:
            if set(analyzer.results.keys()) != set(base_analyzer.results.keys()):
                errormsg = 'The list of analyzers provided to age_results.reduce have different result keys.'
                raise ValueError(errormsg)
            for reskey in base_analyzer.results.keys():
                if not np.array_equal(base_analyzer.result_args[reskey]['timepoints'],analyzer.result_args[reskey]['timepoints']):
                    errormsg = 'The list of analyzers provided to age_results.reduce have different timepoints.'
                    raise ValueError(errormsg)
                if not np.array_equal(base_analyzer.result_args[reskey]['edges'],analyzer.result_args[reskey]['edges']):
                    errormsg = 'The list of analyzers provided to age_pyramid.reduce have different age bin edges.'
                    raise ValueError(errormsg)

        # Initialize the reduced analyzer
        reduced_analyzer = sc.dcp(base_analyzer)
        reduced_analyzer.results = sc.objdict() # Remove the age results so we can rebuild them

        # Aggregate the list of analyzers
        raw = {}
        for reskey in base_analyzer.results.keys():
            raw[reskey] = {}
            reduced_analyzer.results[reskey] = sc.objdict()
            reduced_analyzer.results[reskey]['bins'] = base_analyzer.results[reskey]['bins']
            for date,tp in zip(base_analyzer.result_args[reskey].dates, base_analyzer.result_args[reskey].timepoints):
                ashape = analyzer.results[reskey][date].shape # Figure out dimensions
                new_ashape = ashape + (len(analyzers),)
                raw[reskey][date] = np.zeros(new_ashape)
                for a, analyzer in enumerate(analyzers):
                    vals = analyzer.results[reskey][date]
                    if len(ashape) == 1:
                        raw[reskey][date][:, a] = vals
                    elif len(ashape) == 2:
                        raw[reskey][date][:, :, a] = vals

                # Summarizing the aggregated list
                reduced_analyzer.results[reskey][date] = sc.objdict()
                if use_mean:
                    r_mean = np.mean(raw[reskey][date], axis=-1)
                    r_std = np.std(raw[reskey][date], axis=-1)
                    reduced_analyzer.results[reskey][date].best = r_mean
                    reduced_analyzer.results[reskey][date].low  = r_mean - bounds * r_std
                    reduced_analyzer.results[reskey][date].high = r_mean + bounds * r_std
                else:
                    reduced_analyzer.results[reskey][date].best = np.quantile(raw[reskey][date], q=0.5, axis=-1)
                    reduced_analyzer.results[reskey][date].low  = np.quantile(raw[reskey][date], q=quantiles['low'], axis=-1)
                    reduced_analyzer.results[reskey][date].high = np.quantile(raw[reskey][date], q=quantiles['high'], axis=-1)

        return reduced_analyzer


    def compute(self, key):
        res = []
        resargs = self.result_args[key]
        results = self.results[key]
        for name, group in resargs.data.groupby(['genotype', 'year']):
            genotype = name[0]
            year = str(name[1]) + '.0'
            if 'genotype' in key:
                sim_res = list(results[year][self.glabels.index(genotype)])
                res.extend(sim_res)
            else:
                sim_res = list(results[year])
                res.extend(sim_res)

        self.result_args[key].data['model_output'] = res
        self.result_args[key].data['diffs'] = resargs.data['model_output'] - resargs.data['value']
        self.result_args[key].data['gofs'] = hpm.compute_gof(resargs.data['value'].values, resargs.data['model_output'].values)
        self.result_args[key].data['losses'] = resargs.data['gofs'].values * resargs.weights
        self.result_args[key].mismatch = resargs.data['losses'].sum()

        return self.result_args[key].mismatch

    def get_to_plot(self):
        ''' Get number of plots to make '''

        if len(self.results) == 0:
            errormsg = 'Cannot plot since no age results were recorded)'
            raise ValueError(errormsg)
        else:
            dates_per_result = [len(rk['dates']) for rk in self.result_args.values()]
            n_plots = sum(dates_per_result)
            to_plot_args = []
            for rkey in self.result_keys:
                for date in self.result_args[rkey]['dates']:
                    to_plot_args.append([rkey,date])
        return n_plots, to_plot_args


    def plot_single(self, ax, rkey, date, by_genotype, plot_args=None, scatter_args=None):
        '''
        Function to plot a single age result for a single date. Requires an axis as
        input and will generally be called by a helper function rather than directly.
        '''
        args = sc.objdict()
        args.plot       = sc.objdict(sc.mergedicts(dict(linestyle='--'), plot_args))
        args.scatter    = sc.objdict(sc.mergedicts(dict(marker='s'), scatter_args))

        resdict = self.results[rkey] # Extract the result dictionary...
        resargs = self.result_args[rkey] # ... and the result arguments

        x = np.arange(len(resargs.age_labels)) # Create the label locations

        # Pull out data dataframe, if available
        if 'data' in resargs.keys():
            thisdatadf = resargs.data[(resargs.data.year == float(date)) & (resargs.data.name == rkey)]
            unique_genotypes = thisdatadf.genotype.unique()

        # Plot by genotype
        if by_genotype:
            colors = sc.gridcolors(self.ng) # Overwrite default colors with genotype colors
            for g in range(self.ng):
                color = colors[g]
                glabel = self.glabels[g].upper()
                ax.plot(x, resdict[date][g,:], color=color, **args.plot, label=glabel)

                if ('data' in resargs.keys()) and (len(thisdatadf) > 0):
                    # check if this genotype is in dataframe
                    if self.glabels[g].upper() in unique_genotypes:
                        ydata = np.array(thisdatadf[thisdatadf.genotype==self.glabels[g].upper()].value)
                        ax.scatter(x, ydata, color=color, **args.scatter, label=f'Data - {glabel}')

        # Plot totals
        else:
            ax.plot(x, resdict[date], color=resargs.color, **args.plot, label='Model')
            if ('data' in resargs.keys()) and (len(thisdatadf) > 0):
                ydata = np.array(thisdatadf.value)
                ax.scatter(x, ydata,  color=resargs.color, **args.scatter, label='Data')

        # Labels and legends
        ax.set_xlabel('Age')
        ax.set_title(resargs.name+' - '+date.split('.')[0])
        ax.legend()
        pl.xticks(x, resargs.age_labels)

        return ax



    def plot(self, fig_args=None, axis_args=None, plot_args=None, scatter_args=None,
             do_save=None, fig_path=None, do_show=True, fig=None, ax=None, **kwargs):
        '''
        Plot the age results

        Args:
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            plot_args (dict): passed to plot_single
            scatter_args (dict): passed to plot_single
            do_save (bool): whether to save
            fig_path (str or filepath): filepath to save to
            do_show (bool): whether to show the figure
            kwargs (dict): passed to ``hpv.options.with_style()``; see that function for choices
        '''

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        all_args = sc.mergedicts(fig_args, axis_args)

        # Initialize
        fig = pl.figure(**fig_args)
        n_plots, _ = self.get_to_plot()
        n_rows, n_cols = sc.get_rows_cols(n_plots)

        # Make the figure(s)
        with hpo.with_style(**kwargs):
            plot_count=1
            for rkey,resdict in self.results.items():
                pl.subplots_adjust(**axis_args)
                by_genotype=True if 'genotype' in rkey else False
                for date in self.result_args[rkey]['dates']:
                    ax = pl.subplot(n_rows, n_cols, plot_count)
                    ax = self.plot_single(ax, rkey, date, by_genotype, plot_args=plot_args, scatter_args=scatter_args)
                    plot_count+=1

        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)


class age_causal_infection(Analyzer):
    '''
    Determine the age at which people with cervical cancer were causally infected and
    time spent between infection and cancer.
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.age_causal = []
        self.age_cancer = []
        self.dwelltime = dict()
        for state in ['hpv', 'cin1', 'cin2', 'cin3', 'total']:
            self.dwelltime[state] = []

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                current_age = sim.people.age[cancer_inds]
                date_exposed = sim.people.date_exposed[cancer_genotypes, cancer_inds]
                date_cin1 = sim.people.date_cin1[cancer_genotypes, cancer_inds]
                date_cin2 = sim.people.date_cin2[cancer_genotypes, cancer_inds]
                date_cin3 = sim.people.date_cin3[cancer_genotypes, cancer_inds]
                hpv_time = (date_cin1 - date_exposed) * sim['dt']
                cin1_time = (date_cin2 - date_cin1) * sim['dt']
                cin2_time = (date_cin3 - date_cin2) * sim['dt']
                cin3_time = (sim.t - date_cin3) * sim['dt']
                total_time = (sim.t - date_exposed) * sim['dt']
                self.age_causal += (current_age - total_time).tolist()
                self.age_cancer += current_age.tolist()
                self.dwelltime['hpv'] += hpv_time.tolist()
                self.dwelltime['cin1'] += cin1_time.tolist()
                self.dwelltime['cin2'] += cin2_time.tolist()
                self.dwelltime['cin3'] += cin3_time.tolist()
                self.dwelltime['total'] += total_time.tolist()
        return

    def finalize(self, sim=None):
        ''' Convert things to arrays '''

class cancer_detection(Analyzer):
    '''
    Cancer detection via symptoms
    
    Args:
        symp_prob: Probability of having cancer detected via symptoms, rather than screening
        treat_prob: Probability of receiving treatment for those with symptom-detected cancer
    '''

    def __init__(self, symp_prob=0.01, treat_prob=0.01, product=None, **kwargs):
        super().__init__(**kwargs)
        self.symp_prob = symp_prob
        self.treat_prob = treat_prob
        self.product = product or hpi.radiation()

    def initialize(self, sim):
        super().initialize(sim)
        self.dt = sim['dt']


    def apply(self, sim):
        '''
        Check for new cancer detection, treat subset of detected cancers
        '''
        cancer_genotypes, cancer_inds = sim.people.cancerous.nonzero()  # Get everyone with cancer
        new_detections, new_treatments = 0, 0

        if len(cancer_inds) > 0:

            detection_probs = np.full(len(cancer_inds), self.symp_prob / self.dt, dtype=hpd.default_float)  # Initialize probabilities of cancer detection
            detection_probs[sim.people.detected_cancer[cancer_inds]] = 0
            is_detected = hpu.binomial_arr(detection_probs)
            is_detected_inds = cancer_inds[is_detected]
            new_detections = len(is_detected_inds)

            if new_detections>0:
                sim.people.detected_cancer[is_detected_inds] = True
                sim.people.date_detected_cancer[is_detected_inds] = sim.t
                treat_probs = np.full(len(is_detected_inds), self.treat_prob)
                treat_inds = is_detected_inds[hpu.binomial_arr(treat_probs)]
                if len(treat_inds)>0:
                    self.product.administer(sim, treat_inds)

        # Update flows
        sim.people.flows['detected_cancers'] = new_detections

        return new_detections, new_treatments


#%% Additional utilities
analyzer_map = {
    'snapshot': snapshot,
    'age_pyramid': age_pyramid,
    'age_results': age_results,
    'age_causal_infection': age_causal_infection,
    'cancer_detection': cancer_detection,
}

