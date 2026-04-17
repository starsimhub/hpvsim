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
           'dalys', 'analyzer_map']


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
        result_args (dict): dict of results to generate and associated years/age-bins to generate each result as well as whether to compute_fit
        die         (bool): whether or not to raise an exception if errors are found
        kwargs      (dict): passed to :py:class:`Analyzer`

    **Example**::

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

    '''

    def __init__(self, result_args=None, die=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        self.mismatch       = 0 # TODO, should this be set to np.nan initially?
        self.die            = die  # Whether or not to raise an exception
        self.results        = sc.objdict() # Store the age results
        self.result_args    = result_args
        return


    def initialize(self, sim):

        super().initialize()

        # Handle which results to make. Specification of the results to make is stored in result_args
        if sc.checktype(self.result_args, dict): # Ensure it's an object dict
            self.result_args = sc.objdict(self.result_args)
        else: # Raise an error
            errormsg = f'result_args must be a dict with keys for the years and edges you want to compute, not {type(self.result_args)}.'
            raise TypeError(errormsg)
        self.result_keys = self.result_args.keys()

        # Handle dt - if we're storing annual results we'll need to aggregate them over several consecutive timesteps
        self.dt = sim['dt']
        self.resfreq = sim.resfreq

        # Store genotypes
        self.ng = sim['n_genotypes']
        self.glabels = [g.upper() for g in sim['genotype_map'].values()]

        # Initialize result structure and validate the result variable arguments
        self.validate_variables(sim)

        # Store colors
        for rkey in self.result_args.keys():
            if 'hiv' in rkey:
                self.result_args[rkey].color = sim.hivsim.results[rkey].color
                self.result_args[rkey].name = sim.hivsim.results[rkey].name
            else:
                self.result_args[rkey].color = sim.results[rkey].color
                self.result_args[rkey].name = sim.results[rkey].name

        self.initialized = True

        return


    def validate_variables(self, sim):
        '''
        Check that the variables in result_args are valid, and initialize the result structure
        '''
        choices = sim.result_keys('total')+[k for k in sim.result_keys('genotype')]
        if sim['model_hiv']:
            choices += list(sim.hivsim.results.keys())

        for rk, rdict in self.result_args.items():
            if rk not in choices:
                strm = '\n'.join(choices)
                errormsg = f'Cannot compute age results for {rk}. Please enter one of the standard sim result_keys to the age_results analyzer; choices are {strm}.'
                raise ValueError(errormsg)
            else:
                self.results[rk] = dict() # Store the results. Not an odict because keyed by year

            # If a datafile has been provided, read it in and get the age bins and years
            if 'datafile' in rdict.keys():
                if sc.isstring(rdict.datafile):
                    rdict.data = hpm.load_data(rdict.datafile, check_date=False)
                else:
                    rdict.data = rdict.datafile  # Use it directly
                    rdict.datafile = None

                # Get edges, age bins, and labels from datafile. This assumes
                # that the datafile has bins, and we make the edges by appending
                # the last point of the sim age bin edges.
                rdict.years = rdict.data.year.unique()
                rdict.bins = np.array(rdict.data.age.unique(), dtype=float)
                rdict.edges = np.append(rdict.bins, sim['age_bin_edges'][-1])
                self.results[rk]['bins'] = rdict.bins

            else:

                # Use years and age bin edges provided, or use defaults from sim
                if (not hasattr(rdict,'edges')) or rdict.edges is None:  # Default age bins
                    warnmsg = f'Did not provide edges for age analyzer {rk}'
                    if self.die:
                        raise ValueError(warnmsg)
                    else:
                        warnmsg += ', using age bin edges from sim'
                        hpm.warn(warnmsg)
                        rdict.edges = sim['age_bin_edges']
                rdict.bins = rdict.edges[:-1]  # Don't include the last edge in the bins
                self.results[rk]['bins'] = rdict.bins

                if 'years' not in rdict.keys():
                    warnmsg = f'Did not provide years for age analyzer {rk}'
                    if self.die:
                        raise ValueError(warnmsg)
                    else:
                        warnmsg += ', using final year of sim'
                        hpm.warn(warnmsg)
                        rdict.years = sim['end']
                rdict.years = sc.promotetoarray(rdict.years)

            # Construct age labels used for plotting
            rdict.age_labels = [f'{int(rdict.bins[i])}-{int(rdict.bins[i + 1])}' for i in
                                range(len(rdict.bins) - 1)]
            rdict.age_labels.append(f'{int(rdict.bins[-1])}+')


            # Construct timepoints
            if not rdict.get('timepoints') or rdict.timepoints is None:
                rdict.timepoints = []
                for y in rdict.years:
                    rdict.timepoints.append(sc.findinds(sim.yearvec, y)[0] + int(1 / sim['dt']) - 1)

            # Check that the requested timepoints are in the sim
            max_hist_time = rdict.timepoints[-1]
            max_sim_time = sim.tvec[-1]
            if max_hist_time > max_sim_time:
                errormsg = f'Cannot create age results for {rdict.years[-1]} ({max_hist_time}) because the simulation ends on {self.end} ({max_sim_time})'
                raise ValueError(errormsg)

            # Translate the name of the result to the people attribute
            result_name = sc.dcp(rk)
            na = len(rdict.bins)
            ng = sim['n_genotypes']

            # Clean up the name
            if 'genotype' in result_name: # Results by genotype
                result_name = result_name.replace('_by_genotype','') # remove "by_genotype" from result name
                rdict.size = (na, ng)
                rdict.by_genotype = True
            else: # Total results
                rdict.size = na
                rdict.by_genotype = False
            rdict.by_hiv = False
            hiv_labels = ['_with_hiv', '_no_hiv']
            for hiv_label in hiv_labels:
                if hiv_label in result_name:
                    rdict.by_hiv = True
                    rdict.hiv_attr = 'with' if 'with' in hiv_label else 'no'
                    result_name = result_name.replace(hiv_label, '')  # remove "_with_hiv" or "_no_hiv" from result name
            # Figure out if it's a flow or incidence
            if result_name in hpd.flow_keys or 'incidence' in result_name or 'mortality' in result_name:
                date_attr, attr = self.convert_rname_flows(result_name)
                rdict.result_type = 'flow'
            elif result_name[:2] == 'n_' or 'prevalence' in result_name:
                attr = self.convert_rname_stocks(result_name)  # Convert to a people attribute
                date_attr = None
                rdict.result_type = 'stock'

            rdict.attr = attr
            rdict.date_attr = date_attr

            # Add special case for rate ratios
            if result_name == 'cancer_hiv_rate_ratios':
                rdict.by_hiv = True
                rdict.hiv_attr = 'ratio'
                result_name = 'cancers'
                rdict.result_type = 'stock'
                rdict.attr = 'cancerous'
                rdict.date_attr = 'date_cancerous'

            # Initialize results
            for year in rdict.years:
                self.results[rk][year] = np.zeros(rdict.size)

            # For flows, we calculate results on all the timepoints throughout the year, not just the last one
            if rdict.result_type == 'flow':
                rdict.calcpoints = []
                rdict.calcpointyears = []
                for tpi, tp in enumerate(rdict.timepoints):
                    rdict.calcpoints += [tp+i+1 for i in range(-int(1/self.dt),0)]
                    rdict.calcpointyears += [sim.yearvec[tp-(int(1/sim['dt'])-1)]]*int(1/self.dt)
            else:
                rdict.calcpoints = sc.dcp(rdict.timepoints)
                rdict.calcpointyears = [sim.yearvec[tp-(int(1/sim['dt'])-1)] for tp in rdict.calcpoints]

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
        if attr == 'cancer_by_age': attr = 'cancers'  # cancer is referred to as cancers in the sim
        if attr == 'cancer_mortality': attr = 'cancer_deaths'
        # Handle variable names
        mapping = {
            'infections': ['date_exposed', 'infectious'],
            'cin':  ['date_cin', 'cin'],
            'dysplasias':  ['date_cin', 'cin'],
            'cins':  ['date_cin', 'cin'],
            'cancers': ['date_cancerous', 'cancerous'],
            'cancer': ['date_cancerous', 'cancerous'],
            'cancer_deaths': ['date_dead_cancer', 'dead_cancer'],
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
        for rkey, rdict in self.result_args.items():

            # Establish initial quantities
            bins = rdict.edges
            na = len(rdict.bins)

            # Calculate flows and stocks over all calcpoints
            if sim.t in rdict.calcpoints:

                date_ind = sc.findinds(rdict.calcpoints, sim.t)[0]  # Get the index
                date = rdict.calcpointyears[date_ind]  # Create the date which will be used to key the results

                if 'compute_fit' in rdict.keys():
                    thisdatadf = rdict.data[(rdict.data.year == float(date)) & (rdict.data.name == rkey)]
                    unique_genotypes = thisdatadf.genotype.unique()
                    ng = len(unique_genotypes)  # CAREFUL, THIS IS OVERWRITING

                # Figure out if it's a flow
                if rdict.result_type == 'flow':
                    if not rdict.by_genotype:  # Results across all genotypes
                        if rdict.by_hiv:
                            if rdict.hiv_attr == 'with':
                                inds = ((ppl[rdict.date_attr] == sim.t) * (ppl[rdict.attr]) * (ppl['hiv'])).nonzero()[-1]
                            elif rdict.hiv_attr == 'no':
                                inds = ((ppl[rdict.date_attr] == sim.t) * (ppl[rdict.attr]) * (~ppl['hiv'])).nonzero()[-1]
                        else:
                            inds = ((ppl[rdict.date_attr] == sim.t) * (ppl[rdict.attr])).nonzero()[-1]
                        self.results[rkey][date] += bin_ages(inds, bins)  # Bin the people
                    else:  # Results by genotype
                        for g in range(ng):  # Loop over genotypes
                            inds = ((ppl[rdict.date_attr][g, :] == sim.t) * (ppl[rdict.attr][g, :])).nonzero()[-1]
                            self.results[rkey][date][:, g] += bin_ages(inds, bins)  # Bin the people

                # This section is completed for stocks
                elif rdict.result_type == 'stock':

                    if not rdict.by_genotype:
                        if rdict.by_hiv:
                            if rdict.hiv_attr == 'with':
                                inds = (ppl[rdict.attr].any(axis=0) * ppl['hiv']).nonzero()[-1]
                                self.results[rkey][date] = bin_ages(inds, bins)
                            elif rdict.hiv_attr == 'no':
                                inds = (ppl[rdict.attr].any(axis=0) * ~ppl['hiv']).nonzero()[-1]
                                self.results[rkey][date] = bin_ages(inds, bins)
                            elif rdict.hiv_attr == 'ratio':
                                hiv_inds = (ppl[rdict.attr].any(axis=0) * ppl['hiv']).nonzero()[-1]
                                no_hiv_inds = (ppl[rdict.attr].any(axis=0) * ~ppl['hiv']).nonzero()[-1]
                                self.results[rkey][date] = sc.safedivide(bin_ages(hiv_inds, bins), bin_ages(no_hiv_inds, bins))
                        elif isinstance(rdict.attr, list):
                            inds = (ppl[rdict.attr[0]].any(axis=0) + ppl[rdict.attr[1]].any(axis=0)).nonzero()[-1]
                            inds = np.unique(inds)
                            self.results[rkey][date] = bin_ages(inds, bins)
                        else:
                            inds = ppl[rdict.attr].any(axis=0).nonzero()[-1]
                            self.results[rkey][date] = bin_ages(inds, bins)
                    else:
                        for g in range(ng):
                            inds = ppl[rdict.attr][g, :].nonzero()[-1]
                            self.results[rkey][date][g, :] = bin_ages(inds, bins)  # Bin the people

            # On the final timepoint in the year, normalize
            if sim.t in rdict.timepoints:

                if 'prevalence' in rkey:
                    if 'hpv' in rkey:  # Denominator is whole population
                        if rdict.by_hiv:
                            if rdict.hiv_attr == 'with':
                                inds = sc.findinds(ppl['hiv'])
                            elif rdict.hiv_attr == 'no':
                                inds = sc.findinds(~ppl['hiv'])
                            denom = bin_ages(inds=inds, bins=bins)
                        else:
                            denom = bin_ages(inds=ppl.alive, bins=bins)
                    else:  # Denominator is females
                        denom = bin_ages(inds=ppl.is_female_alive, bins=bins)
                    if rdict.by_genotype: denom = denom[None, :]
                    self.results[rkey][date] = self.results[rkey][date] / (denom)

                if 'incidence' in rkey:
                    if 'hpv' in rkey:  # Denominator is susceptible population
                        inds = sc.findinds(ppl.is_female_alive & ~ppl.cancerous.any(axis=0))
                        denom = bin_ages(inds=hpu.true(ppl.sus_pool), bins=bins)
                    else:  # Denominator is females at risk for cancer
                        if rdict.by_hiv:
                            if rdict.hiv_attr == 'with':
                                inds = sc.findinds(ppl.is_female_alive & ppl['hiv'] * ~ppl.cancerous.any(axis=0))
                            elif rdict.hiv_attr == 'no':
                                inds = sc.findinds(ppl.is_female_alive & ~ppl['hiv'] * ~ppl.cancerous.any(axis=0))
                        else:
                            inds = sc.findinds(ppl.is_female_alive & ~ppl.cancerous.any(axis=0))
                        denom = bin_ages(inds, bins) / 1e5  # CIN and cancer are per 100,000 women
                    # if 'total' not in result and 'cancer' not in result: denom = denom[None, :] # THIS IS IT!!!!
                    self.results[rkey][date] = self.results[rkey][date] / denom

                if 'mortality' in rkey:
                    # first need to find people who died of other causes today and add them back into denom
                    denom = bin_ages(inds=ppl.is_female_alive, bins=bins)
                    scale_factor =  1e5  # per 100,000 women
                    denom /= scale_factor
                    self.results[rkey][date] = self.results[rkey][date] / denom

        return


    def finalize(self, sim):
        super().finalize()
        for rkey, rdict in self.result_args.items():
            recorded_dates = [k for k in self.results[rkey].keys()][1:]
            validate_recorded_dates(sim, requested_dates=rdict.years, recorded_dates=recorded_dates, die=self.die)
            if 'compute_fit' in rdict.keys():
                self.mismatch += self.compute_mismatch(rkey)

        # Add to sim.fit
        if hasattr(sim,'fit'): sim.fit += self.mismatch
        else: sim.fit = self.mismatch

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
            reduced_analyzer.results[reskey] = dict()
            reduced_analyzer.results[reskey]['bins'] = base_analyzer.results[reskey]['bins']
            for year,tp in zip(base_analyzer.result_args[reskey].years, base_analyzer.result_args[reskey].timepoints):
                ashape = analyzer.results[reskey][year].shape # Figure out dimensions
                new_ashape = ashape + (len(analyzers),)
                raw[reskey][year] = np.zeros(new_ashape)
                for a, analyzer in enumerate(analyzers):
                    vals = analyzer.results[reskey][year]
                    if len(ashape) == 1:
                        raw[reskey][year][:, a] = vals
                    elif len(ashape) == 2:
                        raw[reskey][year][:, :, a] = vals

                # Summarizing the aggregated list
                reduced_analyzer.results[reskey][year] = sc.objdict()
                if use_mean:
                    r_mean = np.mean(raw[reskey][year], axis=-1)
                    r_std = np.std(raw[reskey][year], axis=-1)
                    reduced_analyzer.results[reskey][year].best = r_mean
                    reduced_analyzer.results[reskey][year].low  = r_mean - bounds * r_std
                    reduced_analyzer.results[reskey][year].high = r_mean + bounds * r_std
                else:
                    reduced_analyzer.results[reskey][year].best = np.quantile(raw[reskey][year], q=0.5, axis=-1)
                    reduced_analyzer.results[reskey][year].low  = np.quantile(raw[reskey][year], q=quantiles['low'], axis=-1)
                    reduced_analyzer.results[reskey][year].high = np.quantile(raw[reskey][year], q=quantiles['high'], axis=-1)

        return reduced_analyzer


    def compute_mismatch(self, key):
        ''' Compute mismatch between analyzer results and datafile'''

        res = []
        resargs = self.result_args[key]
        results = self.results[key]

        for name, group in resargs.data.groupby(['genotype', 'year']):
            genotype = name[0]
            year = name[1]
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
            years_per_result = [len(rk['years']) for rk in self.result_args.values()]
            n_plots = sum(years_per_result)
            to_plot_args = []
            for rkey in self.result_keys:
                for year in self.result_args[rkey]['years']:
                    to_plot_args.append([rkey,year])
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
            ax.plot(x, resdict[date].T, color=resargs.color, **args.plot, label='Model')
            if ('data' in resargs.keys()) and (len(thisdatadf) > 0):
                ydata = np.array(thisdatadf.value)
                ax.scatter(x, ydata,  color=resargs.color, **args.scatter, label='Data')

        # Labels and legends
        ax.set_xlabel('Age')
        ax.set_title(f'{resargs.name} - {date}')
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
                for year in self.result_args[rkey]['years']:
                    ax = pl.subplot(n_rows, n_cols, plot_count)
                    ax = self.plot_single(ax, rkey, year, by_genotype, plot_args=plot_args, scatter_args=scatter_args)
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
        self.age_cin = []
        self.age_cancer = []
        self.dwelltime = dict()
        for state in ['precin', 'cin', 'total']:
            self.dwelltime[state] = []

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                current_age = sim.people.age[cancer_inds]
                date_exposed = sim.people.date_exposed[cancer_genotypes, cancer_inds]
                date_cin = sim.people.date_cin[cancer_genotypes, cancer_inds]
                hpv_time = (date_cin - date_exposed) * sim['dt']
                cin_time = (sim.t - date_cin) * sim['dt']
                total_time = (sim.t - date_exposed) * sim['dt']
                self.age_causal += (current_age - total_time).tolist()
                self.age_cin += (current_age - cin_time).tolist()
                self.age_cancer += current_age.tolist()
                self.dwelltime['precin'] += hpv_time.tolist()
                self.dwelltime['cin'] += cin_time.tolist()
                self.dwelltime['total'] += total_time.tolist()
        return

    def finalize(self, sim=None):
        ''' Convert things to arrays '''


class dalys(Analyzer):
    """
    Analyzer for computing DALYs.
    """

    def __init__(self, start=None, life_expectancy=84, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.si = None  # Start index - calculated upon initialization based on sim time vector
        self.df = None  # Results dataframe
        self.disability_weights = sc.objdict(
            weights=[0.288, 0.049, 0.451, 0.54],    # From GBD2017 - see Table A2.1 https://www.thelancet.com/cms/10.1016/S2214-109X(20)30022-X/attachment/0f63cf98-5eb9-48eb-af4f-6abe8fdff544/mmc1.pdf
            time_fraction=[0.05, 0.85, 0.09, 0.01],     # Estimates based on durations
        )
        self.life_expectancy = life_expectancy  # Should typically use country-specific values
        return

    @property
    def av_disutility(self):
        """ The average disability weight over duration of cancer """
        dw = self.disability_weights
        len_dw = len(dw.weights)
        return sum([dw.weights[i]*dw.time_fraction[i] for i in range(len_dw)])

    def initialize(self, sim):
        super().initialize(sim)
        if self.start is None: self.start=sim['start']
        self.si = sc.findfirst(sim.res_yearvec, self.start)
        self.npts = len(sim.res_yearvec[self.si:])
        self.years = sim.res_yearvec[self.si:]
        self.yll = np.zeros(self.npts)
        self.yld = np.zeros(self.npts)
        self.dalys = np.zeros(self.npts)
        return

    def apply(self, sim):

        if sim.yearvec[sim.t] >= self.start:
            ppl = sim.people
            li = np.floor(sim.yearvec[sim.t])
            idx = sc.findfirst(self.years, li)

            # Get new people with cancer and add up all their YLL and YLD now (incidence-based DALYs)
            new_cancers = ppl.date_cancerous == sim.t
            new_cancer_inds = hpu.true(new_cancers)
            if len(new_cancer_inds):
                self.yld[idx] += sum(ppl.scale[new_cancer_inds] * ppl.dur_cancer[new_cancers] * self.av_disutility)
                age_death = (ppl.age[new_cancer_inds]+ppl.dur_cancer[new_cancers])
                years_left = np.maximum(0, self.life_expectancy - age_death)
                self.yll[idx] += sum(ppl.scale[new_cancer_inds]*years_left)

        return

    def finalize(self, sim):
        self.dalys = self.yll + self.yld
        return



#%% Additional utilities
analyzer_map = {
    'snapshot': snapshot,
    'age_pyramid': age_pyramid,
    'age_results': age_results,
    'age_causal_infection': age_causal_infection,
    'dalys': dalys,
}

