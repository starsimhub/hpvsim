'''
Core plotting functions for simulations, multisims, and scenarios.
'''

import numpy as np
import pylab as pl
import sciris as sc
import pandas as pd
from . import misc as hpm
from . import defaults as hpd
from .settings import options as hpo


__all__ = ['plot_sim', 'plot_scens', 'plot_scen_age_results', 'plot_result', 'plot_people']


#%% Plotting helper functions

def handle_args(fig_args=None, plot_args=None, scatter_args=None, axis_args=None, fill_args=None,
                bar_args=None, legend_args=None, date_args=None, show_args=None, style_args=None, **kwargs):
    ''' Handle input arguments -- merge user input with defaults; see sim.plot for documentation '''

    # Set defaults
    defaults = sc.objdict()
    defaults.fig        = sc.objdict(figsize=(10, 8), num=None)
    defaults.plot       = sc.objdict(lw=1.5, alpha= 0.7)
    defaults.scatter    = sc.objdict(s=20, marker='s', alpha=0.7, zorder=1.75, datastride=1) # NB: 1.75 is above grid lines but below plots
    defaults.axis       = sc.objdict(left=0.10, bottom=0.08, right=0.95, top=0.95, wspace=0.30, hspace=0.30)
    defaults.fill       = sc.objdict(alpha=0.2)
    defaults.bar        = sc.objdict(width=0.15)
    defaults.legend     = sc.objdict(loc='best', frameon=False)
    defaults.date       = sc.objdict(as_dates=True, dateformat=None, rotation=None, start=None, end=None)
    defaults.show       = sc.objdict(data=True, ticks=True, interventions=True, legend=True, outer=False, tight=False, maximize=False)
    defaults.style      = sc.objdict(style=None, dpi=None, font=None, fontsize=None, grid=None, facecolor=None) # Use HPVsim global defaults

    # Handle directly supplied kwargs
    for dkey,default in defaults.items():
        keys = list(kwargs.keys())
        for kw in keys:
            if kw in default.keys():
                default[kw] = kwargs.pop(kw)

    # Handle what to show
    show_keys = ['data', 'ticks', 'interventions', 'legend']
    if show_args in [True, False]: # Handle all on or all off
        show_bool = show_args
        show_args = dict()
        for k in show_keys:
            show_args[k] = show_bool

    # Merge arguments together
    args = sc.objdict()
    args.fig        = sc.mergedicts(defaults.fig,       fig_args)
    args.plot       = sc.mergedicts(defaults.plot,      plot_args)
    args.scatter    = sc.mergedicts(defaults.scatter,   scatter_args)
    args.axis       = sc.mergedicts(defaults.axis,      axis_args)
    args.fill       = sc.mergedicts(defaults.fill,      fill_args)
    args.bar        = sc.mergedicts(defaults.bar,       bar_args)
    args.legend     = sc.mergedicts(defaults.legend,    legend_args)
    args.date       = sc.mergedicts(defaults.date,      date_args)
    args.show       = sc.mergedicts(defaults.show,      show_args)
    args.style      = sc.mergedicts(defaults.style,     style_args)

    # Handle potential rcParams keys
    keys = list(kwargs.keys())
    for key in keys:
        if key in pl.rcParams:
            args.style[key] = kwargs.pop(key)

    # If unused keyword arguments remain, parse or raise an error
    if len(kwargs):

        # Everything remaining is not found
        notfound = sc.strjoin(kwargs.keys())
        valid = sc.strjoin(sorted(set([k for d in defaults.values() for k in d.keys()]))) # Remove duplicates and order
        errormsg = f'The following keywords could not be processed:\n{notfound}\n\n'
        errormsg += f'Valid keywords are:\n{valid}\n\n'
        errormsg += 'For more precise plotting control, use fig_args, plot_args, etc.'
        raise sc.KeyNotFoundError(errormsg)

    # Handle what to show
    show_keys = ['data', 'ticks', 'interventions', 'legend']
    if show_args in [True, False]: # Handle all on or all off
        for k in show_keys:
            args.show[k] = show_args

    return args


def handle_show(do_show):
    ''' Helper function to handle the slightly complex logic of show -- not for users '''
    backend = pl.get_backend()
    if do_show is None:  # If not supplied, reset to global value
        do_show = hpo.show
    if backend == 'agg': # Cannot show plots for a non-interactive backend
        do_show = False
    if do_show: # Now check whether to show, and atually do it
        pl.show()
    return do_show


def handle_show_return(do_show=None, fig=None, figs=None):
    ''' Helper function to handle both show and what to return -- a nothing if Jupyter, else a figure '''

    figlist = sc.mergelists(fig, figs) # Usually just one figure, but here for completeness

    # Show the figure, or close it
    do_show = handle_show(do_show)
    if hpo.close and not do_show:
        for f in figlist:
            pl.close(f)

    # Return the figure or figures unless we're in Jupyter
    if not hpo.returnfig:
        return
    else:
        if figs is not None:
            return figlist
        else:
            return fig


def handle_to_plot(kind, to_plot, n_cols, sim, check_ready=True):
    ''' Handle which quantities to plot '''

    # Allow default kind to be overwritten by to_plot -- used by msim.plot()
    if isinstance(to_plot, tuple):
        kind, to_plot = to_plot # Split the tuple

    # Check that results are ready
    if check_ready and not sim.results_ready:
        errormsg = 'Cannot plot since results are not ready yet -- did you run the sim?'
        raise RuntimeError(errormsg)

    # Define allowable choices for plotting - default plot type depends on result type
    allkeys = sim.result_keys('all')
    time_series_keys = sim.result_keys('total')+sim.result_keys('genotype')+sim.result_keys('sex')+sim.result_keys('type_dysp')
    age_dist_keys = sim.result_keys('age')
    type_dysp_keys = ['type_dysp']
    valid_keys = allkeys+type_dysp_keys
    def check_plot_type(which):
        if which in time_series_keys: return 'time_series'
        elif which in age_dist_keys: return 'age_dist'
        elif which in type_dysp_keys: return 'type_dysp'
        else:
            raise ValueError(f'Plot type of {which} not understood.')

    analyzer_keys = [a.label for a in sim.get_analyzers()] # Defaults = whatever analyzer.plot() gives
    n_extra_plots = 0 # Keep track of the number of extra plots from analyzers

    # If to_plot is a single valid key, turn it into a list
    if to_plot in valid_keys: to_plot = sc.tolist(to_plot)
    if isinstance(to_plot, hpd.plot_args): to_plot = sc.tolist(to_plot)

    # If not specified or specified as another string, load defaults
    if to_plot is None or isinstance(to_plot, str):
        to_plot = hpd.get_default_plots(which=to_plot, kind=kind, sim=sim)

    # If it's a dictionary, translate it to a list but store the names
    names = None
    if isinstance(to_plot, dict):
        to_plot_orig = to_plot # Hold onto original
        names = [k for k in to_plot.keys()]
        to_plot = [k for k in to_plot.values()]

    # Validate list
    if isinstance(to_plot, list):
        to_plot_orig = to_plot[:] # Hold onto original for the moment
        to_plot = sc.autolist()
        invalid = sc.autolist()

        # Loop over items in list and validate them
        for rn,reskey in enumerate(to_plot_orig):

            # If it's a string, we construct default plot args by checking what kind of result it is
            if isinstance(reskey, str):
                if reskey in valid_keys:
                    if names is not None: name = names[rn]
                    else:
                        if reskey in allkeys:
                            name = sim.results[reskey].name
                        elif reskey == 'type_dysp':
                            name = 'HPV types by cytology'
                    if reskey in time_series_keys:
                        to_plot += hpd.plot_args(reskey, name=name, plot_type='time_series')
                    elif reskey in age_dist_keys:
                        to_plot += hpd.plot_args(reskey, name=name, plot_type='age_dist', year=sim.results['year'][-1])
                    elif reskey in type_dysp_keys:
                        to_plot += hpd.plot_args(reskey, name=name, plot_type='type_dysp', year=sim.results['year'][-1])
                else:
                    invalid += reskey

            # If it's plot args, we validate the years and set defaults
            elif isinstance(reskey, hpd.plot_args):
                if reskey.year == 'last': reskey.year = sim.results['year'][-1]
                if reskey.plot_type is None: # Add sensible defaults if not supplied
                    if reskey.keys[0] in age_dist_keys:
                        reskey.plot_type = 'age_dist'
                    elif reskey.keys[0] in type_dysp_keys:
                        reskey.plot_type = 'type_dysp'
                to_plot += reskey

            # If it's a list, we ned to choose a single plot type
            elif isinstance(reskey, list):
                if names is not None: name = names[rn]
                else: name = sim.results[reskey[0]].name # Use the name of the first result
                plot_types = [check_plot_type(rkey) for rkey in reskey]
                if 'time_series' in plot_types: # If no other info is provided, assume we want to plot them all as time series
                    plot_type = 'time_series'
                    year=None
                elif 'age_dist' in plot_types:
                    plot_type = 'age_dist'
                    year = sim.results['year'][-1]
                else:
                    plot_type = 'type_dysp'
                    year = sim.results['year'][-1]
                to_plot += hpd.plot_args(reskey, name=name, plot_type=plot_type, year=year)

        # Raise an error if there are any invalid keys
        if len(invalid):
            errormsg = f'The following key(s) are invalid:\n{sc.strjoin(invalid)}\n\nValid keys are:\n{sc.strjoin(valid_keys)}.'
            raise sc.KeyNotFoundError(errormsg)

    # Get total number of plots and calculate rows and columns
    n_plots = len(to_plot) + n_extra_plots
    if n_cols is None:
        max_rows = 5 # Assumption -- if desired, the user can override this by setting n_cols manually
        n_cols = int((n_plots-1)//max_rows + 1) # This gives 1 column for 1-4, 2 for 5-8, etc.
    n_rows,n_cols = sc.get_rows_cols(n_plots, ncols=n_cols) # Inconsistent naming due to HPVsim/Matplotlib conventions

    return to_plot, n_cols, n_rows


def create_figs(args, sep_figs, fig=None, ax=None):
    '''
    Create the figures and set overall figure properties. If a figure is supplied,
    reset the axes labels for automatic use by other plotting functions (i.e. ax1, ax2, etc.)
    '''
    if sep_figs:
        fig = None
        figs = []
    else:
        if fig is None:
            if ax is None:
                fig = pl.figure(**args.fig) # Create the figure if none is supplied
            else:
                fig = ax.figure
        else:
            for i,fax in enumerate(fig.axes):
                fax.set_label(f'ax{i+1}')
        figs = None
    pl.subplots_adjust(**args.axis)
    return fig, figs


def create_subplots(figs, fig, shareax, n_rows, n_cols, pnum, fig_args, sep_figs, log_scale, title=None):
    ''' Create subplots and set logarithmic scale '''

    # Try to find axes by label, if they've already been defined -- this is to avoid the deprecation warning of reusing axes
    label = f'ax{pnum+1}'
    ax = None
    try:
        for fig_ax in fig.axes:
            if fig_ax.get_label() == label:
                ax = fig_ax
                break
    except:
        pass

    # Handle separate figs
    if sep_figs:
        figs.append(pl.figure(**fig_args))
        if ax is None:
            ax = pl.subplot(111, label=label)
    else:
        if ax is None:
            ax = pl.subplot(n_rows, n_cols, pnum+1, sharex=shareax, label=label)

    # Handle log scale
    if log_scale:
        if isinstance(log_scale, list):
            if title in log_scale:
                ax.set_yscale('log')
        else:
            ax.set_yscale('log')

    return ax


def plot_data(sim, ax, key, scatter_args, color=None):
    ''' Add data to the plot '''
    if sim.data is not None and key in sim.data and len(sim.data[key]):
        if color is None:
            color = sim.results[key].color
        datastride = scatter_args.pop('datastride', 1) # Temporarily pop so other arguments pass correctly to ax.scatter()
        x = np.array(sim.data.index)[::datastride]
        y = np.array(sim.data[key])[::datastride]
        ax.scatter(x, y, c=[color], label='Data', **scatter_args)
        scatter_args['datastride'] = datastride # Restore
    return


def plot_interventions(sim, ax):
    ''' Add interventions to the plot '''
    for intervention in sim['interventions']:
        if hasattr(intervention, 'plot_intervention'): # Don't plot e.g. functions
            intervention.plot_intervention(sim, ax)
    return


def title_grid_legend(ax, title, grid, commaticks, setylim, legend_args, show_args, show_legend=True):
    ''' Plot styling -- set the plot title, add a legend, and optionally add gridlines'''

    # Handle show_legend being in the legend args, since in some cases this is the only way it can get passed
    if 'show_legend' in legend_args:
        show_legend = legend_args.pop('show_legend')
        popped = True
    else:
        popped = False

    # Show the legend
    if show_legend and show_args['legend']: # It's pretty ugly, but there are multiple ways of controlling whether the legend shows

        # Remove duplicate entries
        handles, labels = ax.get_legend_handles_labels()
        unique_inds = np.sort(np.unique(labels, return_index=True)[1])
        handles = [handles[u] for u in unique_inds]
        labels  = [labels[u]  for u in unique_inds]

        # Actually make legend
        ax.legend(handles=handles, labels=labels, **legend_args)

    # If we removed it from the legend_args dict, put it back now
    if popped:
        legend_args['show_legend'] = show_legend

    # Set the title, gridlines, and color
    ax.set_title(title)

    # Set the y axis style
    if setylim and ax.yaxis.get_scale() != 'log':
        ax.set_ylim(bottom=0)
    if commaticks:
        ylims = ax.get_ylim()
        if ylims[1] >= 1000:
            sc.commaticks(ax=ax)

    # Optionally remove x-axis labels except on bottom plots -- don't use ax.label_outer() since we need to keep the y-labels
    if show_args['outer']:
        lastrow = ax.get_subplotspec().is_last_row()
        if not lastrow:
            for label in ax.get_xticklabels(which="both"):
                label.set_visible(False)
            ax.set_xlabel('')

    return



def tidy_up(fig, figs=None, do_save=False, fig_path=None, do_show=False, args=None):
    ''' Handle saving, figure showing, and what value to return '''

    figlist = sc.mergelists(fig, figs) # Usually just one figure, but here for completeness

    # Optionally maximize -- does not work on all systems
    if args is not None and hasattr(args, 'show') and args.show['maximize']:
        for f in figlist:
            sc.maximize(fig=f)
        pl.pause(0.01) # Force refresh

    # Use tight layout for all figures
    if args is not None and hasattr(args, 'show') and args.show['tight']:
        for f in figlist:
            sc.figlayout(fig=f)

    # Handle saving
    if do_save:
        if isinstance(fig_path, str): # No figpath provided - see whether do_save is a figpath
            fig_path = sc.makefilepath(fig_path) # Ensure it's valid, including creating the folder
        hpm.savefig(fig=figlist, filename=fig_path) # Save the figure

    return handle_show_return(do_show, fig=fig, figs=figs)


def set_line_options(input_args, reskey, resnum, default):
    '''From the supplied line argument, usually a color or label, decide what to use '''
    if input_args is not None:
        if isinstance(input_args, dict): # If it's a dict, pull out this value
            output = input_args[reskey]
        elif isinstance(input_args, list): # If it's a list, ditto
            output = input_args[resnum]
        else: # Otherwise, assume it's the same value for all
            output = input_args
    else:
        output = default # Default value
    return output



#%% Individual plotting functions to create particular plots
def plot_time_series(ax, sim, reskey, resnum, args, colors=None, labels=None, plot_burnin=False):
    ''' Plot time series data, i.e. the usual contents of sim.results '''

    # Initialize some variables
    bi = 0 if plot_burnin else int(sim['burnin'])
    total_keys = sim.result_keys('total')
    sex_keys = sim.result_keys('sex')
    genotype_keys = sim.result_keys('genotype')
    res_t = sim.results['year'][bi:]
    res = sim.results[reskey]

    # The exact plotting call depends on what kind of core result key we're dealing with
    # Simplest case: it's a total result, i.e. not disagreggated by genotype or sex
    if reskey in total_keys:
        color = set_line_options(colors, reskey, resnum, res.color)  # Choose the color
        label = set_line_options(labels, reskey, resnum, res.name)  # Choose the label
        ax.plot(res_t, res.values[bi:], label=label, **args.plot, c=color)  # Plot result

    elif reskey in sex_keys:
        n_sexes = 2
        sex_colors = ['#4679A2', '#A24679']
        sex_labels = ['males', 'females']
        for sex in range(n_sexes):
            # Colors and labels
            v_color = sex_colors[sex]
            v_label = sex_labels[sex]  # TODO this should also come from the sim
            color = set_line_options(colors, reskey, resnum, v_color)  # Choose the color
            label = set_line_options(labels, reskey, resnum, res.name)  # Choose the label
            if label:   label += f' - {v_label}'
            else:       label = v_label
            ax.plot(res_t, res.values[sex, bi:], label=label, **args.plot, c=color)  # Plot result

    elif reskey in genotype_keys:
        ng = sim['n_genotypes']
        g_colors = sc.gridcolors(ng)
        for genotype in range(ng):
            # Colors and labels
            g_color = g_colors[genotype]
            geno_obj = sim['genotypes'][genotype]
            if sc.isnumber(geno_obj):  # TODO: figure out why this is sometimes an int and sometimes an obj
                v_label = str(geno_obj)
            elif sc.isstring(geno_obj):
                v_label = geno_obj
            else:
                v_label = geno_obj.label
            color = set_line_options(colors, reskey, resnum, g_color)  # Choose the color
            label = set_line_options(labels, reskey, resnum, res.name)  # Choose the label
            if label:
                label += f' - {v_label}'
            else:
                label = v_label
            ax.plot(res_t, res.values[genotype, bi:], label=label, **args.plot, c=color)  # Plot result

    else:
        raise ValueError(f'Result {reskey} not understood.')

    return ax, color


def plot_type_bars(sim, ax, date, args):
    '''
    Plot HPV types by cytology
    '''

    idx = sc.findinds(sim.res_yearvec, date)[0]
    labels = sc.autolist()
    resdict = sc.objdict()
    for rkey in sim.result_keys('type_dysp'):
        labels += sim.results[rkey].name
        resdict[rkey] = sim.results[rkey][:,idx]
    g_labels = sim['genotypes']

    # Grouped bar plot with n_groups bars (one for each state) and ng bars per group
    n_bars_per_group = sim['n_genotypes']
    n_groups = len(resdict)
    x = np.arange(n_groups)
    width = args.bar.width

    # Set position of bar on x axis
    xpositions = [x]
    for group_no in range(1, n_bars_per_group):
        xpositions.append([xi + width for xi in xpositions[-1]])

    # Plot bars
    for bar_no in range(n_bars_per_group):
        ydata = [resdict[k][bar_no] for k in range(n_groups)]  # Have to rearrange
        ax.bar(xpositions[bar_no], ydata, **args.bar, label=g_labels[bar_no])

    # Add xticks on the middle of the group bars
    ax.set_xticks([r + width for r in range(len(x))], labels)

    return ax


def plot_age_dist(sim, ax, reskey, date, args):
    '''
    Function to plot a single age result for a single date. Requires an axis as
    input and will generally be called by a helper function rather than directly.
    '''
    idx = sc.findinds(sim.res_yearvec, date)[0]
    res = sim.results[reskey]
    x = sim['age_bins'][:-1]
    ax.plot(x, res.values[:,idx], color=res.color, **args.plot, label=res.name)
    return ax



#%% Core plotting functions that unite the individual plotting functions to create figures for sims, scenarios, multisims, etc

def plot_sim(to_plot=None, sim=None, fig=None, ax=None, do_save=None, fig_path=None,
             fig_args=None, plot_args=None, scatter_args=None, axis_args=None, fill_args=None,
             legend_args=None, date_args=None, show_args=None, style_args=None, n_cols=None,
             grid=True, commaticks=True, setylim=True, log_scale=False, colors=None, labels=None,
             do_show=None, sep_figs=False, plot_burnin=False, **kwargs):
    ''' Plot the results of a single simulation -- see Sim.plot() for documentation '''

    # Handle inputs
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args, fill_args=fill_args,
                       legend_args=legend_args, show_args=show_args, date_args=date_args, style_args=style_args, **kwargs)
    to_plot, n_cols, n_rows = handle_to_plot('sim', to_plot, n_cols, sim)

    # Do the plotting
    with hpo.with_style(args.style):

        # Create the figures
        fig, figs = create_figs(args, sep_figs, fig, ax)

        # Determine whether to share x axis
        do_sharex = False
        plot_types = [tp.plot_type for tp in to_plot]
        if len(set(plot_types))==1: do_sharex = True

        # Iterate through to_plot to figure out what to plot & how to plot it
        for pnum,plot_arg in enumerate(to_plot):
            sharex = ax if do_sharex else None
            title = plot_arg.name
            plot_type = plot_arg.plot_type
            ax = create_subplots(figs, fig, sharex, n_rows, n_cols, pnum, args.fig, sep_figs, log_scale, title)

            if plot_type == 'time_series':
                for resnum,reskey in enumerate(plot_arg.keys):
                    ax, color = plot_time_series(ax, sim, reskey, resnum, args, labels=labels, colors=colors, plot_burnin=plot_burnin)
                    if args.show['data']:
                        plot_data(sim, ax, reskey, args.scatter, color=color)  # Plot the data
                title_grid_legend(ax, title, grid, commaticks, setylim, args.legend, args.show)

            elif plot_type == 'type_dysp':
                ax = plot_type_bars(sim, ax, plot_arg.year, args)
                title_grid_legend(ax, title, grid, commaticks, setylim, sc.mergedicts(args.legend,dict(title=int(plot_arg.year))), args.show)

            elif plot_type == 'age_dist':
                for resnum,reskey in enumerate(plot_arg.keys):
                    ax = plot_age_dist(sim, ax, reskey, plot_arg.year, args)
                title_grid_legend(ax, title, grid, commaticks, setylim, sc.mergedicts(args.legend,dict(title=int(plot_arg.year))), args.show)


        output = tidy_up(fig, figs, do_save, fig_path, do_show, args)

    return output


def plot_scens(to_plot=None, scens=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
         scatter_args=None, axis_args=None, fill_args=None, legend_args=None, date_args=None,
         show_args=None, style_args=None, n_cols=None, grid=False, commaticks=True, setylim=True,
         log_scale=False, colors=None, labels=None, do_show=None, sep_figs=False, fig=None, ax=None,
         plot_burnin=False,**kwargs):
    ''' Plot the results of a scenario -- see Scenarios.plot() for documentation '''

    # Handle inputs
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args, fill_args=fill_args,
                   legend_args=legend_args, show_args=show_args, date_args=date_args, style_args=style_args, **kwargs)
    to_plot, n_cols, n_rows = handle_to_plot('scens', to_plot, n_cols, sim=scens.base_sim, check_ready=False) # Since this sim isn't run

    # Do the plotting
    with hpo.with_style(args.style):
        fig, figs = create_figs(args, sep_figs, fig, ax)
        default_colors = sc.gridcolors(ncolors=len(scens.sims))
        for pnum,plot_arg in enumerate(to_plot):
            title = plot_arg.name
            ax = create_subplots(figs, fig, ax, n_rows, n_cols, pnum, args.fig, sep_figs, log_scale, title)
            reskeys = sc.promotetolist(plot_arg.keys) # In case it's a string
            for reskey in reskeys:
                res_t = scens.res_yearvec
                resdata = scens.results[reskey]
                for snum,scenkey,scendata in resdata.enumitems():
                    sim = scens.sims[scenkey][0] # Pull out the first sim in the list for this scenario
                    bi = 0 if plot_burnin else int(sim['burnin'])
                    genotypekeys = sim.result_keys('genotype')
                    sexkeys = sim.result_keys('sex')
                    if reskey in genotypekeys:
                        ng = sim['n_genotypes']
                        genotype_colors = sc.gridcolors(ng)
                        for genotype in range(ng):
                            res_y = scendata.best
                            color = genotype_colors[genotype]
                            label = sim['genotypes'][genotype]
                            ax.fill_between(res_t, scendata.low[genotype,:], scendata.high[genotype,:], color=color, **args.fill)  # Create the uncertainty bound
                            ax.plot(res_t[bi:], res_y[genotype,bi:], label=label, c=color, **args.plot)  # Plot the actual line
                    elif reskey in sexkeys:
                        n_sexes = 2
                        sex_colors = ['#4679A2', '#A24679']
                        sex_labels = ['males', 'females']
                        for sex in range(n_sexes):
                            # Colors and labels
                            res_y = scendata.best[sex, :]
                            color = sex_colors[sex]
                            label = reskey + sex_labels[sex]
                            ax.fill_between(res_t[bi:], scendata.low[genotype, bi:], scendata.high[genotype, bi:],
                                            color=color, **args.fill)  # Create the uncertainty bound
                            ax.plot(res_t[bi:], res_y[bi:], label=label, c=color, **args.plot)  # Plot the actual line
                    else:
                        res_y = scendata.best
                        color = set_line_options(colors, scenkey, snum, default_colors[snum])  # Choose the color
                        label = set_line_options(labels, scenkey, snum, scendata.name)  # Choose the label
                        ax.fill_between(res_t[bi:], scendata.low[bi:], scendata.high[bi:], color=color, **args.fill)  # Create the uncertainty bound
                        ax.plot(res_t[bi:], res_y[bi:], label=label, c=color, **args.plot)  # Plot the actual line

                    if args.show['interventions']:
                        plot_interventions(sim, ax) # Plot the interventions
            if args.show['legend']:
                title_grid_legend(ax, title, grid, commaticks, setylim, args.legend, args.show, pnum==0) # Configure the title, grid, and legend -- only show legend for first

    return tidy_up(fig, figs, do_save, fig_path, do_show, args)


def plot_scen_age_results(analyzer_ref=0, to_plot=None, scens=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
         scatter_args=None, axis_args=None, fill_args=None, legend_args=None, date_args=None,
         show_args=None, style_args=None, n_cols=None, grid=False, commaticks=True, setylim=True,
         log_scale=False, colors=None, labels=None, do_show=None, sep_figs=False, fig=None, ax=None,
         plot_burnin=False, plot_type='sns.boxplot', **kwargs):
    ''' Plot age results of a scenario'''

    # Import Seaborn here since slow
    if sc.isstring(plot_type) and plot_type.startswith('sns'):
        import seaborn as sns
        plot_func = getattr(sns, plot_type.split('.')[1])
    else:
        plot_func = plot_type

    # Handle inputs
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args, fill_args=fill_args,
                   legend_args=legend_args, show_args=show_args, date_args=date_args, style_args=style_args, **kwargs)

    # Get the analyzer details from the base sim
    base_analyzer = scens.sims[0][0].get_analyzer(analyzer_ref)
    if not len(base_analyzer.results):
        errormsg = 'Cannot plot since no age results were recorded.'
        raise ValueError(errormsg)
    base_res = base_analyzer.results[0]

    result_keys   = base_analyzer.result_keys.keys()
    all_dates = [[date for date in r.keys() if date != 'bins'] for r in base_analyzer.results.values()]
    dates_per_result = [len(date_list) for date_list in all_dates]
    n_plots = sum(dates_per_result)
    n_rows, n_cols = sc.get_rows_cols(n_plots)

    # Construct dataframe for result storage
    n_runs = scens['n_runs']

    # Do the plotting
    with hpo.with_style(args.style):
        fig, figs = create_figs(args, sep_figs, fig, ax)

        pnum = 0
        age_labels = {}
        for rn,reskey in enumerate(result_keys):
            age_bins = base_analyzer.results[reskey]['bins']
            age_labels[reskey] = [str(int(age_bins[i])) + '-' + str(int(age_bins[i + 1])) for i in range(len(age_bins) - 1)]
            age_labels[reskey].append(str(int(age_bins[-1])) + '+')

            for tp in all_dates[rn]:

                # Construct a dataframe with things in the most logical order for plotting
                bins = []
                scen_names = []
                values = []
                n_bins = len(base_res['bins'])
                for bno, bin in enumerate(base_res['bins']):
                    for sno in range(len(scens.scenarios)):
                        for rep in range(n_runs):
                            bins.append(bin)
                            scen_key = scens.sims.keys()[sno]
                            scen_names.append(scens.scenarios[scen_key]['name'])
                            values.append(scens.sims[sno][rep].get_analyzer(analyzer_ref).results[reskey][tp][bno])
                replicates = np.arange(n_runs).tolist() * n_bins * len(scens.scenarios)
                resdict = dict(bin=bins, scen_name=scen_names, replicate=replicates, value=values)
                resdf = pd.DataFrame(resdict)

                # Start plot
                ax = pl.subplot(n_rows, n_cols, pnum+1)
                ax = plot_func(ax=ax, x="bin", y="value", hue="scen_name", data=resdf, dodge=True)
                ax.legend([], [], frameon=False) # Temporarily turn off legend
                title = f'{base_analyzer.result_properties[reskey].name} - {int(float(tp))}'
                if args.show['legend']:
                    title_grid_legend(ax, title, grid, commaticks, setylim, args.legend, args.show, pnum == 0)  # Configure the title, grid, and legend -- only show legend for first
                ax.set_xlabel("Age group")
                ax.set_xticklabels(age_labels[reskey])
                ax.set_ylabel("")
                pnum +=1


    return tidy_up(fig, figs, do_save, fig_path, do_show, args)


def plot_result(key, sim=None, fig_args=None, plot_args=None, axis_args=None, scatter_args=None,
                date_args=None, style_args=None, grid=False, commaticks=True, setylim=True, color=None, label=None,
                do_show=None, do_save=False, fig_path=None, fig=None, ax=None, plot_burnin=False, **kwargs):
    ''' Plot a single result -- see ``hpv.Sim.plot_result()`` for documentation '''

    # Handle inputs
    sep_figs = False # Only one figure
    fig_args  = sc.mergedicts({'figsize':(8,5)}, fig_args)
    axis_args = sc.mergedicts({'top': 0.95}, axis_args)
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args,
                       date_args=date_args, style_args=style_args, **kwargs)

    # Gather results
    res = sim.results[key]
    res_t = sim.results['year']
    bi = 0 if plot_burnin else int(sim['burnin'])
    if color is None:
        color = res.color

    # Do the plotting
    with hpo.with_style(args.style):
        fig, figs = create_figs(args, sep_figs, fig, ax)

        # Reuse the figure, if available
        if ax is None: # Otherwise, make a new one
            try:
                ax = fig.axes[0]
            except:
                ax = fig.add_subplot(111, label='ax1')

        if label is None:
            label = res.name
        if res.low is not None and res.high is not None:
            ax.fill_between(res_t[bi:], res.low[bi:], res.high[bi:], color=color, **args.fill) # Create the uncertainty bound

        ax.plot(res_t[bi:], res.values[bi:], c=color, label=label, **args.plot)
        plot_interventions(sim, ax) # Plot the interventions
        title_grid_legend(ax, res.name, grid, commaticks, setylim, args.legend, args.show) # Configure the title, grid, and legend

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, args)


# def plot_compare(df, log_scale=True, fig_args=None, axis_args=None, style_args=None, grid=False,
#                  commaticks=True, setylim=True, color=None, label=None, fig=None,
#                  do_save=None, do_show=None, fig_path=None, **kwargs):
#     ''' Plot a MultiSim comparison -- see MultiSim.plot_compare() for documentation '''

#     # Handle inputs
#     sep_figs = False
#     fig_args  = sc.mergedicts({'figsize':(8,8)}, fig_args)
#     axis_args = sc.mergedicts({'left': 0.16, 'bottom': 0.05, 'right': 0.98, 'top': 0.98, 'wspace': 0.50, 'hspace': 0.10}, axis_args)
#     args = handle_args(fig_args=fig_args, axis_args=axis_args, style_args=style_args, **kwargs)

#     # Map from results into different categories
#     mapping = {
#         'cum': 'Cumulative counts',
#         'new': 'New counts',
#         'n': 'Number in state',
#         'r': 'R_eff',
#         }
#     category = []
#     for v in df.index.values:
#         v_type = v.split('_')[0]
#         if v_type in mapping:
#             category.append(v_type)
#         else:
#             category.append('other')
#     df['category'] = category

#     # Plot
#     with cvo.with_style(args.style):
#         fig, figs = create_figs(args, sep_figs=False, fig=fig)
#         for i,m in enumerate(mapping):
#             not_r_eff = m != 'r'
#             if not_r_eff:
#                 ax = fig.add_subplot(2, 2, i+1)
#             else:
#                 ax = fig.add_subplot(8, 2, 10)
#             dfm = df[df['category'] == m]
#             logx = not_r_eff and log_scale
#             dfm.plot(ax=ax, kind='barh', logx=logx, legend=False)
#             if not(not_r_eff):
#                 ax.legend(loc='upper left', bbox_to_anchor=(0,-0.3))
#             ax.grid(True)

#     return tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, args)


#%% Other plotting functions
def plot_people(people, bins=None, width=1.0, alpha=0.6, fig_args=None, axis_args=None,
                plot_args=None, style_args=None, do_show=None, fig=None):
    ''' Plot statistics of a population -- see People.plot() for documentation '''

    # Handle inputs
    if bins is None:
        bins = np.arange(0,101)

    # Set defaults
    color     = [0.1,0.1,0.1] # Color for the age distribution
    n_rows    = 4 # Number of rows of plots
    offset    = 0.5 # For ensuring the full bars show up
    gridspace = 10 # Spacing of gridlines
    zorder    = 10 # So plots appear on top of gridlines

    # Handle other arguments
    fig_args   = sc.mergedicts(dict(figsize=(18,11)), fig_args)
    axis_args  = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35), axis_args)
    plot_args  = sc.mergedicts(dict(lw=1.5, alpha=0.6, c=color, zorder=10), plot_args)
    style_args = sc.mergedicts(style_args)

    # Compute statistics
    min_age = min(bins)
    max_age = max(bins)
    edges = np.append(bins, np.inf) # Add an extra bin to end to turn them into edges
    age_counts = np.histogram(people.age, edges)[0]

    with hpo.with_style(style_args):

        # Create the figure
        if fig is None:
            fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)

        # Plot age histogram
        pl.subplot(n_rows,2,1)
        pl.bar(bins, age_counts, color=color, alpha=alpha, width=width, zorder=zorder)
        pl.xlim([min_age-offset,max_age+offset])
        pl.xticks(np.arange(0, max_age+1, gridspace))
        pl.xlabel('Age')
        pl.ylabel('Number of people')
        pl.title(f'Age distribution ({len(people):n} people total)')

        # Plot cumulative distribution
        pl.subplot(n_rows,2,2)
        age_sorted = sorted(people.age)
        y = np.linspace(0, 100, len(age_sorted)) # Percentage, not hard-coded!
        pl.plot(age_sorted, y, '-', **plot_args)
        pl.xlim([0,max_age])
        pl.ylim([0,100]) # Percentage
        pl.xticks(np.arange(0, max_age+1, gridspace))
        pl.yticks(np.arange(0, 101, gridspace)) # Percentage
        pl.xlabel('Age')
        pl.ylabel('Cumulative proportion (%)')
        pl.title(f'Cumulative age distribution (mean age: {people.age.mean():0.2f} years)')

        # Calculate contacts
        lkeys = people.layer_keys()
        n_layers = len(lkeys)
        contact_counts = sc.objdict()
        for lk in lkeys:
            layer = people.contacts[lk]
            p1ages = people.age[layer['f']]
            p2ages = people.age[layer['m']]
            contact_counts[lk] = np.histogram(p1ages, edges)[0] + np.histogram(p2ages, edges)[0]

        # Plot contacts
        layer_colors = sc.gridcolors(n_layers)
        share_ax = None
        for w,w_type in enumerate(['total', 'percapita', 'weighted']): # Plot contacts in different ways
            for i,lk in enumerate(lkeys):
                contacts_lk = people.contacts[lk]
                members_lk = contacts_lk.members
                n_contacts = len(contacts_lk)
                n_members = len(members_lk)
                if w_type == 'total':
                    weight = 1
                    total_contacts = 2*n_contacts # x2 since each contact is undirected
                    ylabel = 'Number of contacts'
                    participation = n_members/len(people) # Proportion of people that have contacts in this layer
                    title = f'Total contacts for layer "{lk}": {total_contacts:n}\n({participation*100:.0f}% participation)'
                elif w_type == 'percapita':
                    age_counts_within_layer = np.histogram(people.age[members_lk], edges)[0]
                    weight = np.divide(1.0, age_counts_within_layer, where=age_counts_within_layer>0)
                    mean_contacts_within_layer = 2*n_contacts/n_members if n_members else 0  # Factor of 2 since edges are bi-directional
                    ylabel = 'Per capita number of contacts'
                    title = f'Mean contacts for layer "{lk}": {mean_contacts_within_layer:0.2f}'
                elif w_type == 'weighted':
                    weight = people.pars['beta']
                    total_weight = np.round(weight*2*n_contacts)
                    ylabel = 'Weighted number of contacts'
                    title = f'Total weight for layer "{lk}": {total_weight:n}'

                ax = pl.subplot(n_rows, n_layers, n_layers*(w+1)+i+1, sharey=share_ax)
                pl.bar(bins, contact_counts[lk]*weight, color=layer_colors[i], width=width, zorder=zorder, alpha=alpha)
                pl.xlim([min_age-offset,max_age+offset])
                pl.xticks(np.arange(0, max_age+1, gridspace))
                pl.xlabel('Age')
                pl.ylabel(ylabel)
                pl.title(title)
                if w_type == 'weighted':
                    share_ax = ax # Update shared axis



    return handle_show_return(fig=fig, do_show=do_show)

