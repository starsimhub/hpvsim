'''
Define the calibration class
'''

import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from . import misc as hpm
from . import plotting as hppl
from . import analysis as hpa
from . import parameters as hppar
from .settings import options as hpo # For setting global options


__all__ = ['Calibration']

def import_optuna():
    ''' A helper function to import Optuna, which is an optional dependency '''
    try:
        import optuna as op # Import here since it's slow
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = f'Optuna import failed ({str(E)}), please install first (pip install optuna)'
        raise ModuleNotFoundError(errormsg)
    return op


class Calibration(sc.prettyobj):
    '''
    A class to handle calibration of HPVsim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org), which must be installed separately (via
    pip install optuna).

    Note: running a calibration does not guarantee a good fit! You must ensure that
    you run for a sufficient number of iterations, have enough free parameters, and
    that the parameters have wide enough bounds. Please see the tutorial on calibration
    for more information.

    Args:
        sim          (Sim)  : the simulation to calibrate
        datafiles    (list) : list of datafile strings to calibrate to
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        genotype_pars(dict) : a dictionary of the genotype-specific parameters to calibrate of the format dict(genotype=dict(key1=[best, low, high]))
        hiv_pars     (dict) : a dictionary of the hiv-specific parameters to calibrate of the format dict(key1=[best, low, high])
        extra_sim_results (list) : list of result strings to store
        fit_args     (dict) : a dictionary of options that are passed to sim.compute_fit() to calculate the goodness-of-fit
        par_samplers (dict) : an optional mapping from parameters to the Optuna sampler to use for choosing new points for each; by default, suggest_float
        n_trials     (int)  : the number of trials per worker
        n_workers    (int)  : the number of parallel workers (default: maximum
        total_trials (int)  : if n_trials is not supplied, calculate by dividing this number by n_workers)
        name         (str)  : the name of the database (default: 'hpvsim_calibration')
        db_name      (str)  : the name of the database file (default: 'hpvsim_calibration.db')
        keep_db      (bool) : whether to keep the database after calibration (default: false)
        storage      (str)  : the location of the database (default: sqlite)
        rand_seed    (int)  : if provided, use this random seed to initialize Optuna runs (for reproducibility)
        label        (str)  : a label for this calibration object
        die          (bool) : whether to stop if an exception is encountered (default: false)
        verbose      (bool) : whether to print details of the calibration
        kwargs       (dict) : passed to hpv.Calibration()

    Returns:
        A Calibration object

    **Example**::

        sim = hpv.Sim(pars, genotypes=[16, 18])
        calib_pars = dict(beta=[0.05, 0.010, 0.20],hpv_control_prob=[.9, 0.5, 1])
        calib = hpv.Calibration(sim, calib_pars=calib_pars,
                                datafiles=['test_data/south_africa_hpv_data.xlsx',
                                           'test_data/south_africa_cancer_data.xlsx'],
                                total_trials=10, n_workers=4)
        calib.calibrate()
        calib.plot()

    '''

    def __init__(self, sim, datafiles, calib_pars=None, genotype_pars=None, hiv_pars=None, fit_args=None, extra_sim_result_keys=None, 
                 par_samplers=None, n_trials=None, n_workers=None, total_trials=None, name=None, db_name=None, estimator=None,
                 keep_db=None, storage=None, rand_seed=None, sampler=None, label=None, die=False, verbose=True):

        import multiprocessing as mp # Import here since it's also slow

        # Handle run arguments
        if n_trials  is None: n_trials  = 20
        if n_workers is None: n_workers = mp.cpu_count()
        if name      is None: name      = 'hpvsim_calibration'
        if db_name   is None: db_name   = f'{name}.db'
        if keep_db   is None: keep_db   = False
        if storage   is None: storage   = f'sqlite:///{db_name}'
        if total_trials is not None: n_trials = int(np.ceil(total_trials/n_workers))
        self.run_args   = sc.objdict(n_trials=int(n_trials), n_workers=int(n_workers), name=name, db_name=db_name,
                                     keep_db=keep_db, storage=storage, rand_seed=rand_seed, sampler=sampler)

        # Handle other inputs
        self.label          = label
        self.sim            = sim
        self.calib_pars     = calib_pars
        self.genotype_pars  = genotype_pars
        self.hiv_pars       = hiv_pars
        self.extra_sim_result_keys = extra_sim_result_keys
        self.fit_args       = sc.mergedicts(fit_args)
        self.par_samplers   = sc.mergedicts(par_samplers)
        self.die            = die
        self.verbose        = verbose
        self.calibrated     = False

        # Create age_results intervention
        self.target_data = []
        for datafile in datafiles:
            self.target_data.append(hpm.load_data(datafile))

        sim_results = sc.objdict()
        age_result_args = sc.objdict()

        # Go through each of the target keys and determine how we are going to get the results from sim
        for targ in self.target_data:
            targ_keys = targ.name.unique()
            if len(targ_keys) > 1:
                errormsg = f'Only support one set of targets per datafile, {len(targ_keys)} provided'
                raise ValueError(errormsg)
            if 'age' in targ.columns:
                age_result_args[targ_keys[0]] = sc.objdict(
                    datafile=sc.dcp(targ),
                    compute_fit=True,
                )
            else:
                sim_results[targ_keys[0]] = sc.objdict(
                    data=sc.dcp(targ)
                )

        ar = hpa.age_results(result_args=age_result_args)
        self.sim['analyzers'] += [ar]
        if hiv_pars is not None:
            self.sim['model_hiv'] = True # if calibrating HIV parameters, make sure model is running HIV
        self.sim.initialize()
        for rkey in sim_results.keys():
            sim_results[rkey].timepoints = sim.get_t(sim_results[rkey].data.year.unique()[0], return_date_format='str')[0]//sim.resfreq
            if 'weights' not in sim_results[rkey].data.columns:
                sim_results[rkey].weights = np.ones(len(sim_results[rkey].data))
        self.age_results_keys = age_result_args.keys()
        self.sim_results = sim_results
        self.sim_results_keys = sim_results.keys()

        self.result_args = sc.objdict()
        for rkey in self.age_results_keys + self.sim_results_keys:
            self.result_args[rkey] = sc.objdict()
            if 'hiv' in rkey:
                self.result_args[rkey].name = self.sim.hivsim.results[rkey].name
                self.result_args[rkey].color = self.sim.hivsim.results[rkey].color
            else:
                self.result_args[rkey].name = self.sim.results[rkey].name
                self.result_args[rkey].color = self.sim.results[rkey].color

        if self.extra_sim_result_keys:
            for rkey in self.extra_sim_result_keys:
                self.result_args[rkey] = sc.objdict()
                self.result_args[rkey].name = self.sim.results[rkey].name
                self.result_args[rkey].color = self.sim.results[rkey].color
        # Temporarily store a filename
        self.tmp_filename = 'tmp_calibration_%05i.obj'

        return


    def run_sim(self, calib_pars=None, genotype_pars=None, hiv_pars=None, label=None, return_sim=False):
        ''' Create and run a simulation '''
        sim = sc.dcp(self.sim)
        if label: sim.label = label

        new_pars = self.get_full_pars(sim=sim, calib_pars=calib_pars, genotype_pars=genotype_pars, hiv_pars=hiv_pars)
        sim.update_pars(new_pars)
        sim.initialize(reset=True, init_analyzers=False) # Necessary to reinitialize the sim here so that the initial infections get the right parameters

        # Run the sim
        try:
            sim.run()
            if return_sim:
                return sim
            else:
                return sim.fit

        except Exception as E:
            if self.die:
                raise E
            else:
                warnmsg = f'Encountered error running sim!\nParameters:\n{new_pars}\nTraceback:\n{sc.traceback()}'
                hpm.warn(warnmsg)
                output = None if return_sim else np.inf
                return output
            
    @staticmethod
    def update_dict_pars(name_pars, value_pars):
        ''' Function to update parameters from nested dict to nested dict's value '''
        new_pars = sc.dcp(name_pars)
        target_pars_flatten = sc.flattendict(value_pars)
        for key, val in target_pars_flatten.items():
            try: 
                sc.setnested(new_pars, list(key), val)
            except Exception as e:
                errormsg = f"Parameter {'_'.join(key)} is not part of the sim, nor is a custom function specified to use them"
                raise ValueError(errormsg)
        return new_pars
    
    def update_dict_pars_from_trial(self, name_pars, value_pars):
        ''' Function to update parameters from nested dict to trial parameter's value '''
        # new_pars = sc.dcp(name_pars)
        new_pars = {}
        name_pars_keys = sc.flattendict(name_pars).keys()
        for key in name_pars_keys:
            name = '_'.join(key)
            sc.setnested(new_pars, list(key), value_pars[name])
        return new_pars
    
    def update_dict_pars_init_and_bounds(self, initial_pars, par_bounds, target_pars):
        ''' Function to update initial parameters and parameter bounds from a trial pars dict'''
        target_pars_keys = sc.flattendict(target_pars)
        for key, val in target_pars_keys.items():
            name = '_'.join(key)
            initial_pars[name] = val[0]
            par_bounds[name] = np.array([val[1], val[2]])
        return initial_pars, par_bounds
       
    
    def get_full_pars(self, sim=None, calib_pars=None, genotype_pars=None, hiv_pars=None):
        ''' Make a full pardict from the subset of regular sim parameters, genotype parameters, and hiv parameters used in calibration'''
        # Prepare the parameters
        new_pars = {}     

        if genotype_pars is not None:
            new_pars['genotype_pars'] = self.update_dict_pars(sim['genotype_pars'], genotype_pars)

        if hiv_pars is not None:
            new_pars['hiv_pars'] = self.update_dict_pars(sim.hivsim.pars['hiv_pars'], hiv_pars)

        if calib_pars is not None:
            calib_pars_flatten = sc.flattendict(calib_pars)
            for key, val in calib_pars_flatten.items():
                if key[0] in sim.pars and key[0] not in new_pars:
                    new_pars[key[0]] = sc.dcp(sim.pars[key[0]])
                try:
                    sc.setnested(new_pars, list(key), val) # only update on keys that have values in sim.pars. If this line makes error, raise error errormsg 
                except Exception as e:
                    errormsg = f"Parameter {'_'.join(key)} is not part of the sim, nor is a custom function specified to use them"
                    raise ValueError(errormsg)       
        return new_pars

    def trial_pars_to_sim_pars(self, trial_pars=None, which_pars=None, return_full=True):
        '''
        Create genotype_pars and pars dicts from the trial parameters.
        Note: not used during self.calibrate.
        Args:
            trial_pars (dict): dictionary of parameters from a single trial. If not provided, best parameters will be used
            return_full (bool): whether to return a unified par dict ready for use in a sim, or the sim pars and genotype pars separately

        **Example**::
        
            sim = hpv.Sim(genotypes=[16, 18])
            calib_pars = dict(beta=[0.05, 0.010, 0.20],hpv_control_prob=[.9, 0.5, 1])
            genotype_pars = dict(hpv16=dict(prog_time=[3, 3, 10]))
            calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars
                                datafiles=['test_data/south_africa_hpv_data.xlsx',
                                           'test_data/south_africa_cancer_data.xlsx'],
                                total_trials=10, n_workers=4)
            calib.calibrate()
            new_pars = calib.trial_pars_to_sim_pars() # Returns best parameters from calibration in a format ready for sim running
            sim.update_pars(new_pars)
            sim.run()
        '''

        # Initialize
        calib_pars = sc.objdict()
        genotype_pars = sc.objdict()
        hiv_pars = sc.objdict()

        # Deal with trial parameters
        if trial_pars is None:
            try:
                if which_pars is None or which_pars==0:
                    trial_pars = self.best_pars
                else:
                    ddict = self.df.to_dict(orient='records')[which_pars]
                    trial_pars = {k:v for k,v in ddict.items() if k not in ['index','mismatch']}
            except:
                errormsg = 'No trial parameters provided.'
                raise ValueError(errormsg)

        # Handle genotype parameters
        if self.genotype_pars is not None:
            genotype_pars = self.update_dict_pars_from_trial(self.genotype_pars, trial_pars)

        # Handle hiv sim parameters
        if self.hiv_pars is not None:
            hiv_pars = self.update_dict_pars_from_trial(self.hiv_pars, trial_pars)

        # Handle regular sim parameters
        if self.calib_pars is not None:
            calib_pars = self.update_dict_pars_from_trial(self.calib_pars, trial_pars)

        # Return
        if return_full:
            all_pars = self.get_full_pars(sim=self.sim, calib_pars=calib_pars, genotype_pars=genotype_pars, hiv_pars=hiv_pars)
            return all_pars
        else:
            return calib_pars, genotype_pars, hiv_pars

    def sim_to_sample_pars(self):
        ''' Convert sim pars to sample pars '''

        initial_pars = sc.objdict()
        par_bounds = sc.objdict()

        # Convert regular sim pars
        if self.calib_pars is not None:
            initial_pars, par_bounds = self.update_dict_pars_init_and_bounds(initial_pars, par_bounds, self.calib_pars)

        # Convert genotype pars
        if self.genotype_pars is not None:
            initial_pars, par_bounds = self.update_dict_pars_init_and_bounds(initial_pars, par_bounds, self.genotype_pars)

       # Convert hiv pars
        if self.hiv_pars is not None:
            initial_pars, par_bounds = self.update_dict_pars_init_and_bounds(initial_pars, par_bounds, self.hiv_pars)

        return initial_pars, par_bounds

    def trial_to_sim_pars(self, pardict=None, trial=None):
        '''
        Take in an optuna trial and sample from pars, after extracting them from the structure they're provided in
        '''
        pars = sc.dcp(pardict)
        pars_flatten = sc.flattendict(pardict)
        for key, val in pars_flatten.items():
            sampler_key = '_'.join(key)
            low, high = val[1], val[2]
            step = val[3] if len(val) > 3 else None

            if key in self.par_samplers:  # If a custom sampler is used, get it now (Not working properly for now)
                try:
                    sampler_fn = getattr(trial, self.par_samplers[key])
                except Exception as E:
                    errormsg = 'The requested sampler function is not found: ensure it is a valid attribute of an Optuna Trial object'
                    raise AttributeError(errormsg) from E
            else:
                sampler_fn = trial.suggest_float
            sc.setnested(pars, list(key), sampler_fn(sampler_key, low, high, step=step))
        return pars

    def run_trial(self, trial, save=True):
        ''' Define the objective for Optuna '''
        if self.genotype_pars is not None:
            genotype_pars = self.trial_to_sim_pars(self.genotype_pars, trial)
        else:
            genotype_pars = None
        if self.hiv_pars is not None:
            hiv_pars = self.trial_to_sim_pars(self.hiv_pars, trial)
        else:
            hiv_pars = None
        if self.calib_pars is not None:
            calib_pars = self.trial_to_sim_pars(self.calib_pars, trial)
        else:
            calib_pars = None

        sim = self.run_sim(calib_pars, genotype_pars, hiv_pars, return_sim=True)

        # Compute fit for sim results and save sim results (TODO: THIS IS FOR A SINGLE TIMEPOINT. GENERALIZE THIS)
        sim_results = sc.objdict()
        for rkey in self.sim_results:
            if sim.results[rkey][:].ndim==1:
                model_output = sim.results[rkey][self.sim_results[rkey].timepoints[0]]
            else:
                model_output = sim.results[rkey][:,self.sim_results[rkey].timepoints[0]]
            diffs = self.sim_results[rkey].data.value - model_output
            gofs = hpm.compute_gof(self.sim_results[rkey].data.value, model_output)
            losses = gofs * self.sim_results[rkey].weights
            mismatch = losses.sum()
            sim.fit += mismatch
            sim_results[rkey] = model_output


        extra_sim_results = sc.objdict()
        if self.extra_sim_result_keys:
            for rkey in self.extra_sim_result_keys:
                model_output = sim.results[rkey]
                extra_sim_results[rkey] = model_output

        # Store results in temporary files (TODO: consider alternatives)
        if save:
            results = dict(sim=sim_results, analyzer=sim.get_analyzer('age_results').results,
                           extra_sim_results=extra_sim_results)
            filename = self.tmp_filename % trial.number
            sc.save(filename, results)

        return sim.fit


    def worker(self):
        ''' Run a single worker '''
        op = import_optuna()
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler = self.run_args.sampler)
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials, callbacks=None)
        return output


    def run_workers(self):
        ''' Run multiple workers in parallel '''
        if self.run_args.n_workers > 1: # Normal use case: run in parallel
            output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
        else: # Special case: just run one
            output = [self.worker()]
        return output


    def remove_db(self):
        '''
        Remove the database file if keep_db is false and the path exists.
        '''
        try:
            op = import_optuna()
            op.delete_study(study_name=self.run_args.name, storage=self.run_args.storage)
            if self.verbose:
                print(f'Deleted study {self.run_args.name} in {self.run_args.storage}')
        except Exception as E:
            print('Could not delete study, skipping...')
            print(str(E))
        if os.path.exists(self.run_args.db_name):
            os.remove(self.run_args.db_name)
            if self.verbose:
                print(f'Removed existing calibration {self.run_args.db_name}')
        return


    def make_study(self):
        ''' Make a study, deleting one if it already exists '''
        op = import_optuna()
        if not self.run_args.keep_db:
            self.remove_db()
        if self.run_args.rand_seed is not None:
            sampler = op.samplers.RandomSampler(self.run_args.rand_seed)
            sampler.reseed_rng()
            raise NotImplementedError('Implemented but does not work')
        else:
            sampler = None
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler=sampler)
        return output


    def calibrate(self, calib_pars=None, genotype_pars=None, hiv_pars=None, verbose=True, load=True, tidyup=True, **kwargs):
        '''
        Actually perform calibration.

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            verbose (bool): whether to print output from each trial
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        '''
        op = import_optuna()

        # Load and validate calibration parameters
        if calib_pars is not None:
            self.calib_pars = calib_pars
        if genotype_pars is not None:
            self.genotype_pars = genotype_pars
        if hiv_pars is not None:
            self.hiv_pars = hiv_pars
        if (self.calib_pars is None) and (self.genotype_pars is None) and (self.hiv_pars is None):
            errormsg = 'You must supply calibration parameters (calib_pars or genotype_pars) either when creating the calibration object or when calling calibrate().'
            raise ValueError(errormsg)
        self.run_args.update(kwargs) # Update optuna settings

        # Run the optimization
        t0 = sc.tic()
        self.make_study()
        self.run_workers()
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler = self.run_args.sampler)
        self.best_pars = sc.objdict(study.best_params)
        self.elapsed = sc.toc(t0, output=True)

        # Collect analyzer results
        # Load a single sim
        sim = self.sim # TODO: make sure this is OK #sc.jsonpickle(self.study.trials[0].user_attrs['jsonpickle_sim'])
        self.ng = sim['n_genotypes']
        self.glabels = [g.upper() for g in sim['genotype_map'].values()]

        # Replace with something else, this is fragile
        self.analyzer_results = []
        self.sim_results = []
        self.extra_sim_results = []
        if load:
            print('Loading saved results...')
            for trial in study.trials:
                n = trial.number
                try:
                    filename = self.tmp_filename % trial.number
                    results = sc.load(filename)
                    self.sim_results.append(results['sim'])
                    self.analyzer_results.append(results['analyzer'])
                    self.extra_sim_results.append(results['extra_sim_results'])
                    if tidyup:
                        try:
                            os.remove(filename)
                            print(f'    Removed temporary file {filename}')
                        except Exception as E:
                            errormsg = f'Could not remove {filename}: {str(E)}'
                            print(errormsg)
                    print(f'  Loaded trial {n}')
                except Exception as E:
                    errormsg = f'Warning, could not load trial {n}: {str(E)}'
                    print(errormsg)

        # Compare the results
        self.initial_pars, self.par_bounds = self.sim_to_sample_pars()
        self.parse_study(study)

        # Tidy up
        self.calibrated = True
        if not self.run_args.keep_db:
            self.remove_db()

        return self


    def parse_study(self, study):
        '''Parse the study into a data frame -- called automatically '''
        best = study.best_params
        self.best_pars = best

        print('Making results structure...')
        results = []
        n_trials = len(study.trials)
        failed_trials = []
        for trial in study.trials:
            data = {'index':trial.number, 'mismatch': trial.value}
            for key,val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    hpm.warn(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.data = data
        self.df = pd.DataFrame.from_dict(data)
        self.df = self.df.sort_values(by=['mismatch']) # Sort

        return


    def to_json(self, filename=None, indent=2, **kwargs):
        '''
        Convert the data to JSON.
        '''
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        self.json = json
        if filename:
            return sc.savejson(filename, json, indent=indent, **kwargs)
        else:
            return json


    def plot(self, res_to_plot=None, fig_args=None, axis_args=None, data_args=None, show_args=None, do_save=None,
             fig_path=None, do_show=True, plot_type='sns.boxplot', **kwargs):
        '''
        Plot the calibration results

        Args:
            res_to_plot (int): number of results to plot. if None, plot them all
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            do_save (bool): whether to save
            fig_path (str or filepath): filepath to save to
            do_show (bool): whether to show the figure
            kwargs (dict): passed to ``hpv.options.with_style()``; see that function for choices
        '''

        # Import Seaborn here since slow
        if sc.isstring(plot_type) and plot_type.startswith('sns'):
            import seaborn as sns
            if plot_type.split('.')[1]=='boxplot':
                extra_args=dict(boxprops=dict(alpha=.3), showfliers=False)
            else: extra_args = dict()
            plot_func = getattr(sns, plot_type.split('.')[1])
        else:
            plot_func = plot_type
            extra_args = dict()

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))
        show_args = sc.objdict(sc.mergedicts(dict(show=dict(tight=True, maximize=False)), show_args))
        all_args = sc.objdict(sc.mergedicts(fig_args, axis_args, d_args, show_args))

        # Pull out results to use
        analyzer_results = sc.dcp(self.analyzer_results)
        sim_results = sc.dcp(self.sim_results)

        # Get rows and columns
        if not len(analyzer_results) and not len(sim_results):
            errormsg = 'Cannot plot since no results were recorded)'
            raise ValueError(errormsg)
        else:
            all_dates = [[date for date in r.keys() if date != 'bins'] for r in analyzer_results[0].values()]
            dates_per_result = [len(date_list) for date_list in all_dates]
            other_results = len(sim_results[0].keys())
            n_plots = sum(dates_per_result) + other_results
            n_rows, n_cols = sc.get_rows_cols(n_plots)

        # Initialize
        fig, axes = pl.subplots(n_rows, n_cols, **fig_args)
        if n_plots>1:
            for ax in axes.flat[n_plots:]:
                ax.set_visible(False)
            axes = axes.flatten()
        pl.subplots_adjust(**axis_args)

        # Pull out attributes that don't vary by run
        age_labels = sc.objdict()
        for resname,resdict in zip(self.age_results_keys, analyzer_results[0].values()):
            age_labels[resname] = [str(int(resdict['bins'][i])) + '-' + str(int(resdict['bins'][i + 1])) for i in range(len(resdict['bins']) - 1)]
            age_labels[resname].append(str(int(resdict['bins'][-1])) + '+')

        # determine how many results to plot
        if res_to_plot is not None:
            index_to_plot = self.df.iloc[0:res_to_plot, 0].values
            analyzer_results = [analyzer_results[i] for i in index_to_plot]
            sim_results = [sim_results[i] for i in index_to_plot]

        # Make the figure
        with hpo.with_style(**kwargs):

            plot_count = 0
            for rn, resname in enumerate(self.age_results_keys):
                x = np.arange(len(age_labels[resname]))  # the label locations

                for date in all_dates[rn]:

                    # Initialize axis and data storage structures
                    if n_plots>1:
                        ax = axes[plot_count]
                    else:
                        ax = axes
                    bins = []
                    genotypes = []
                    values = []

                    # Pull out data
                    thisdatadf = self.target_data[rn][(self.target_data[rn].year == float(date)) & (self.target_data[rn].name == resname)]
                    unique_genotypes = thisdatadf.genotype.unique()

                    # Start making plot
                    if 'genotype' in resname:
                        for g in range(self.ng):
                            glabel = self.glabels[g].upper()
                            # Plot data
                            if glabel in unique_genotypes:
                                ydata = np.array(thisdatadf[thisdatadf.genotype == glabel].value)
                                ax.scatter(x, ydata, color=self.result_args[resname].color[g], marker='s', label=f'Data - {glabel}')

                            # Construct a dataframe with things in the most logical order for plotting
                            for run_num, run in enumerate(analyzer_results):
                                genotypes += [glabel]*len(x)
                                bins += x.tolist()
                                values += list(run[resname][date][g])

                        # Plot model
                        modeldf = pd.DataFrame({'bins':bins, 'values':values, 'genotypes':genotypes})
                        ax = plot_func(ax=ax, x='bins', y='values', hue="genotypes", data=modeldf, **extra_args)

                    else:
                        # Plot data
                        ydata = np.array(thisdatadf.value)
                        ax.scatter(x, ydata, color=self.result_args[resname].color, marker='s', label='Data')
                        # Construct a dataframe with things in the most logical order for plotting
                        for run_num, run in enumerate(analyzer_results):
                            bins += x.tolist()
                            values += list(run[resname][date])

                        # Plot model
                        modeldf = pd.DataFrame({'bins':bins, 'values':values})
                        ax = plot_func(ax=ax, x='bins', y='values', data=modeldf, color=self.result_args[resname].color, **extra_args)

                    # Set title and labels
                    ax.set_xlabel('Age group')
                    ax.set_title(f'{self.result_args[resname].name}, {date}')
                    ax.legend()
                    ax.set_xticks(x, age_labels[resname], rotation=45)
                    plot_count += 1

            for rn, resname in enumerate(self.sim_results_keys):
                if n_plots > 1:
                    ax = axes[plot_count]
                else:
                    ax = axes
                bins = sc.autolist()
                values = sc.autolist()
                thisdatadf = self.target_data[rn+sum(dates_per_result)][self.target_data[rn + sum(dates_per_result)].name == resname]
                ydata = np.array(thisdatadf.value)
                x = np.arange(len(ydata))
                ax.scatter(x, ydata, color=pl.cm.Reds(0.95), marker='s', label='Data')

                # Construct a dataframe with things in the most logical order for plotting
                for run_num, run in enumerate(sim_results):
                    bins += x.tolist()
                    if sc.isnumber(run[resname]):
                        values += sc.promotetolist(run[resname])
                    else:
                        values += run[resname].tolist()

                # Plot model
                modeldf = pd.DataFrame({'bins': bins, 'values': values})
                ax = plot_func(ax=ax, x='bins', y='values', data=modeldf, **extra_args)

                # Set title and labels
                date = thisdatadf.year[0]
                ax.set_title(self.result_args[resname].name + ', ' + str(date))
                ax.legend()
                if 'genotype_dist' in resname:
                    ax.set_xticks(x, self.glabels)
                    ax.set_xlabel('Genotype')
                plot_count += 1

        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)