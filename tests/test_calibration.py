'''
Test calibration
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv
import numpy as np
import pylab as pl

do_plot = 1
do_save = 0
n_agents = 2e3


#%% Define the tests

def estimator(actual, predicted):
    ''' Custom estimator to use for bounded target data'''
    actuals = []
    for i in actual:
        if isinstance(i, (int, float, np.generic)):
            i_list = [i,i]
        else:
            i_list = [idx for idx in i.split(',')]
            i_list[0] = float(i_list[0].replace('[', ''))
            i_list[1] = float(i_list[1].replace(']', ''))
        actuals.append(i_list)
    gofs = np.zeros(len(predicted))
    for iv, val in enumerate(predicted):
        if val> np.max(actuals[iv]):
            gofs[iv] = abs(np.max(actuals[iv])-val)
        elif val < np.min(actuals[iv]):
            gofs[iv] = abs(np.min(actuals[iv])-val)
    actual_max = np.array(actuals).max()
    if actual_max > 0:
        gofs /= actual_max

    gofs = np.mean(gofs)

    return gofs

def test_calibration(do_plot=True):

    sc.heading('Testing calibration')

    pars = dict(n_agents=n_agents, start=1980, end=2020, dt=0.25, location='south africa')
    pars['init_hpv_prev'] = 0.6 # Set high init_prev to generate more cancers

    # Change the sim age bins so they're the same as the analyzer age bins
    age_bin_edges = np.array([ 0., 20., 30., 40., 50., 60., 70., 80., 100])
    pars['age_bin_edges'] = age_bin_edges
    pars['standard_pop']  = np.array([age_bin_edges, [.4, .16, .12, .12, .09, .07, .03, .01, 0]])
    pars['standard_pop']  = np.array([age_bin_edges, [.4, .16, .12, .12, .09, .07, .03, .01, 0]])
    az = hpv.age_results(
        result_args=sc.objdict(
            cancers=sc.objdict(
                years=2020,
                edges=age_bin_edges,
            )
        )
    )

    # Save a snapshot so we can compare people later if needed
    sim = hpv.Sim(pars, analyzers=[hpv.snapshot(timepoints=['1980']), az])

    # Define the calibration parameters
    calib_pars = dict(
        beta=[0.25, 0.10, 0.30],

    )
    genotype_pars = dict(
        hpv16=dict(
            cin_fn=dict(k=[0.5, 0.2, 1.0]),
            ),
        hpv18=dict(
            cin_fn=dict(k=[0.5, 0.2, 1.0]),
        )
    )

    # Save some extra sim results
    extra_sim_result_keys = ['cancer_incidence', 'asr_cancer_incidence']
    
    # Make the calibration
    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            datafiles=[
                                'test_data/south_africa_hpv_data.csv',
                                'test_data/south_africa_cancer_data_2020.csv',
                            ], estimator = estimator,
                            extra_sim_result_keys=extra_sim_result_keys,
                            total_trials=2, n_workers=1, die=True)
    calib.calibrate()
    calib.plot(res_to_plot=4)

    # Make sure that rerunning the sims with the best pars from the calibration gives the same results
    calib_pars = calib.trial_pars_to_sim_pars(which_pars=0)
    pars = sc.mergedicts(pars,calib_pars)
    sim = hpv.Sim(pars, analyzers=[hpv.snapshot(timepoints=['1980']), az])
    sim.run()

    # Check sim results against stored results in calib
    best_run = calib.df.index[0] # Pull out the index of the best run
    year = 2020
    yind = sc.findinds(sim.results['year'], year)[0]
    calib_cancer_results = calib.analyzer_results[best_run]['cancers'][2020] # Pull out the analyzer from the best run
    sim_cancer_results = sim.results['cancers_by_age'][:, yind] # Pull out the sim results from the sim run with the best pars
    az_cancer_results = sim.get_analyzer('age_results').results['cancers'][2020] # Pull out the analyzer results from the sim run with the best pars

    # Do plots for visual inspection
    if do_plot:
        x = calib.analyzer_results[best_run]['cancers']['bins']

        fig, axes = pl.subplots(1,2)
        axes[0].plot(x, sim_cancer_results, label='sim results')
        axes[0].plot(x, calib_cancer_results, label='calib results')
        axes[0].plot(x, az_cancer_results, label='analyzer results')
        axes[0].set_title('Cancers by age')
        axes[0].set_xlabel('Age')
        axes[0].legend()

        sim_asr = sim.results['asr_cancer_incidence'].values
        calib_asr = calib.extra_sim_results[best_run]['asr_cancer_incidence'].values
        x = sim.results['year']

        axes[1].plot(x, sim_asr, label='sim results')
        axes[1].plot(x, calib_asr, label='calib results')
        axes[1].set_title('ASR cancer incidence')
        axes[1].set_xlabel('Year')
        fig.tight_layout()
        fig.show()

    # In addition to the plots, assert that they must be equal
    assert np.allclose(calib_cancer_results,sim_cancer_results, rtol=0.1)

    return sim, calib




#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim, calib = test_calibration()

    sc.toc(T)
    print('Done.')
# %%
