'''
Test calibration
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv
import numpy as np

do_plot = 1
do_save = 0
n_agents = 2e3


#%% Define the tests
def test_calibration():

    sc.heading('Testing calibration')

    pars = dict(n_agents=n_agents, start=1980, end=2020, dt=0.25, location='south africa')
    # pars['age_bins']  = np.array([ 0., 20., 25., 30., 40., 45., 50., 55., 65., 100])
    # pars['standard_pop']    = np.array([pars['age_bins'],
    #                              [.4, .08, .08, .12, .06, .06, .05, .07, .08, 0]])
    pars['init_hpv_prev'] = 0.6
    pars['age_bins']  = np.array([ 0., 20., 30., 40., 50., 60., 70., 80., 100])
    pars['standard_pop']    = np.array([pars['age_bins'],
                                 [.4, .16, .12, .12, .09, .07, .03, .01, 0]])

    sim = hpv.Sim(pars)
    calib_pars = dict(
        beta=[0.5, 0.3, 0.8],
        dur_transformed=dict(par1=[5, 3, 10]),
    )
    genotype_pars = dict(
        hpv16=dict(
            sev_fn=dict(k=[0.5, 0.2, 1.0]),
            ),
        hpv18=dict(
            sev_fn=dict(k=[0.5, 0.2, 1.0]),
        )
    )

    extra_sim_results = ['cancer_incidence', 'asr_cancer_incidence']

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            datafiles=[
                                'test_data/south_africa_hpv_data.csv',
                                'test_data/south_africa_cancer_data_2020.csv',
                            ],
                            extra_sim_results=extra_sim_results,
                            total_trials=2, n_workers=1)
    calib.calibrate(die=True)
    calib.plot(res_to_plot=4)

    # Make sure that rerunning the sims with the best pars from the calibration gives the same results
    calib_pars = calib.trial_pars_to_sim_pars(which_pars=0)
    pars = sc.mergedicts(pars,calib_pars)
    sim = hpv.Sim(pars)
    sim.run().plot()


    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim, calib = test_calibration()

    # Check sim results against stored results in calib
    best_run = calib.df.index[0]
    year = 2019
    yind = sc.findinds(sim.results['year'], year)[0]
    calib_cancer_results = calib.analyzer_results[best_run]['cancers'][2019]
    sim_cancer_results = sim.results['cancers_by_age'][:, yind]
    # np.allclose(calib_cancer_results,sim_cancer_results) # THESE SHOULD BE THE SAME -- WHY AREN'T THEY??

    # Loading the sim saved during calibration and check that the results there are the same
    calib_sim = sc.load(f'sim{best_run}.obj')
    calib_sim_cancer_results = calib_sim.results['cancers_by_age'][:, yind]
    assert np.allclose(calib_cancer_results, calib_sim_cancer_results) # THIS WORKS

    # Now copy the pars from the saved sim into a new one and rerun it
    calib_sim_pars = calib_sim.pars
    calib_sim_pars['analyzers'] = []
    rerun_sim = hpv.Sim(pars=calib_sim_pars)
    rerun_sim.run()
    rerun_sim_cancer_results = rerun_sim.results['cancers_by_age'][:, yind]
    assert np.allclose(rerun_sim_cancer_results, sim_cancer_results)  # THIS WORKS
    # assert np.allclose(rerun_sim_cancer_results, calib_sim_cancer_results)  # THIS DOESN'T



    sc.toc(T)
    print('Done.')
