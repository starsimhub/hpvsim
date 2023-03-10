'''
Test calibration
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv

do_plot = 1
do_save = 0
n_agents = 2e3


#%% Define the tests
def test_calibration():

    sc.heading('Testing calibration')

    pars = dict(n_agents=n_agents, start=1980, end=2020, dt=0.25, location='south africa')
    sim = hpv.Sim(pars)
    calib_pars = dict(
        beta=[0.05, 0.010, 0.20],
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
                            total_trials=3, n_workers=1)
    calib.calibrate(die=True)
    calib.plot(res_to_plot=4)
    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim2, calib = test_calibration()

    sc.toc(T)
    print('Done.')
