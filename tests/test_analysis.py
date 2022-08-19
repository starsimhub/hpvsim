'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv

do_plot = 1
do_save = 0
n_agents = 5e3


#%% Define the tests
def test_snapshot():

    sc.heading('Testing snapshot analyzer')

    pars = dict(n_years=10, dt=0.5)

    sim = hpv.Sim(pars, analyzers=hpv.snapshot(['2016', '2019']))
    sim.run()
    snapshot = sim.get_analyzer()
    people1 = snapshot.snapshots[0]         # Option 1
    people2 = snapshot.snapshots['2016.0']  # Option 2
    people3 = snapshot.snapshots['2019.0']
    people4 = snapshot.get()                # Option 3

    assert people1 == people2, 'Snapshot options should match but do not'
    assert people3 != people4, 'Snapshot options should not match but do'
    return people4


def test_age_pyramids(do_plot=True):

    sc.heading('Testing age pyramids')

    pars = dict(n_agents=n_agents, start=2000, n_years=30, dt=0.5)

    # Loop over countries and their population sizes in the year 2000
    for country in ['south_africa', 'australia']:

        age_pyr = hpv.age_pyramid(
            timepoints=['2010', '2020'],
            datafile=f'test_data/{country}_age_pyramid.csv',
            edges=np.linspace(0, 100, 21))

        sim = hpv.Sim(
            pars,
            location = country.replace('_',' '),
            analyzers=age_pyr)

        sim.run()
        a = sim.get_analyzer()

        # Check plot()
        if do_plot:
            fig = a.plot(percentages=True)

    return sim, a


def test_age_results(do_plot=True):

    sc.heading('Testing by-age results')

    pars = dict(n_agents=n_agents, start=1970, n_years=50, dt=0.5, network='default', location='kenya')
    pars['beta'] = .5

    pars['init_hpv_prev'] = {
        'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
        'm'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
        'f'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
    }
    az1 = hpv.age_results(
        result_keys=sc.objdict(
            hpv_prevalence=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            ),
            cancer_incidence=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0.,20.,25.,30.,40.,45.,50.,55.,65.,100.]),
            ),
            cancer_mortality=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            )
        )
    )

    sim = hpv.Sim(pars, genotypes=[16, 18], analyzers=[az1])

    sim.run()
    a = sim.get_analyzer(0)

    to_plot = {
        'HPV prevalence': [
            'hpv_prevalence',
        ],
        'CIN prevalence': [
            'cin_prevalence',
        ],
        'Cervical cancer incidence': [
            'cancer_incidence',
        ],
        'Cervical cancer mortality': [
            'cancer_mortality',
        ],
    }

    # Check plot()
    if do_plot:
        sim.plot(to_plot=to_plot)
        fig0 = sim.get_analyzer(0).plot()
        # fig1 = sim.get_analyzer(1).plot()

    return sim, a


def test_calibration():

    sc.heading('Testing calibration')

    pars = dict(n_agents=n_agents, start=1980, end=2020, dt=0.5, location='south africa',
                init_hpv_dist=dict(
                    hpv16=0.9,
                    hpv18=0.1
                ))
    sim = hpv.Sim(pars, genotypes=[16,18])
    calib_pars = dict(
        beta=[0.05, 0.010, 0.20],
        hpv_control_prob=[.9, 0.1, 1],
    )
    genotype_pars = dict(
        hpv16=dict(
            dysp_rate=[0.5, 0.2, 1.0],
            prog_rate=[0.5, 0.2, 1.0],
            dur_none = dict(par1=[1.0, 0.5, 2.5])
        ),
        hpv18=dict(
            dysp_rate=[0.5, 0.2, 1.0],
            prog_rate=[0.5, 0.2, 1.0],
            dur_none=dict(par1=[1.0, 0.5, 2.5])
        )
    )

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            datafiles=[
                                'test_data/south_africa_hpv_data.csv',
                                'test_data/south_africa_cancer_data_2020.csv',
                                # 'test_data/south_africa_type_distribution_cancer.csv'
                            ],
                            total_trials=2, n_workers=1)
    calib.calibrate()
    calib.plot(top_results=4)
    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    # people      = test_snapshot()
    # sim0, a0    = test_age_pyramids()
    sim1, a1    = test_age_results()
    # sim2, calib = test_calibration()



    sc.toc(T)
    print('Done.')
