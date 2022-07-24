'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv

do_plot = 1
do_save = 0


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

    n_agents = 50e3
    pars = dict(pop_size=n_agents, start=2000, n_years=30, dt=0.5)

    # Loop over countries and their population sizes in the year 2000
    for country,total_pop in zip(['south_africa', 'australia'], [45e6,17e6]):

        age_pyr = hpv.age_pyramid(
            timepoints=['2010', '2020'],
            datafile=f'test_data/{country}_age_pyramid.xlsx',
            edges=np.linspace(0, 100, 21))

        sim = hpv.Sim(
            pars,
            location = country.replace('_',' '),
            pop_scale=total_pop/n_agents,
            analyzers=age_pyr)

        sim.run()
        a = sim.get_analyzer()

        # Check plot()
        if do_plot:
            fig = a.plot(percentages=True)

    return sim, a


def test_age_results(do_plot=True):

    sc.heading('Testing by-age results')

    n_agents = 50e3
    pars = dict(pop_size=n_agents, pop_scale=25e6/n_agents, start=1980, n_years=40, dt=0.5, location='south africa')
    hpv16 = hpv.genotype('hpv16')
    hpv18 = hpv.genotype('hpv18')

    az1 = hpv.age_results(
        result_keys=sc.objdict(
            cancer_deaths=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0.,20.,25.,30.,40.,45.,50.,55.,65.,100.]),
            ),
            detected_cancer_deaths=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            ),
            cancer_incidence=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            ),
            detected_cancer_incidence=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0.,20.,25.,30.,40.,45.,50.,55.,65.,100.]),
            )
        )
    )

    sim = hpv.Sim(pars, genotypes=[hpv16, hpv18], analyzers=[az1])

    sim.run()
    a = sim.get_analyzer(0)

    to_plot = {
        'HPV prevalence': [
            'hpv_prevalence',
        ],
        'Cancer deaths': [
            'cancer_deaths',
        ],
        'Cervical cancer incidence': [
            'cancer_incidence',
        ],
    }

    # Check plot()
    if do_plot:
        # sim.plot(to_plot=to_plot)
        fig0 = sim.get_analyzer(0).plot()
        # fig1 = sim.get_analyzer(1).plot()

    return sim, a


def test_calibration():

    sc.heading('Testing calibration')

    pars = dict(pop_size=50e3, pop_scale=36.8e6/20e3, start=1980, end=2020, dt=0.5, location='south africa',
                init_hpv_dist=dict(
                    hpv16=0.9,
                    hpv18=0.1
                ))
    hpv16 = hpv.genotype('hpv16')
    hpv18 = hpv.genotype('hpv18')
    sim = hpv.Sim(pars, genotypes=[hpv16, hpv18])
    calib_pars = dict(
        beta=[0.05, 0.010, 0.20],
        hpv_control_prob=[.9, 0.1, 1],
    )
    genotype_pars = dict(
        hpv16=dict(
            dysp_rate=[0.2, 0.5, 1.0],
            prog_rate=[0.2, 0.5, 1.0]),
        hpv18=dict(
            dysp_rate=[0.2, 0.5, 1.0],
            prog_rate=[0.2, 0.5, 1.0]),
    )

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            datafiles=['test_data/south_africa_hpv_data.xlsx',
                                       'test_data/south_africa_cancer_data.xlsx'],
                            total_trials=10, n_workers=4)
    calib.calibrate()
    calib.plot()
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
