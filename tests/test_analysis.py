'''
Tests for single simulations
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc
import numpy as np

# Add module to paths and import hpvsim
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import hpvsim.sim as hps

do_plot = 1
do_save = 0


#%% Define the tests
def test_snapshot():

    sc.heading('Testing snapshot analyzer')
    import hpvsim.analysis as hpa
    import hpvsim.sim as hps

    pars = dict(n_years=10, dt=0.5)

    sim = hps.Sim(pars, analyzers=hpa.snapshot(['2016', '2019']))
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
    import hpvsim.analysis as hpa
    import hpvsim.sim as hps

    n_agents = 50e3
    pars = dict(pop_size=n_agents, start=2000, n_years=30, dt=0.5)

    # Loop over countries and their population sizes in the year 2000
    for country,total_pop in zip(['south_africa', 'australia'], [45e6,17e6]):

        age_pyr = hpa.age_pyramid(
            timepoints=['2010', '2020'],
            datafile=f'test_data/{country}_age_pyramid.xlsx',
            edges=np.linspace(0, 100, 21))

        sim = hps.Sim(
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
    import hpvsim.analysis as hpa
    import hpvsim.sim as hps
    from hpvsim.immunity import genotype

    pars = dict(pop_size=50e3, pop_scale=36.8e6/20e3, start=1990, n_years=40, dt=0.5, location='south africa')
    hpv16 = genotype('hpv16')
    hpv18 = genotype('hpv18')
    timepoints = ['2010']
    edges = np.array([0.,20.,25.,30.,40.,45.,50.,55.,65.,100.])
    az = hpa.age_results(timepoints=timepoints,
                         result_keys=['hpv_prevalence'],
                         datafile='test_data/south_africa_hpv_data.xlsx',
                         edges=edges)
    sim = hps.Sim(pars, genotypes=[hpv16, hpv18], analyzers=az)
    sim.run()
    a = sim.get_analyzer()

    # Check plot()
    if do_plot:
        fig = a.plot()

    return sim, a




#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    people      = test_snapshot()
    sim0, a0    = test_age_pyramids()
    sim1, a1    = test_age_results()

    sc.toc(T)
    print('Done.')
