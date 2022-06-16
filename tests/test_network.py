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
def test_network(do_plot=True):

    sc.heading('Testing new network structure')
    import hpvsim.sim as hps
    import hpvsim.analysis as hpa
    from hpvsim.immunity import genotype

    n_agents = 50e3
    pars = dict(pop_size=n_agents,
                start=1990,
                n_years=30,
                dt=0.5,
                pop_scale=25.2e6/n_agents,
                debut = dict(f=dict(dist='normal', par1=15., par2=2.1),
                             m=dict(dist='normal', par1=16., par2=1.8))
                )
    hpv6 = genotype('HPV6')
    hpv11 = genotype('HPV11')
    hpv16 = genotype('HPV16')
    hpv18 = genotype('HPV18')

    # Loop over countries and their population sizes in the year 2000
    age_pyr = hpa.age_pyramid(
        timepoints=['1990', '2020'],
        datafile=f'test_data/tanzania_age_pyramid.xlsx',
        edges=np.linspace(0, 100, 21))

    az = hpa.age_results(
        timepoints=['1990', '2020'],
        result_keys=['total_hpv_incidence']
    )

    sim = hps.Sim(
        pars,
        genotypes = [hpv16, hpv11, hpv6, hpv18],
        network='basic',
        location = 'tanzania',
        analyzers=[age_pyr, az])

    sim.run()
    a = sim.get_analyzer()

    # Check plot()
    if do_plot:
        fig = a.plot()
        sim.plot()

    return sim, a



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim, a    = test_network()

    sc.toc(T)
    print('Done.')
