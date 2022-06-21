'''
Tests for single simulations
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc
import numpy as np
import hpvsim as hpv


do_plot = 1
do_save = 0


#%% Define the tests
def test_network(do_plot=True):

    sc.heading('Testing new network structure')

    n_agents = 50e3
    pars = dict(pop_size=n_agents,
                start=1990,
                n_years=30,
                dt=0.5,
                pop_scale=25.2e6/n_agents,
                network='default',
                debut = dict(f=dict(dist='normal', par1=15., par2=1),
                             m=dict(dist='normal', par1=16., par2=1))
                )
    hpv6    = hpv.genotype('HPV6')
    hpv11   = hpv.genotype('HPV11')
    hpv16   = hpv.genotype('HPV16')
    hpv18   = hpv.genotype('HPV18')

    # Loop over countries and their population sizes in the year 2000
    age_pyr = hpv.age_pyramid(
        timepoints=['1990', '2020'],
        datafile=f'test_data/tanzania_age_pyramid.xlsx',
        edges=np.linspace(0, 100, 21))

    az = hpv.age_results(
        timepoints=['1990', '2000'],
        result_keys=['total_infections', 'total_hpv_incidence']
    )

    # TO PLOT:
    # 1. Heat map of discordant partners
    # m on one axis, f on other, colour by number of HPV discordant partnerships
    # 2. Individual lifetime
    # 3. Plot gaps between relationships


    sim = hpv.Sim(
        pars,
        genotypes = [hpv16, hpv11, hpv6, hpv18],
        network='default',
        location = 'tanzania',
        analyzers=[age_pyr, az])

    sim.run()
    a = sim.get_analyzer()

    # Check plot()
    if do_plot:
        fig = a.plot()
        sim.plot()

        # Plot age mixing
        import pylab as pl
        pl.rcParams.update({'font.size': 22})
        fig, ax = pl.subplots(figsize=(12, 8))
        h = ax.hist2d(sim.people.contacts['m']['age_f'], sim.people.contacts['m']['age_m'],
                      bins=np.linspace(0, 100, 21))
        ax.set_xlabel('Age of female partner')
        ax.set_ylabel('Age of male partner')
        fig.colorbar(h[3], ax=ax)
        pl.show()

    return sim, a



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim, a    = test_network()

    sc.toc(T)
    print('Done.')
