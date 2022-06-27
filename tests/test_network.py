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
                n_years=40,
                burnin=10,
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
        timepoints=['2000', '2020'],
        result_keys=['total_infections']
    )

    snap = hpv.snapshot(
        timepoints=['1990', '2000', '2010', '2020'],
    )

    sim = hpv.Sim(
        pars,
        genotypes = [hpv16, hpv11, hpv6, hpv18],
        network='default',
        location = 'tanzania',
        analyzers=[age_pyr, az, snap])

    sim.run()
    a = sim.get_analyzer(1)

    # Check plot()
    if do_plot:
        fig = a.plot()
        sim.plot()

        snapshot = sim.get_analyzer()
        people1990 = snapshot.snapshots[0]
        people2000 = snapshot.snapshots[1]
        people2010 = snapshot.snapshots[2]
        people2020 = snapshot.snapshots[3]

        # Plot age mixing
        import pylab as pl
        pl.rcParams.update({'font.size': 14})
        for lkey in ['m','c']:
            fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(12, 8))
            ax = axes.flatten()
            for ai,people in enumerate([people1990, people2000, people2010, people2020]):
                h = ax[ai].hist2d(people.contacts[lkey]['age_f'], people.contacts[lkey]['age_m'], bins=np.linspace(0, 100, 21))
                ax[ai].set_xlabel('Age of female partner')
                ax[ai].set_ylabel('Age of male partner')
                fig.colorbar(h[3], ax=ax[ai])
                ax[ai].set_title(snapshot.dates[ai])

        pl.show()


    return sim, a



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim, a    = test_network()

    sc.toc(T)
    print('Done.')
