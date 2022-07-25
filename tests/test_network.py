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
                start=1970,
                n_years=60,
                burnin=30,
                dt=0.2,
                network='default',
                debut = dict(f=dict(dist='normal', par1=15., par2=1),
                             m=dict(dist='normal', par1=16., par2=1))
                )
    # hpv6    = hpv.genotype('HPV6')
    # hpv11   = hpv.genotype('HPV11')
    hpv16   = hpv.genotype('HPV16')
    hpv18   = hpv.genotype('HPV18')

    # Loop over countries and their population sizes in the year 2000
    age_pyr = hpv.age_pyramid(
        timepoints=['2020'],
        datafile=f'test_data/kenya_age_pyramid.csv',
        edges=np.linspace(0, 100, 21))

    az = hpv.age_results(
        result_keys=sc.objdict(
            total_infections=sc.objdict(
                timepoints=['2000', '2020'],
            ),
        )
    )

    snap = hpv.snapshot(
        timepoints=['1990', '2000', '2010', '2020'],
    )

    sim = hpv.Sim(
        pars,
        genotypes = [hpv16, hpv18],
        network='default',
        location = 'kenya',
        datafile=f'test_data/kenya_data.csv',
        analyzers=[age_pyr, az, snap])

    sim.run()

    # Check plot()
    if do_plot:
        snapshot = sim.get_analyzer()
        people1990 = snapshot.snapshots[0]
        people2000 = snapshot.snapshots[1]
        people2010 = snapshot.snapshots[2]
        people2020 = snapshot.snapshots[3]

        # Plot age mixing
        import pylab as pl
        import matplotlib as mpl
        snapshot = sim.get_analyzer()
        people2020 = snapshot.snapshots[3]
        font_size = 15
        font_family = 'Libertinus Sans'
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = font_family
        fig, ax = pl.subplots(nrows=1, ncols=1, figsize=(5, 4))
        # ax = axes.flatten()
        people = people2020
        lkey='m'
        # for ai,lkey in enumerate(['m','c']):
        fc = people.contacts[lkey]['age_f']
        mc = people.contacts[lkey]['age_m']
        h = ax.hist2d(fc, mc, bins=np.linspace(0, 75, 16), density=True, norm=mpl.colors.LogNorm())
        ax.set_xlabel('Age of female partner')
        ax.set_ylabel('Age of male partner')
        fig.colorbar(h[3], ax=ax)
        ax.set_title('Marital age mixing')
        fig.tight_layout()
        pl.savefig(f"networks.png", dpi=100)

        import pylab as pl
        fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(8, 2))
        ax = axes.flatten()
        types = ['casual', 'one-off']
        xx = people.lag_bins[1:15]*sim['dt']
        for cn,lkey in enumerate(['c','o']):
            yy = people.rship_lags[lkey][:14]/sum(people.rship_lags[lkey])
            ax[cn].bar(xx,yy, width=0.2)
            ax[cn].set_xlabel(f'Time between {types[cn]} relationships')
        fig.tight_layout()
        pl.savefig(f"lags.png", dpi=100)


    return sim, a



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim, a    = test_network()

    sc.toc(T)
    print('Done.')
