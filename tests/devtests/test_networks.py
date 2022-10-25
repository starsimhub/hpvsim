'''
Plot network outputs
''' 
import os
import pytest
import sys
import sciris as sc
import numpy as np
import hpvsim as hpv
import matplotlib.pyplot as plt


do_plot = 1
do_save = 0


# %% Define the tests
def test_network(do_plot=True):
    sc.heading('Testing network')

    pars = dict(n_agents=50e3,
                start=1975,
                n_years=50,
                burnin=25,
                dt=0.5,
                network='default',
                location='kenya',
                genotypes=[16,18]
                )

    snap = hpv.snapshot(
        timepoints=['1990', '2000', '2010', '2020'],
    )

    sim = hpv.Sim(
        pars,
        analyzers=[snap]
    )

    sim.run()

    # Check plot()
    if do_plot:

        a = sim.get_analyzer()
        people1990 = a.snapshots[0]
        people2000 = a.snapshots[1]
        people2010 = a.snapshots[2]
        people2020 = a.snapshots[3]

        # Plot age mixing
        import matplotlib as mpl
        import pylab as pl
        snapshot = sim.get_analyzer()
        people2020 = snapshot.snapshots[3]
        font_size = 15
        font_family = 'Libertinus Sans'
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = font_family
        fig, ax = pl.subplots(nrows=1, ncols=1, figsize=(5, 4))
        # ax = axes.flatten()
        people = people2020
        lkey = 'm'
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

        fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(8, 2))
        ax = axes.flatten()
        types = ['casual', 'one-off']
        xx = people.lag_bins[1:15] * sim['dt']
        for cn, lkey in enumerate(['c', 'o']):
            yy = people.rship_lags[lkey][:14] / sum(people.rship_lags[lkey])
            ax[cn].bar(xx, yy, width=0.2)
            ax[cn].set_xlabel(f'Time between {types[cn]} relationships')
        fig.tight_layout()
        pl.savefig(f"lags.png", dpi=100)

    return sim, a



class rship_count(hpv.Analyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_edges = dict()
        self.n_edges_norm = dict()
        for rtype in ['m','c','o']:
            self.n_edges[rtype] = []
            self.n_edges_norm[rtype] = []

    def initialize(self, sim):
        super().initialize()
        self.yearvec = sim.yearvec

    def apply(self, sim):
        for rtype in ['m','c','o']:
            self.n_edges[rtype].append(len(sim.people.contacts[rtype]))
            age = sim.people.age[sim.people.is_active]
            denom = ((age>14) * age<65).sum()
            self.n_edges_norm[rtype].append(len(sim.people.contacts[rtype])/denom)
        return

    def plot(self, do_save=False, filename=None, from_when=1990):
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        yi = sc.findinds(self.yearvec, from_when)[0]
        for rn,rtype in enumerate(['m','c','o']):
            ax[0,rn].plot(self.yearvec[yi:], self.n_edges[rtype][yi:])
            ax[0,rn].set_title(f'Edges - {rtype}')
            ax[1,rn].plot(self.yearvec[yi:], self.n_edges_norm[rtype][yi:])
            ax[1,rn].set_title(f'Normalized edges - {rtype}')
        plt.tight_layout()
        if do_save:
            fn = 'networks' or filename
            fig.savefig(f'{filename}.png')
        else:
            plt.show()

def test_network_time(do_plot=do_plot):
    sc.heading('Testing numbers of partners over time')

    pars = dict(n_agents=5e3,
                start=1950,
                end=2050,
                dt=1.0,
                location='india',
                genotypes=[16,18,'hrhpv'],
                )

    sim = hpv.Sim(pars=pars, analyzers=rship_count())
    sim.run()

    a = sim.get_analyzer()
    fig = a.plot()

    return sim, a



# %% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    sim0, a0 = test_network(do_plot=do_plot)
    sim1, a1 = test_network_time(do_plot=do_plot)
    ssim1, a1 = test_network_time(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
