'''
Tests for network options (geostructure and diagnostic visualizations)
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
import matplotlib as mpl
import pylab as pl
import pandas as pd

do_plot = 1
do_save = 0
base_pars = {
    'n_agents': 10e3,
    'start': 1970,
    'end': 2020,
    'location': 'nigeria'
}

#%% Network analyzer
# TODO: move this to analysis.py?
class new_pairs_snap(hpv.Analyzer):
    def __init__(self, start_year=None, by_year=3, **kwargs):
        super().__init__(**kwargs)
        self.new_pairs = pd.DataFrame(columns = ['f', 'm', 'acts', 'dur', 'start', 'end', 'age_f', 'age_m', 'year', 'rtype'])
        self.start_year = start_year
        self.yearvec = None
        self.by_year = by_year

    def initialize(self, sim):
        super().initialize()
        self.yearvec = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            tind = sim.yearvec[sim.t] - sim['start']
            for rtype in ['m','c','o']:
                new_rship_inds = (sim.people.contacts[rtype]['start'] == tind).nonzero()[0]
                if len(new_rship_inds):
                    contacts = pd.DataFrame.from_dict(sim.people.contacts[rtype].get_inds(new_rship_inds))
                    #contacts = pd.DataFrame.from_dict(sim.people.contacts[rtype])
                    contacts['year'] = int(sim.yearvec[sim.t])
                    contacts['rtype'] = rtype
                    self.new_pairs = pd.concat([self.new_pairs, contacts])
        return


    def plot(self, do_save=False, filename=None, ag=False):
        n_time = len(self.new_pairs[0])
        check_square = n_time % np.sqrt(n_time)
        non_square = 1
        if check_square == 0:
            nrows = int(np.sqrt(n_time))
            ncols = int(np.sqrt(n_time))
        else:
            nrows = int(np.sqrt(n_time)) + non_square
            ncols = int(np.sqrt(n_time))


        fig, ax = pl.subplots(nrows, ncols, figsize=(15, 8))
        yi = sc.findinds(self.yearvec, from_when)[0]
        for rn, rtype in enumerate(['m', 'c', 'o']):
            ax[0, rn].plot(self.yearvec[yi:], self.n_edges[rtype][yi:])
            ax[0, rn].set_title(f'Edges - {rtype}')
            ax[1, rn].plot(self.yearvec[yi:], self.n_edges_norm[rtype][yi:])
            ax[1, rn].set_title(f'Normalized edges - {rtype}')
        pl.tight_layout()
        if do_save:
            fn = 'networks' or filename
            fig.savefig(f'{filename}.png')
        else:
            pl.show()

def run_network(geos, geo_mix, start, end, pop):

    #labels = ['Clustered network', 'Status quo']
    labels = []
    snap = hpv.snapshot(
        timepoints=['1990', '2000', '2010', '2020'],
    )
    snaps = []
    new_pairs = new_pairs_snap(start_year = 2010)
    df_new_pairs = pd.DataFrame(columns = ['f', 'm', 'acts', 'dur', 'start', 'end', 'age_f', 'age_m', 'year', 'rtype', 'sim'])
    fig0, axes = pl.subplots(2, 1)
    for i, (geostruct, geo_mixing) in enumerate(zip(geos, geo_mix)):
        print(i)
        pars = dict(
            n_agents=pop,
            start=start,
            end=end,
            location='nigeria',
            ms_agent_ratio=100,
            geostructure=geostruct,
            #clustered_risk=risk,
            #geo_mixing=geo_mixing,
            geo_mixing_steps = geo_mixing,
            #random_pairing=True,
            analyzers=[snap, new_pairs]
        )

        sim = hpv.Sim(pars=pars)
        sim.run()
        # Plot age mixing
        labels += ['{} geo-cluster, {} mixing steps'.format(geostruct, len(geo_mixing))]
        snaps.append(sim.get_analyzer([0]))
        new_pairs_snaps = sim.get_analyzer([1]).new_pairs
        new_pairs_snaps['sim'] = i
        df_new_pairs = pd.concat([df_new_pairs, new_pairs_snaps])
        ## Network diagnostics
        plot_mixing(sim, df_new_pairs)

        axes[0].plot(sim.results['year'], sim.results['infections'], label=labels[i])
        axes[1].plot(sim.results['year'], sim.results['cancers'])


    axes[0].legend()
    axes[0].set_ylabel('Infections')
    axes[1].set_ylabel('Cancers')
    fig0.show()

    fig, axes = pl.subplots(nrows=i+1, ncols=3, figsize=(14, 10), sharey='col')
    for i, isnap in enumerate(snaps):
        people2020 = isnap.snapshots[3]
        font_size = 15
        font_family = 'Libertinus Sans'
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = font_family

        # ax = axes.flatten()
        people = people2020
        rships_f = np.zeros((3, len(people.age_bin_edges)))
        rships_m = np.zeros((3, len(people.age_bin_edges)))
        for lk, lkey in enumerate(['m', 'c', 'o']):
            active_ages = people.age#[(people.n_rships[lk,:] >= 1)]
            n_rships = people.n_rships#[:,(people.n_rships[lk,:] >= 1)]
            age_bins = np.digitize(active_ages, bins=people.age_bin_edges) - 1


            for ab in np.unique(age_bins):
                inds_f = (age_bins==ab) & people.is_female
                inds_m = (age_bins==ab) & people.is_male
                rships_f[lk,ab] = n_rships[lk,inds_f].sum()/len(hpv.true(inds_f))
                rships_m[lk, ab] = n_rships[lk, inds_m].sum() / len(hpv.true(inds_m))
            ax = axes[i, lk]
            yy_f = rships_f[lk,:]
            yy_m = rships_m[lk,:]
            ax.bar(people.age_bin_edges-1, yy_f, width=1.5, label='Female')
            ax.bar(people.age_bin_edges+1, yy_m, width=1.5, label='Male')
            ax.set_xlabel(f'Age')
            ax.set_title(f'Average number of relationships, {lkey}')
        axes[i, 0].set_ylabel(labels[i])
    axes[0,2].legend()
    fig.tight_layout()
    fig.show()

    fig, axes = pl.subplots(nrows=i+1, ncols=3, figsize=(14, 10), sharey='col')
    for i, isnap in enumerate(snaps):
        people2020 = isnap.snapshots[3]
        font_size = 15
        font_family = 'Libertinus Sans'
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = font_family

        # ax = axes.flatten()
        people = people2020

        types = ['marital', 'casual', 'one-off']
        xx = people.lag_bins[1:15] * sim['dt']
        for cn, lkey in enumerate(['m', 'c', 'o']):
            ax = axes[i,cn]
            yy = people.rship_lags[lkey][:14] / sum(people.rship_lags[lkey])
            ax.bar(xx, yy, width=0.2)
            ax.set_xlabel(f'Time between {types[cn]} relationships')
        axes[i,0].set_ylabel(labels[i])

    fig.tight_layout()
    fig.show()


#%% Define the tests


def plot_mixing(sim, df_new_pairs):
    for runind in df_new_pairs.sim.unique():
        for i, rtype in enumerate(['m','c','o']):
            df = df_new_pairs[(df_new_pairs['sim'] == runind) & (df_new_pairs['rtype'] == rtype)]
            n_time = len(df.year.unique())
            check_square = n_time % np.sqrt(n_time)
            non_square = 1
            if check_square == 0:
                nr = int(np.sqrt(n_time))
                nc = int(np.sqrt(n_time))
            else:
                nr = int(np.sqrt(n_time)) + non_square
                nc = int(np.sqrt(n_time))
                if nr * nc < n_time:
                    nc += 1
            fig, ax = pl.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, figsize=(15, 12))
            for j, year in enumerate(df_new_pairs.year.unique()):
                df_year = df[df['year']==year]
                fc = df_year.age_f  # Get the age of female contacts in marital partnership
                mc = df_year.age_m  # Get the age of male contacts in marital partnership
                h = ax[j//nc, j%nc].hist2d(fc, mc, bins=np.linspace(0, 75, 16), density=False, norm=mpl.colors.LogNorm())
                ax[j//nc, j%nc].set_title(year)

            fig.colorbar(h[3], ax=ax)
            mixing = sim['mixing'][rtype]
            age_bins = mixing[:,0]
            mixing = mixing[:,1:]
            mixing_norm_col = mixing / mixing.max(axis=0)
            mixing_norm_col[np.isnan(mixing_norm_col)] = 0
            X, Y = np.meshgrid(age_bins, age_bins)
            h = ax[nr-1, nc-1].pcolormesh(X, Y, mixing_norm_col, norm=mpl.colors.LogNorm())
            ax[nr-1, nc-1].set_title('Input')

            fig.text(0.5, 0.04, 'Age of female partner', ha='center', fontsize=24)
            fig.text(0.04, 0.5, 'Age of male partner', va='center', rotation='vertical', fontsize=24)

            fig.suptitle(rtype, fontsize=24)
            fig.tight_layout(h_pad=0.5)
            fig.subplots_adjust(top=0.9, left=0.1, bottom=0.1, right=0.75)
            fig.show()


def geo_demo():
    sc.heading('Geostructure test')

    sim0 = hpv.Sim(pars=base_pars)
    sim0['geostructure'] = 10
    sim0.update_pars()
    sim0.run()
    # Default: well-mixed (1 geo cluster)
    assert sim0['geostructure'] == 1
    # Multiple geo clusters
    pars1 = base_pars
    pars1['geostructure'] = 10
    pars1['geo_mixing_steps'] = np.repeat(1,9) # TODO: automatically adjust geo_mixing_steps as well-mixed when given geostructure input > 1?
    sim1 = hpv.Sim(pars=pars1)
    print(sim1['geomixing'])
    # Modifying mixing steps
    pars2 = pars1
    pars2['geo_mixing_steps'] = [0.5, 0.01] # diagonal is 1 by default, set relative mixing at 0.5 for adjacent clusters, 0.01 for clusters with distance = 2
    sim2 = hpv.Sim(pars=pars2)
    print(sim2['geomixing'])


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()
    geos = [1, 10]
    geo_mix = [[1], np.repeat(1,9)]
    start = 1970
    end = 2020
    pop = 10e3
    run_network(geos, geo_mix, start, end, pop)
    geo_demo()

    sc.toc(T)
    print('Done.')