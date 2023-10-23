'''
Tests for network options (beyond-age assortativity and diagnostic visualizations)
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.linalg

#%% Network analyzer

class new_pairs_snap(hpv.Analyzer):
    # analyzer for recording new partnerships of each timestep
    def __init__(self, start_year=None, year_mod=None, **kwargs):
        super().__init__(**kwargs)
        self.new_pairs = pd.DataFrame()
        self.start_year = start_year
        self.year_mod = year_mod
        self.yearvec = None

    def initialize(self, sim):
        super().initialize()
        self.yearvec = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']

    def apply(self, sim):
        if sim.yearvec[sim.t] < self.start_year:
            return

        if self.year_mod is None or sim.yearvec[sim.t] % self.year_mod == 0:
            year_since_start = sim.yearvec[sim.t] - sim['start']
            layer_keys = sim.people.layer_keys()
            for rtype in layer_keys:
                new_rship_inds = (sim.people.contacts[rtype]['start'] == year_since_start).nonzero()[0]
                if len(new_rship_inds):
                    contacts = pd.DataFrame.from_dict(sim.people.contacts[rtype].get_inds(new_rship_inds))
                    contacts['year'] = int(sim.yearvec[sim.t])
                    contacts['rtype'] = rtype
                    self.new_pairs = pd.concat([self.new_pairs, contacts])
        return

def network_demo():
    clusters = [5, 5]
    mixing_mats = [np.ones((5,5)), scipy.linalg.circulant([1,0.5,0.1,0.1,0.5])]
    start = 1970
    end = 2020
    pop = 2e4
    labels = ['status quo', 'clustered']

    sims = []
    snap = hpv.snapshot(
        timepoints=['1970', '1980', '1990', '2000', '2010', '2020'],
    )
    new_pairs = new_pairs_snap(start_year=start, year_mod=10)
    for n_clusters, mixing, label in zip(clusters, mixing_mats, labels):
        pars = dict(
            n_agents=pop,
            start=start,
            end=end,
            location='nigeria',
            ms_agent_ratio=100,
            n_clusters=n_clusters,
            add_mixing=mixing,
            analyzers=[snap, new_pairs]
        )
        sim = hpv.Sim(pars=pars, label=label)
        sims.append(sim)
    msim = hpv.MultiSim(sims)
    msim.run()
    msim.plot()

    for sim in msim.sims:
        # plot age and cluster mixing by year
        plot_mixing(sim, 'age')
        plot_mixing(sim, 'cluster')
        # plot number of relationships over time
        plot_rships(sim)

def plot_mixing(sim, dim):
    df_new_pairs = sim.get_analyzer('new_pairs_snap').new_pairs
    if dim == 'age':
        bins = np.linspace(0, 75, 16, dtype=int)
        bins = np.append(bins, 100)
        df_new_pairs['x_bins'] = pd.cut(df_new_pairs['age_f'], bins, right=False)
        df_new_pairs['y_bins'] = pd.cut(df_new_pairs['age_m'], bins, right=False)
    elif dim == 'cluster':
        df_new_pairs['x_bins'] = df_new_pairs['cluster_f'].astype('category')
        df_new_pairs['y_bins'] = df_new_pairs['cluster_m'].astype('category')

    count_df = df_new_pairs.groupby(['rtype', 'year', 'x_bins', 'y_bins']).size().reset_index(name='count')
    def facet(data, **kwargs):
        data = data.pivot(index='y_bins', columns='x_bins', values='count')
        ax = sns.heatmap(data, **kwargs)
        ax.invert_yaxis()
    g = sns.FacetGrid(count_df, col='year', row='rtype', height=4)
    g.map_dataframe(facet, cmap='viridis', cbar=True, square=True)
    g.set_axis_labels(f'{dim} of female partners', f'{dim} of male partners')
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(sim.label)
    g.tight_layout()

def plot_rships(sim):
    layer_keys = list(sim['partners'].keys())
    snaps = sim.get_analyzer('snapshot')

    dfs = []
    years = []
    for year, people in snaps.snapshots.items():
        df = pd.DataFrame({'age':people.age[people.alive==True], 'sex':people.is_female[people.alive==True]})
        df['sex'].replace({True:'Female', False:'Male'}, inplace=True)
        df['Age Bin'] = pd.cut(df['age'], people.age_bin_edges, right=False)
        df['Year'] = year
        for lk, lkey in enumerate(layer_keys):
            df[lkey] = people.n_rships[lk, people.alive==True]
        dfs.append(df)
        years.append(year)
    
    df = pd.concat(dfs)
    dfm = df.melt(id_vars=['Age Bin', 'sex', 'Year'], value_vars=layer_keys, var_name='Layer', value_name='n_rships')
    g = sns.catplot(data=dfm, kind='bar', x='Age Bin', y='n_rships', hue='sex', col='Year', row='Layer', sharey=False, height=5, aspect=0.75, legend_out=False, palette='tab10')
    g.tick_params(axis='x', which='both', rotation=70)
    g.set_ylabels('Number of Relationships')
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(sim.label)

    dfm_fem = dfm[(dfm['sex'] == 'Female') & (dfm['Year'] == years[-1])]
    h = sns.catplot(data=dfm_fem, kind='box', x='Age Bin', y='n_rships', col='Layer', sharey=False,
                    height=5, aspect=0.75, legend_out=False)
    h.tick_params(axis='x', which='both', rotation=70)
    h.set_ylabels('Number of Relationships')
    h.fig.tight_layout()
    h.fig.subplots_adjust(top=0.9)
    h.fig.suptitle(sim.label)

#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()
    network_demo()
    sc.toc(T)
    plt.show()
    print('Done.')