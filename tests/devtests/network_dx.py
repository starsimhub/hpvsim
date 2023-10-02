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

hpv.options(verbose=False)

base_pars = {
    'n_agents': 2e4,
    'start': 1970,
    'end': 2020,
    'location': 'nigeria'
}

#%% Network analyzer

class new_pairs_snap(hpv.Analyzer):
    # analyzer for recording new partnerships of each timestep
    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.new_pairs = pd.DataFrame(columns = ['f', 'm', 'acts', 'dur', 'start', 'end', 'age_f', 'age_m', 'year', 'rtype'])
        self.start_year = start_year
        self.yearvec = None

    def initialize(self, sim):
        super().initialize()
        self.yearvec = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            tind = sim.yearvec[sim.t] - sim['start']
            layer_keys = sim.people.layer_keys()
            for rtype in layer_keys:
                new_rship_inds = (sim.people.contacts[rtype]['start'] == tind).nonzero()[0]
                if len(new_rship_inds):
                    contacts = pd.DataFrame.from_dict(sim.people.contacts[rtype].get_inds(new_rship_inds))
                    #contacts = pd.DataFrame.from_dict(sim.people.contacts[rtype])
                    contacts['year'] = int(sim.yearvec[sim.t])
                    contacts['rtype'] = rtype
                    self.new_pairs = pd.concat([self.new_pairs, contacts])
        return

def cluster_demo():
    sc.heading('Cluster test')
    # Default: well-mixed (1 cluster)
    sim0 = hpv.Sim(pars=base_pars)
    assert sim0['n_clusters'] == 1
    # Multiple clusters
    pars1 = base_pars
    pars1['n_clusters'] = 10
    pars1['mixing_steps'] = np.repeat(1,9)
    sim1 = hpv.Sim(pars=pars1)
    print(sim1['add_mixing'])
    # Modifying mixing steps
    pars2 = pars1
    pars2['mixing_steps'] = [0.5, 0.01] # diagonal is 1 by default, set relative mixing at 0.5 for adjacent clusters, 0.01 for clusters with distance = 2
    sim2 = hpv.Sim(pars=pars2)
    print(sim2['add_mixing'])


def network_demo():
    clusters = [10, 10]
    mixing_steps = [np.ones(9), [0.9,0.5,0.1]]
    start = 1970
    end = 2020
    pop = 2e4
    labels = ['status quo', 'clustered']

    sims = []
    snap = hpv.snapshot(
        timepoints=['1990', '2000', '2010', '2020'],
    )
    new_pairs = new_pairs_snap(start_year = 2017)
    for n_clusters, mixing, label in zip(clusters, mixing_steps, labels):
        pars = dict(
            n_agents=pop,
            start=start,
            end=end,
            location='nigeria',
            ms_agent_ratio=100,
            n_clusters=n_clusters,
            mixing_steps=mixing,
            analyzers=[snap, new_pairs]
        )
        sim = hpv.Sim(pars=pars, label=label)
        sims.append(sim)
    msim = hpv.MultiSim(sims)
    msim.run()
    msim.plot(style='simple')

    for sim in msim.sims:
        # plot age and cluster mixing by year
        plot_mixing(sim, 'age')
        plot_mixing(sim, 'cluster')
        # plot number of relationships overtime
        plot_rships(sim)

def plot_mixing(sim, dim):
    df_new_pairs = sim.get_analyzer('new_pairs_snap').new_pairs
    if dim == 'age':
        bins = np.linspace(0, 75, 16, dtype=int)
        bins = np.append(bins, 100)
        df_new_pairs['x_bins'] = pd.cut(df_new_pairs['age_f'], bins)
        df_new_pairs['y_bins'] = pd.cut(df_new_pairs['age_m'], bins)
    elif dim == 'cluster':
        df_new_pairs['x_bins'] = df_new_pairs['cluster_f']
        df_new_pairs['y_bins'] = df_new_pairs['cluster_m']

    count_df = df_new_pairs.groupby(['rtype', 'year', 'x_bins', 'y_bins']).size().reset_index(name='count')
    def facet(data, **kwargs):
        data = data.pivot(index='x_bins', columns='y_bins', values='count')
        ax = sns.heatmap(data, **kwargs)
        ax.invert_yaxis()
    g = sns.FacetGrid(count_df, col='year', row='rtype', height=4)
    g.map_dataframe(facet, cmap='viridis', cbar=True, square=True)
    g.set_axis_labels(f'{dim} of female partners', f'{dim} of male partners')
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(sim.label)
    g.tight_layout()
    plt.show()

def plot_rships(sim):
    layer_keys = list(sim['partners'].keys())
    snaps = sim.get_analyzer('snapshot')
    people = snaps.snapshots[-1] # snapshot from 2020
    df = pd.DataFrame({'age':people.age, 'sex':people.is_female})
    df['sex'].replace({True:'Female', False:'Male'}, inplace=True)
    df['Age Bin'] = pd.cut(df['age'], people.age_bin_edges)
    for lk, lkey in enumerate(layer_keys):
        df[lkey] = people.n_rships[lk]
    dfm = df.melt(id_vars=['Age Bin', 'sex'], value_vars=layer_keys, var_name='Layer', value_name='n_rships')
    g = sns.catplot(data=dfm, kind='bar', x='Age Bin', y='n_rships', hue='sex', col='Layer', sharey=False, height=8, aspect=0.5, legend_out=False, palette='tab10')
    g.tick_params(axis='x', which='both', rotation=70)
    g.set_ylabels('Number of Relationships')
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(sim.label)


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()
    network_demo()
    #cluster_demo()
    sc.toc(T)
    print('Done.')