'''
Run dwelltime tests
'''

import hpvsim as hpv
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt

class dwelltime(hpv.Analyzer):
    '''
    Determine time spent in health states for those who do NOT get cancer
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.dwelltime = dict()
        for state in ['hpv', 'cin1', 'cin2', 'cin3', 'total']:
            self.dwelltime[state] = []

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            genotypes, inds = (sim.people.date_clearance == sim.t).nonzero()
            if len(inds):
                date_exposed = sim.people.date_exposed[genotypes, inds]
                cin1_inds = hpv.true(~np.isnan(sim.people.date_cin1[genotypes, inds]))
                cin2_inds = hpv.true(~np.isnan(sim.people.date_cin2[genotypes, inds]))
                cin3_inds = hpv.true(~np.isnan(sim.people.date_cin3[genotypes, inds]))
                hpv_time = ((sim.people.date_cin1[genotypes[cin1_inds], inds[cin1_inds]] - date_exposed[cin1_inds]) * sim['dt']).tolist() + \
                           ((sim.people.date_cin1[genotypes[cin2_inds], inds[cin2_inds]] - date_exposed[cin2_inds])*sim['dt']).tolist() + \
                           ((sim.people.date_cin1[genotypes[cin3_inds], inds[cin3_inds]] - date_exposed[cin3_inds])*sim['dt']).tolist()

                cin1_time = ((sim.t - sim.people.date_cin1[genotypes[cin1_inds], inds[cin1_inds]])*sim['dt']).tolist() + \
                            ((sim.people.date_cin2[genotypes[cin2_inds], inds[cin2_inds]] - sim.people.date_cin1[genotypes[cin2_inds], inds[cin2_inds]]) * sim['dt']).tolist() + \
                            ((sim.people.date_cin2[genotypes[cin3_inds], inds[cin3_inds]] - date_exposed[cin3_inds])*sim['dt']).tolist()

                cin2_time = ((sim.t - sim.people.date_cin2[genotypes[cin2_inds], inds[cin2_inds]])*sim['dt']).tolist() + \
                            ((sim.people.date_cin3[genotypes[cin3_inds], inds[cin3_inds]] - sim.people.date_cin2[genotypes[cin3_inds], inds[cin3_inds]]) * sim['dt']).tolist()
                cin3_time = ((sim.t - sim.people.date_cin3[genotypes[cin3_inds], inds[cin3_inds]]) * sim['dt']).tolist()
                total_time = ((sim.t - date_exposed)*sim['dt']).tolist()
                self.dwelltime['hpv'] += hpv_time
                self.dwelltime['cin1'] += cin1_time
                self.dwelltime['cin2'] += cin2_time
                self.dwelltime['cin3'] += cin3_time
                self.dwelltime['total'] += total_time


def run_sim(location=None, seed=0, debug=0, use_ccut=False):
    # Parameters
    pars = dict(n_agents=[50e3, 5e3][debug],
                start=[1950, 1980][debug],
                end=2050,
                dt=[0.5, 1.0][debug],
                network='default',
                location=location,
                genotypes=[16,18,'hrhpv'],
                rand_seed=seed,
                verbose=0.0,

                )

    if use_ccut:
        pars['clinical_cutoffs'] = {'cin1': 0.8, 'cin2': 0.9, 'cin3': 0.99}

    sim = hpv.Sim(pars=pars, analyzers=[dwelltime(start_year=2000)])

    sim.run()

    a = sim.get_analyzer(dwelltime)


    return sim, a


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    dwelltimes = []
    sims = []
    for ccut in [False, True]:
        sim, a = run_sim(location='india', use_ccut=ccut)
        dwelltimes.append(a)
        sims.append(sim)

    fig, ax = plt.subplots(3, 2, figsize=(12, 12), sharey=True)
    ax[0,0].boxplot([dwelltimes[0].dwelltime['hpv'],dwelltimes[1].dwelltime['hpv']], widths=0.6, showfliers=False)
    ax[0,0].set_title('HPV dwelltime')
    ax[0,0].set_xticklabels(['', ''])

    ax[0,1].boxplot([dwelltimes[0].dwelltime['cin1'],dwelltimes[1].dwelltime['cin1']], widths=0.6, showfliers=False)
    ax[0,1].set_title('CIN1 dwelltime')
    ax[0,1].set_xticklabels(['', ''])

    ax[1,0].boxplot([dwelltimes[0].dwelltime['cin2'],dwelltimes[1].dwelltime['cin2']], widths=0.6, showfliers=False)
    ax[1,0].set_title('CIN2 dwelltime')
    ax[1,0].set_xticklabels(['Fast prog', 'Slow prog'])

    ax[1,1].boxplot([dwelltimes[0].dwelltime['cin3'],dwelltimes[1].dwelltime['cin3']], widths=0.6, showfliers=False)
    ax[1,1].set_title('CIN3 dwelltime')
    ax[1,1].set_xticklabels(['Fast prog', 'Slow prog'])

    ax[2,0].boxplot([dwelltimes[0].dwelltime['total'],dwelltimes[1].dwelltime['total']], widths=0.6, showfliers=False)
    ax[2,0].set_title('Total dwelltime')
    ax[2,0].set_xticklabels(['Fast prog', 'Slow prog'])
    fig.suptitle('Dwelltime for non-cancer causing HPV/Dysplasia')
    plt.tight_layout()
    fig.show()

    fig, ax = plt.subplots(2,1)
    labels = ['Fast prog', 'Slow prog']
    for i, sim in enumerate(sims):
        ax[0].plot(sim.results['year'], sim.results['total_infections'], label=labels[i])
        ax[1].plot(sim.results['year'], sim.results['total_cancers'], label=labels[i])
    fig.show()
    sc.toc(T)
    print('Done.')