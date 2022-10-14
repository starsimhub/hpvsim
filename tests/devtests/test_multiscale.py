'''
Test multiscale assumptions
''' 

import numpy as np
import sciris as sc
import pylab as plt
import hpvsim as hpv

T = sc.timer()

repeats = 2

pars = dict(
    n_agents  = 1e3,
    start     = 1975,
    n_years   = 50,
    burnin    = 25,
    genotypes = [16,18],
    verbose = -1,
)


#% Define analyzer

class multitest(hpv.Analyzer):
    
    
    def initialize(self, sim):
        super().initialize()
        self.res = sc.objdict()
        for k in ['n_sus', 'n_inf', 'n_can', 'new_inf', 'new_rec', 'new_can', 'cum_inf', 'cum_can', 'alive','age']:
            self.res[k + '_agents'] = np.zeros(sim.npts)
            self.res[k + '_people'] = np.zeros(sim.npts)
    
    def apply(self, sim):
        
        dt = sim['dt']
        npts = sim.npts
        self.res['t'] = np.arange(0, npts*dt, dt)
        self.res['n_sus_agents'][self.ti] = self.susceptible.sum()
        self.res['n_sus_people'][self.ti] = (self.susceptible * self.scale).sum()
        self.res['n_inf_agents'][self.ti] = (self.infectious.sum()).sum()
        self.res['n_inf_people'][self.ti] = (self.infectious.sum() * self.scale).sum()
        self.res['n_can_agents'][self.ti] = self.cancerous.sum()
        self.res['n_can_people'][self.ti] = (self.cancerous.sum(axis=0) * self.scale).sum()

        self.res['age_agents'][self.ti] = np.average(self.age, weights=(~self.removed))
        self.res['age_people'][self.ti] = np.average(self.age, weights=(~self.removed)*self.scale)

        self.res['alive_agents'][self.ti] = (~self.removed).sum()
        self.res['alive_people'][self.ti] = self.scale[~self.removed].sum()
        self.res['age_agents'][self.ti] = np.average(self.age)
        self.res['age_people'][self.ti] = np.average(self.age, weights=self.scale)

        if self.ti == npts - 1:
            self.res['cum_inf_agents'] = np.cumsum(self.res['new_inf_agents'])
            self.res['cum_inf_people'] = np.cumsum(self.res['new_inf_people'])
            self.res['cum_can_agents'] = np.cumsum(self.res['new_can_agents'])
            self.res['cum_can_people'] = np.cumsum(self.res['new_can_people'])
            
    @property
    def df(self):
        return sc.dataframe(self.res)
        
    def plot(self, fig=None, color='b', alpha=1):
        
        df = self.df
        
        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        else:
            plt.figure(fig)

        for i, key in enumerate(df.columns):
            if key != 't':
                plt.subplot(5, 5, i + 1)
                plt.plot(df.t, df[key], color=color, alpha=alpha)
                plt.title(key)
        plt.tight_layout()
        return fig


sims = []
for use_multiscale in [False, True]:
    for r in range(repeats):
        label = ['default', 'multiscale'][use_multiscale]
        label += f'{r}'
        simpars = dict(use_multiscale=use_multiscale, rand_seed=r, label=label, analyzers=multitest())
        sim = hpv.Sim(pars, **simpars)
        sims.append(sim)

msim = hpv.parallel(sims)
df = msim.compare(output=True, show_match=True)
print(df.to_string())

T.toc('Done')