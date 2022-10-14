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
        self.states = ['n_sus', 'n_inf', 'n_can', 'alive','age']
        self.res = sc.objdict()
        for k in self.states:
            self.res[k + '_agents'] = np.zeros(sim.npts)
            self.res[k + '_people'] = np.zeros(sim.npts)
    
    def apply(self, sim):
        
        dt = sim['dt']
        npts = sim.npts
        ppl = sim.people
        
        self.res['n_sus_agents'][ppl.t] = ppl.susceptible.sum()
        self.res['n_sus_people'][ppl.t] = (ppl.susceptible * ppl.scale).sum()
        self.res['n_inf_agents'][ppl.t] = (ppl.infectious.sum()).sum()
        self.res['n_inf_people'][ppl.t] = (ppl.infectious.sum() * ppl.scale).sum()
        self.res['n_can_agents'][ppl.t] = ppl.cancerous.sum()
        self.res['n_can_people'][ppl.t] = (ppl.cancerous.sum(axis=0) * ppl.scale).sum()

        self.res['age_agents'][ppl.t] = np.average(ppl.age, weights=(ppl.alive))
        self.res['age_people'][ppl.t] = np.average(ppl.age, weights=(ppl.alive)*ppl.scale)

        self.res['alive_agents'][ppl.t] = (ppl.alive).sum()
        self.res['alive_people'][ppl.t] = ppl.scale[ppl.alive].sum()
        self.res['age_agents'][ppl.t] = np.average(ppl.age)
        self.res['age_people'][ppl.t] = np.average(ppl.age, weights=ppl.scale)

        if ppl.t == npts - 1:
            self.res['t'] = np.arange(0, npts*dt, dt)
            self.res['year'] = sim.yearvec
            # self.res['cum_inf_agents'] = np.cumsum(self.res['new_inf_agents'])
            # self.res['cum_inf_people'] = np.cumsum(self.res['new_inf_people'])
            # self.res['cum_can_agents'] = np.cumsum(self.res['new_can_agents'])
            # self.res['cum_can_people'] = np.cumsum(self.res['new_can_people'])
            
    @property
    def df(self):
        return sc.dataframe(self.res)
        
    def plot(self, fig=None, color='b', alpha=1):
        
        df = self.df
        
        nrows,ncols = sc.getrowscols(len(df.columns)-2)
        
        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        else:
            plt.figure(fig)

        for i, key in enumerate(df.columns):
            if key not in ['t', 'year']:
                plt.subplot(nrows, ncols, i + 1)
                plt.plot(df.year, df[key], color=color, alpha=alpha)
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


if __name__ == '__main__':
    
    msim = hpv.parallel(sims)
    df = msim.compare(output=True, show_match=True)
    print(df.to_string())
    
    fig = None
    for sim in msim.sims:
        a = sim.get_analyzer()
        color = ['b','r'][sim['use_multiscale']]
        fig = a.plot(fig=fig, color=color, alpha=0.3)
    plt.show()
    
    T.toc('Done')