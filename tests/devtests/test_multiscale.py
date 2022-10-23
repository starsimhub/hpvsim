'''
Test multiscale assumptions
''' 

import numpy as np
import sciris as sc
import pylab as pl
import hpvsim as hpv

T = sc.timer()

repeats = 10
parallel = True
showlegend = False

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
        self.label = sim.label
        self.states = ['n_sus', 'n_inf', 'n_can', 'alive','age']
        self.res = sc.objdict()
        for k in self.states:
            self.res[k + '_agents'] = np.zeros(sim.npts)
            self.res[k + '_people'] = np.zeros(sim.npts)
        self.age = sc.objdict()
        self.agebins = np.arange(0,101,10)
        self.multiscale = sim['use_multiscale']
        return
    
    
    def apply(self, sim):
        
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


    def finalize(self, sim):
        super().finalize()
        
        dt = sim['dt']
        npts = sim.npts
        
        self.res['t'] = np.arange(0, npts*dt, dt)
        self.res['year'] = sim.yearvec
        ages = sim.people.age
        date_cancerous = np.nansum(sim.people.date_cancerous, axis=0)
        cancer_inci_inds = sc.findinds(date_cancerous) # WARNING, won't work if dead agents are removed
        cancer_death_inds = sc.findinds(sim.people.dead_cancer)
        self.age['bins'] = self.agebins[:-1] # Use start rather than edges
        self.age['cancer_inci'], _   = np.histogram(ages[cancer_inci_inds], self.agebins)
        self.age['cancer_deaths'], _ = np.histogram(ages[cancer_death_inds], self.agebins)
        return
    
        
    def df(self):
        r = sc.objdict()
        r.res = sc.dataframe(self.res)
        r.age = sc.dataframe(self.age)
        return r
        
    
def plot_compare_multiscale(msim, fig=None, alpha=0.3):
    
    def plot_single(analyzer, fig, alpha=1):
        r = analyzer.df()
        
        nrows,ncols = sc.getrowscols((len(r.res.columns) + len(r.age.columns) - 3)*2, ncols=4)
        
        if fig is None:
            fig = pl.figure(figsize=(18, 14))
        else:
            pl.figure(fig)

        ms = analyzer.multiscale
        index = ms - 1
        color = ['b','r'][ms]
        for i, key in enumerate(r.res.columns):
            if key not in ['t', 'year']:
                index += 2
                pl.subplot(nrows, ncols, index)
                pl.plot(r.res.year, r.res[key], color=color, alpha=alpha, label=analyzer.label)
                pl.title(key)
                if showlegend:
                    pl.legend()
        
        for i, key in enumerate(r.age.columns):
            if key not in ['bins']:
                index += 2
                pl.subplot(nrows, ncols, index)
                pl.plot(r.age.bins, r.age[key], color=color, alpha=alpha, label=analyzer.label)
                pl.title(key)
                if showlegend:
                    pl.legend()
        
        pl.tight_layout()
        return fig
    
    fig = None
    for sim in msim.sims:
        a = sim.get_analyzer()
        fig = plot_single(analyzer=a, fig=fig, alpha=alpha)
    
    pl.show()
    
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
    
    msim = hpv.parallel(sims, keep_people=True, parallel=parallel)
    df = msim.compare(output=True, show_match=True)
    if showlegend:
        df.disp()
    
    plot_compare_multiscale(msim)
    
    T.toc('Done')