'''
Test multiscale assumptions
''' 

import numpy as np
import sciris as sc
import pylab as pl
import hpvsim as hpv

T = sc.timer()

repeats    = 10
parallel   = 1

large_pop = 10e3
small_pop = 1e3
ratio = large_pop/small_pop
offset = 100

pars = dict(
    total_pop      = large_pop,
    ms_agent_ratio = 10,
    start          = 1975,
    n_years        = 50,
    genotypes      = [16,18],
    verbose        = -1,
)

loop_pars = [
    sc.objdict(n_agents=large_pop, use_multiscale=0),
    sc.objdict(n_agents=large_pop, use_multiscale=1),
    sc.objdict(n_agents=small_pop, use_multiscale=0),
    sc.objdict(n_agents=small_pop, use_multiscale=1),
    ]


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
        self.n_agents = sim['n_agents']
        return
    
    
    def apply(self, sim):
        
        ppl = sim.people
        
        self.res['n_sus_agents'][ppl.t] = ppl.susceptible.sum()
        self.res['n_sus_people'][ppl.t] = (ppl.susceptible * ppl.scale).sum()
        self.res['n_inf_agents'][ppl.t] = ppl.infectious.sum()
        self.res['n_inf_people'][ppl.t] = (ppl.infectious.sum(axis=0) * ppl.scale).sum()
        self.res['n_can_agents'][ppl.t] = ppl.cancerous.sum()
        self.res['n_can_people'][ppl.t] = (ppl.cancerous.sum(axis=0) * ppl.scale).sum()

        self.res['alive_agents'][ppl.t] = (ppl.alive).sum()
        self.res['alive_people'][ppl.t] = ppl.scale[ppl.alive].sum()
        
        self.res['age_agents'][ppl.t] = np.average(ppl.age, weights=(ppl.alive))
        self.res['age_people'][ppl.t] = np.average(ppl.age, weights=(ppl.alive)*ppl.scale)
        
        return


    def finalize(self, sim):
        super().finalize()
        
        dt = sim['dt']
        npts = sim.npts
        ppl = sim.people
        
        self.res['t'] = np.arange(0, npts*dt, dt)
        self.res['year'] = sim.yearvec
        ages = ppl.age
        date_cancerous = np.nansum(ppl.date_cancerous, axis=0)
        cancer_inci_inds = sc.findinds(date_cancerous) # WARNING, won't work if dead agents are removed
        cancer_death_inds = sc.findinds(ppl.dead_cancer)
        self.age['bins'] = self.agebins[:-1] # Use start rather than edges
        self.age['cancer_inci'], _   = np.histogram(ages[cancer_inci_inds], self.agebins, weights=ppl.scale[cancer_inci_inds])
        self.age['cancer_deaths'], _ = np.histogram(ages[cancer_death_inds], self.agebins, weights=ppl.scale[cancer_death_inds])
        return
    
        
    def df(self):
        r = sc.objdict()
        r.res = sc.dataframe(self.res)
        r.age = sc.dataframe(self.age)
        return r
        
    
def plot_compare_multiscale(msim, fig=None):
    
    def plot_single(analyzer, shared=None, fig=None, alpha=1, lw=1, factor=1):
        r = analyzer.df()
        
        nkinds = len(loop_pars)
        nrows,ncols = sc.getrowscols((len(r.res.columns) + len(r.age.columns) - 3)*nkinds, ncols=nkinds*2)
        
        if fig is None:
            fig = pl.figure(figsize=(30, 20))
        else:
            pl.figure(fig)

        ms_bool = analyzer.multiscale
        agent_bool = analyzer.n_agents != loop_pars[0].n_agents
        index = 0
        label = ['default', 'multiscale'][ms_bool]
        label += f' (n={analyzer.n_agents})'
        color = ['k','seagreen'][ms_bool]
        offset = 1 + 2*agent_bool + ms_bool
        
        # Handle shared
        if isinstance(shared, dict):
            if label in shared:
                first = False
            else:
                first = True
                shared[label] = sc.dcp(analyzer)
            sh = shared[label]
            
        for i, key in enumerate(r.res.columns):
            title = f'{key}\n{label}'
            if key not in ['t', 'year']:
                pl.subplot(nrows, ncols, index+offset)
                pl.plot(r.res.year, r.res[key]*factor, color=color, alpha=alpha, lw=lw, label=analyzer.label)
                if shared is not None and not first: # Calculate average on the fly
                    sh.res[key] += r.res[key]
                pl.title(title)
                index += nkinds
        
        for i, key in enumerate(r.age.columns):
            title = f'{key}\n{label}'
            if key not in ['bins']:
                pl.subplot(nrows, ncols, index+offset)
                pl.plot(r.age.bins, r.age[key]*factor, color=color, alpha=alpha, lw=lw, label=analyzer.label)
                if shared is not None and not first:
                    sh.age[key] += r.age[key]
                pl.title(title)
                index += nkinds
        
        return fig
    
    fig = None
    shared = sc.objdict()
    for s,sim in enumerate(msim.sims):
        a = sim.get_analyzer()
        fig = plot_single(analyzer=a, shared=shared, fig=fig, alpha=0.2, lw=1)
    
    for sh in shared.values():
        fig = plot_single(analyzer=sh, shared=None, fig=fig, alpha=1.0, lw=2, factor=1/repeats)
    
    sc.figlayout()
    pl.show()
    
    return fig


msims = sc.autolist()
for p in loop_pars:
    sims = sc.autolist()
    for r in range(repeats):
        label = ['default', 'multiscale'][p.use_multiscale]
        label += f'{r}'
        simpars = dict(**p, rand_seed=r+offset, label=label, analyzers=multitest())
        sim = hpv.Sim(pars, **simpars)
        sims += sim
    msims += hpv.MultiSim(sims)
msim = hpv.MultiSim.merge(*msims)


if __name__ == '__main__':
    
    msim.run(keep_people=True, parallel=parallel)
    mm = msim.split()
    for m in mm:
        m.mean()
    merged = hpv.MultiSim.merge(mm, base=True)
    dfall = msim.compare(output=True, show_match=True)
    dfmerged = merged.compare(output=True, show_match=True)
    dfmerged.disp()
    
    plot_compare_multiscale(msim)
    
    T.toc('Done')