'''
Debug Kenya population growth
'''

import sciris as sc
import numpy as np
import pylab as pl
import hpvsim as hpv
sc.options(dpi=150, font='monospace')


class track_n_alive(hpv.Analyzer):
    ''' Deprecated now that n_alive is tracked '''

    def __init__(self, data=None):
        self.t = sc.autolist()
        self.n_alive = sc.autolist()
        self.data = data
        return

    def apply(self, sim):
        self.t += sim.t
        self.n_alive += sim.people.alive.sum()
        self.sim = sim
        return

    def plot(self, factor=1e6):
        years = self.sim.yearvec[self.t] # + 1 since update happens before analyzer
        pop = np.array(self.n_alive)*self.sim["pop_scale"]/factor
        pop0 = pop[0]
        pop1 = pop[-1]
        y0 = years[0]
        y1 = years[-1]
        df = self.data
        df = df[np.multiply(df.year.values>=y0, df.year.values<=y1)].reset_index()
        df.pop_size /= factor
        ind0 = sc.findnearest(df.year, y0)
        ind1 = sc.findnearest(df.year, y1)
        dy0 = df.year[ind0]
        dy1 = df.year[ind1]
        data0 = df.pop_size[ind0]
        data1 = df.pop_size[ind1]
        ratio = pop1/data1
        self.out = sc.objdict(years=years, pop=pop, pop0=pop0, pop1=pop1, y0=y0, y1=y1, data0=data0, data1=data1, ratio=ratio)

        pl.figure()
        pl.scatter(df.year, df.pop_size, c='k', label='Data', alpha=0.4)
        pl.plot(years, pop, label='Model', lw=2, alpha=1.0)
        pl.xlabel('Year')
        pl.ylabel('Population size')
        pl.title(f'''
{sim.label}
Model: {y0:0.1f} = {pop0:0.2f}m; {y1:0.1f} = {pop1:0.2f}m
Data:  {dy0:0.1f} = {data0:0.2f}m; {dy1:0.1f} = {data1:0.2f}m
Ratio: {ratio:0.3f}''')
        pl.legend()
        sc.figlayout()
        pl.show()
        return


loc = 'kenya'
df = hpv.data.get_total_pop(loc)


for label,mig in {'No migration':False, 'With migration':True}.items():
    pars = dict(
        n_agents = round(df.pop_size[0]/1e3),
        start = df.year[0],
        end = 2030,
        location = loc,
        pop_scale = 1e3,
        genotypes = [],
        beta = 0.0,
        dt = 0.5,
        use_migration = mig,
    )


    sim = hpv.Sim(pars, label=label, analyzers=track_n_alive(data=df))
    sim.run()

    a = sim.get_analyzer()
    a.plot()