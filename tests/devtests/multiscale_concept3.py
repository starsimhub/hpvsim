import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
# import hpvsim as hpv
import hpvsim.utils as hpu
from collections import Counter
import pandas as pd

npts = 300
dt = 0.25

popsize = 100e3
# scale = 100 # people per agent for non-cancer and non-death states
scale_can = 50 # people per agent for cancer and death states

can_prob = 0.005 # probability of getting cancer if infected
f = 1.0
time_to_rec = dict(dist='lognormal', par1=2.5, par2=f*4.0) # average infection duration, given recovery
time_to_can = dict(dist='lognormal', par1=5.5, par2=f*3.0) # average time to cancer
beta = 0.11 # transmission scale factor

class People(sc.objdict):
    people_keys = ['age', 'sus', 'inf', 'can', 'scale', 'date_rec', 'date_can', 'date_remove', 'removed']

    def __init__(self, n, n_infected):

        self.ti = 0
        arr = lambda val=None: np.full(n, val)

        self.age = np.random.uniform(20, 40, size=n)
        self.sus = arr(True)
        self.inf = arr(False)
        self.can = arr(False)
        self.scale = arr(popsize/n)
        self.date_rec = arr(np.nan)
        self.date_can = arr(np.nan)
        self.date_remove = arr(np.nan)
        self.removed = arr(False)

        self.res = sc.objdict()
        for k in ['n', 'n_sus', 'n_inf', 'n_can', 'new_inf', 'new_rec', 'new_can', 'cum_inf', 'cum_can', 'alive','age']:
            self.res[k + '_agents'] = np.zeros(npts)
            self.res[k + '_people'] = np.zeros(npts)

        self.infect(np.arange(0, n_infected))

    @property
    def t(self):
        return self.ti*dt

    def extend(self, ind, n_copies):
        for key in self.people_keys:
            orig_len = len(self[key])
            to_copy = [self[key][ind]]*n_copies
            self[key] = sc.cat(self[key], to_copy)
            new_len = len(self[key])
        new_inds = np.arange(orig_len, new_len)
        return new_inds

    def step(self):

        self.age[~self.removed] += dt

        new_rec = self.check_recovery()
        self.res['new_rec_agents'][self.ti] = len(new_rec)
        self.res['new_rec_people'][self.ti] = self.scale[new_rec].sum()

        new_inf = self.check_transmission()
        self.res['new_inf_agents'][self.ti] = len(new_inf)
        self.res['new_inf_people'][self.ti] = self.scale[new_inf].sum()

        self.infect(new_inf)

        new_can = self.check_cancer()
        self.res['new_can_agents'][self.ti] = len(new_can)
        self.res['new_can_people'][self.ti] = self.scale[new_can].sum()

        self.check_removed()

        self.update_results()

        self.ti += 1

    def check_transmission(self):
        inds = np.random.choice(hpu.true(self.sus), hpu.poisson(np.count_nonzero(self.inf) * beta))
        return inds

    def infect(self, inds):
        # Calculate transmission
        self.sus[inds] = False
        self.inf[inds] = True

        # Assign everyone a recovery date
        self.date_rec[inds] = self.t + hpu.sample(**time_to_rec, size=len(inds))

        # Who has cancer?
        can_inds = hpu.binomial_filter(can_prob, inds)
        self.date_rec[can_inds] = np.nan
        self.date_can[can_inds] = self.t + hpu.sample(**time_to_can, size=len(can_inds))

    def remove(self, inds):
        self.sus[inds] = False
        self.inf[inds] = False
        self.can[inds] = False
        self.removed[inds] = True

    def check_removed(self):
        new_removed =  hpu.true(~self.removed & (self.t >= self.date_remove))
        self.remove(new_removed)

    def check_recovery(self):
        new_rec =  hpu.true(self.inf & (self.t >= self.date_rec))
        self.sus[new_rec] = True
        self.inf[new_rec] = False
        return new_rec

    def check_cancer(self):
        new_can =  hpu.true(self.inf & (self.t >= self.date_can))
        self.inf[new_can] = False
        self.can[new_can] = True
        return new_can

    def update_results(self):

        self.res['t'] = np.arange(0, npts*dt, dt)
        self.res['n_agents'][self.ti] = len(self.scale)
        self.res['n_people'][self.ti] = self.scale.sum()
        self.res['n_sus_agents'][self.ti] = (self.sus).sum()
        self.res['n_sus_people'][self.ti] = (self.sus * self.scale).sum()
        self.res['n_inf_agents'][self.ti] = (self.inf).sum()
        self.res['n_inf_people'][self.ti] = (self.inf * self.scale).sum()
        self.res['n_can_agents'][self.ti] = (self.can).sum()
        self.res['n_can_people'][self.ti] = (self.can * self.scale).sum()

        self.res['age_agents'][self.ti] = np.average(self.age, weights=(~self.removed).astype(float))
        self.res['age_people'][self.ti] = np.average(self.age, weights=(~self.removed).astype(float)*self.scale)

        self.res['alive_agents'][self.ti] = (~self.removed).sum()
        self.res['alive_people'][self.ti] = self.scale[~self.removed].sum()
        self.res['age_agents'][self.ti] = np.average(self.age)
        self.res['age_people'][self.ti] = np.average(self.age, weights=self.scale)

        if self.ti == npts - 1:
            self.res['cum_inf_agents'] = np.cumsum(self.res['new_inf_agents'])
            self.res['cum_inf_people'] = np.cumsum(self.res['new_inf_people'])
            self.res['cum_can_agents'] = np.cumsum(self.res['new_can_agents'])
            self.res['cum_can_people'] = np.cumsum(self.res['new_can_people'])

    def treat(self, inds):
        inds = inds[~(self.removed[inds] | self.can[inds]) ]
        self.sus[inds] = True
        self.inf[inds] = False
        self.date_can[inds] = np.nan
        return inds


class MultiScalePeople(People):

    def __init__(self, n, *args, **kwargs):
        self.cancer_queue = [] # contains (time to create, index to create from)
        super().__init__(n, *args, **kwargs)

    def infect(self, inds):
        # Calculate transmission
        self.sus[inds] = False
        self.inf[inds] = True

        # Assign everyone a recovery date
        self.date_rec[inds] = self.t + hpu.sample(**time_to_rec, size=len(inds))

        # Pick which agents will give rise to cancers - remove the scaled agent when they progress
        can_inds = hpu.binomial_filter(can_prob, inds)
        self.date_rec[can_inds] = np.nan
        self.date_remove[can_inds] = self.t + hpu.sample(**time_to_can, size=len(can_inds))

        # Queue creation of cancer agents
        for ind in inds:
            n_cancers = np.random.binomial(int(np.round(scale_can)),can_prob)
            date_can  = self.t + hpu.sample(**time_to_can, size=n_cancers)
            self.cancer_queue.extend([(d,ind) for d in date_can])

    def check_cancer(self):
        t = self.t
        cancers_today = Counter([x[1] for x in self.cancer_queue if x[0] <= t])
        new_can = []
        for ind, n in cancers_today.items():
            inds = self.extend(ind, n)
            new_can.extend(inds)
            self.sus[inds] = False
            self.removed[inds] = False
            self.inf[inds] = False
            self.can[inds] = True
            self.scale[inds] = scale_can
            self.date_remove[inds] = np.nan
        self.cancer_queue = [x for x in self.cancer_queue if x[0] > t]
        return new_can

    def treat(self, inds):
        inds = super().treat(inds)
        inds = set(inds)
        self.cancer_queue = [x for x in self.cancer_queue if x[1] not in inds]


class Sim(sc.prettyobj):

    def __init__(self, people):
        self.people = people

    @property
    def results(self):
        return self.people.res

    @property
    def df(self):
        return sc.dataframe(self.results)

    def run(self):
        for i in range(npts):
            self.people.step()


def plot(df, fig=None, color='b', alpha=1):
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
    else:
        plt.figure(fig)

    for i, key in enumerate(df.columns):
        if key != 't':
            plt.subplot(6, 5, i + 1)
            plt.plot(df.t, df[key], color=color, alpha=alpha)
            plt.title(key)
    plt.tight_layout()
    return fig

def run_single_scale(n=10_000, *args, **kwargs):
    sim = Sim(People(n, n_infected=100))
    sim.run()
    return sim.df

def run_multi_scale(n=1_000, *args, **kwargs):
    sim = Sim(MultiScalePeople(n, n_infected=100))
    sim.run()
    return sim.df

if __name__ == '__main__':
    
    repeats = 20

    df = run_single_scale()
    fig = plot(df)

    df = run_multi_scale()
    plot(df, fig, 'r')

    dfs = sc.parallelize(run_single_scale, repeats)
    fig = None
    for df in dfs:
        fig = plot(df, fig=fig, alpha=0.3)
    df_single_median = pd.concat(dfs).groupby(level=0).median()
    plot(df_single_median, fig=fig, color='k')

    dfs = sc.parallelize(run_multi_scale, repeats)
    fig = None
    for df in dfs:
        fig = plot(df, fig=fig, color='r', alpha=0.3)
    df_multi_median = pd.concat(dfs).groupby(level=0).median()
    plot(df_multi_median, fig=fig, color='k')

    fig = plot(df_single_median)
    plot(df_multi_median, fig, 'r')
    
    plt.show()
