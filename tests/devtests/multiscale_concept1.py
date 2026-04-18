'''
Todo:
- Add transmission
- Same scale factor with all states, large number of agents, ground truth
'''

import numpy as np
import sciris as sc
import pylab as pl

debug = 0

default_pars = sc.objdict(
    n = 1000,
    npts = 200,
    dt = 0.1,
    scale = [100, 1][debug],
    scale_dys = [10, 1][debug],
    scale_can = 1,
    dys_prob = 0.001,
    can_prob = 0.001,
    # death_prob = 0.1,
    # conn_prob = 0.01,
)


class People(sc.objdict):

    people_keys = ['age', 'sus', 'dys', 'can', 'dead', 'scale']

    def __init__(self, pars):

        def arr(val=None):
            return np.full(pars.n, val)

        self.pars = pars
        self.age = np.random.uniform(20, 40, size=self.pars.n)
        self.sus = arr(True)
        self.dys = arr(False)
        self.can = arr(False)
        self.dead = arr(False)
        self.scale = arr(self.pars.scale)
        return


    def extend(self, inds):
        for key in self.people_keys:
            orig_len = len(self[key])
            to_copy = self[key][inds]
            self[key] = sc.cat(self[key], to_copy)
            new_len = len(self[key])

        new_inds = np.arange(orig_len, new_len)
        return new_inds


    def update(self, t):
        self.age += self.pars.dt

        # Make some people dysplastic
        sus_inds = sc.findinds(self.sus)
        n_sus_inds = len(sus_inds)
        dys_throws = np.random.rand(n_sus_inds)
        sus_to_dys = sus_inds[sc.findinds(dys_throws < self.pars.dys_prob)]
        dys_ratio = self.pars.scale/self.pars.scale_dys
        all_dys = sus_inds[sc.findinds(dys_throws < self.pars.dys_prob*dys_ratio)] # Need to resample with replacement
        to_new_dys = np.setdiff1d(all_dys, sus_to_dys)

        # Create new agents and set states
        new_dys_inds = self.extend(to_new_dys)
        all_dys_inds = sc.cat(sus_to_dys, new_dys_inds)
        self.sus[all_dys_inds] = False
        self.dys[all_dys_inds] = True
        self.scale[all_dys_inds] = self.pars.scale_dys

        # Make cancer
        dys_inds = sc.findinds(self.dys)
        n_dys_inds = len(dys_inds)
        can_throws = np.random.rand(n_dys_inds)
        dys_to_can = dys_inds[sc.findinds(can_throws < self.pars.can_prob)]
        can_ratio = self.pars.scale_dys/self.pars.scale_can
        all_can = dys_inds[sc.findinds(can_throws < self.pars.can_prob*can_ratio)] # Need to resample with replacement
        to_new_can = np.setdiff1d(all_can, dys_to_can)

        # Create new agents and set states
        new_can_inds = self.extend(to_new_can)
        all_can_inds = sc.cat(dys_to_can, new_can_inds)
        self.sus[all_can_inds] = False
        self.dys[all_can_inds] = False
        self.can[all_can_inds] = True
        self.scale[all_can_inds] = self.pars.scale_can

        res = sc.objdict(
            t = t,
            n = self.scale.sum(),
            age = np.average(self.age, weights=self.scale),
            sus = np.average(self.sus, weights=self.scale),
            dys = np.average(self.dys, weights=self.scale),
            can = np.average(self.can, weights=self.scale),
            agent_sus = self.sus.sum(),
            agent_dys = self.dys.sum(),
            agent_can = self.can.sum(),
        )

        print(sc.strjoin([f'{k}={v:n}' for k,v in res.items()]))

        return res



class Sim(sc.prettyobj):

    def __init__(self, pars=None):
        pars = sc.mergedicts(default_pars, pars)
        self.pars = pars
        self.people = People(pars)
        self.results = []
        self.t = 0
        return

    def run(self):
        for t in range(self.pars.npts):
            self.t += self.pars.dt
            res = self.people.update(t)
            self.results.append(res)

        self.df = sc.dataframe(self.results)
        return


    def plot(self):
        fig = pl.figure(figsize=(14,10))
        for i,key in enumerate(self.df.columns):
            if key != 't':
                pl.subplot(2,4,i)
                pl.plot(self.df.t, self.df[key])
                pl.title(key)

        pl.show()
        return fig



sim = Sim()
sim.run()
fig = sim.plot()