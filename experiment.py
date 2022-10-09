'''
Todo:
- Add transmission
- Same scale factor with all states, large number of agents, ground truth
'''

import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
# import hpvsim as hpv
import hpvsim.utils as hpu

n = 1000
npts = 300
dt = 0.1
scale = 100
scale_can = 1

recover_prob = 1 / 10
can_prob = 0.001
death_prob = 0.1
beta = 0.15

class People():
    people_keys = ['age', 'sus', 'inf', 'can', 'dead', 'scale']

    def __init__(self):

        self.age = np.random.uniform(20, 40, size=n)

        arr = lambda val=None: np.full(n, val)
        self.sus = arr(True)
        self.inf = arr(False)
        self.can = arr(False)
        self.dead = arr(False)
        self.scale = arr(scale)

        self.res = sc.objdict()
        for k in ['n_sus', 'n_inf', 'n_can', 'n_dead', 'new_inf', 'new_can', 'new_dead', 'cum_inf', 'cum_can', 'cum_dead', 'alive','age']:
            self.res[k + '_agents'] = np.zeros(npts)
            self.res[k + '_people'] = np.zeros(npts)

        self.sus[0:10] = False
        self.inf[0:10] = True

    def extend(self, inds):
        for key in self.people_keys:
            orig_len = len(self[key])
            to_copy = self[key][inds]
            self[key] = sc.cat(self[key], to_copy)
            new_len = len(self[key])
        new_inds = np.arange(orig_len, new_len)
        return new_inds

    def update(self, ti):

        self.age += dt

        self.step_transmission(ti)
        self.step_cancer(ti)
        self.update_results(ti)

    def step_transmission(self, ti):
        # Calculate transmission
        new_inf = np.random.choice(hpu.true(self.sus), hpu.poisson(np.count_nonzero(self.inf) * beta))
        self.sus[new_inf] = False
        self.inf[new_inf] = True
        self.res['new_inf_agents'][ti] = len(new_inf)
        self.res['new_inf_people'][ti] = len(new_inf) * scale

        # Calculate recovery
        recover_inds = hpu.binomial_filter(recover_prob, hpu.true(self.inf))
        self.sus[recover_inds] = True
        self.inf[recover_inds] = False

    def step_cancer(self, ti):
        # Calculate cancer
        new_can = hpu.binomial_filter(can_prob, hpu.true(self.inf))
        self.inf[new_can] = False
        self.can[new_can] = True
        self.res['new_can_agents'][ti] = len(new_can)
        self.res['new_can_people'][ti] = len(new_can) * scale

        # Calculate death
        new_dead = hpu.binomial_filter(death_prob, hpu.true(self.can))
        self.can[new_dead] = False
        self.dead[new_dead] = True
        self.res['new_dead_agents'][ti] = len(new_dead)
        self.res['new_dead_people'][ti] = len(new_dead) * scale

    def update_results(self, ti):

        self.res['t'] = np.arange(0, npts*dt, dt)
        self.res['n_sus_agents'][ti] = (self.sus).sum()
        self.res['n_sus_people'][ti] = (self.sus * self.scale).sum()
        self.res['n_inf_agents'][ti] = (self.inf).sum()
        self.res['n_inf_people'][ti] = (self.inf * self.scale).sum()
        self.res['n_can_agents'][ti] = (self.can).sum()
        self.res['n_can_people'][ti] = (self.can * self.scale).sum()
        self.res['n_dead_agents'][ti] = (self.dead).sum()
        self.res['n_dead_people'][ti] = (self.dead * self.scale).sum()

        self.res['alive_agents'][ti] = (~self.dead).sum()
        self.res['alive_people'][ti] = ((~self.dead) * self.scale).sum()
        self.res['age_agents'][ti] = np.average(self.age, weights=(~self.dead).astype(float))
        self.res['age_people'][ti] = np.average(self.age, weights=(~self.dead).astype(float)*self.scale)

        if ti == npts - 1:
            self.res['cum_inf_agents'] = np.cumsum(self.res['new_inf_agents'])
            self.res['cum_inf_people'] = np.cumsum(self.res['new_inf_people'])
            self.res['cum_can_agents'] = np.cumsum(self.res['new_can_agents'])
            self.res['cum_can_people'] = np.cumsum(self.res['new_can_people'])
            self.res['cum_dead_agents'] = np.cumsum(self.res['new_dead_agents'])
            self.res['cum_dead_people'] = np.cumsum(self.res['new_dead_people'])

class MultiScalePeople(People):

    def step_cancer(self, ti):


        # Calculate cancer
        new_can = hpu.binomial_filter(can_prob, hpu.true(self.inf))
        self.inf[new_can] = False
        self.can[new_can] = True
        self.res['new_can_agents'][ti] = len(new_can)
        self.res['new_can_people'][ti] = len(new_can) * scale

        # Calculate death
        new_dead = hpu.binomial_filter(death_prob, hpu.true(self.can))
        self.can[new_dead] = False
        self.dead[new_dead] = True
        self.res['new_dead_agents'][ti] = len(new_dead)
        self.res['new_dead_people'][ti] = len(new_dead) * scale



class Sim(sc.prettyobj):

    def __init__(self, people):
        self.people = people
        self.ti = 0

    @property
    def results(self):
        return self.people.res

    @property
    def df(self):
        return sc.dataframe(self.results)

    def run(self):
        for ti in range(npts):
            self.ti += dt
            self.people.update(ti)
    #
    # # def plot(self):
    #     fig = plt.figure(figsize=(14, 10))
    #     for i, key in enumerate(sim.df.columns):
    #         if key != 't':
    #             plt.subplot(2, 4, i)
    #             plt.plot(sim.df.t, sim.df[key])
    #             plt.title(key)
    #
    #     plt.show()
    #     return fig


sim = Sim(People())
sim.run()
# fig = sim.plot()

fig = plt.figure(figsize=(14, 10))
for i, key in enumerate(sim.df.columns):
    if key != 't':
        plt.subplot(5, 5, i+1)
        plt.plot(sim.df.t, sim.df[key])
        plt.title(key)

plt.show()
