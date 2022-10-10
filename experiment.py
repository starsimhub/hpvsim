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
from collections import Counter

n = 1000
npts = 300
dt = 0.25

scale = 100 # people per agent for non-cancer and non-death states
scale_can = 1 # people per agent for cancer and death states

can_prob = 0.05 # probability of getting cancer if infected
time_to_rec = dict(dist='lognormal', par1=2.5, par2=4.0) # average infection duration, given recovery
time_to_can = dict(dist='lognormal', par1=5.5, par2=3.0) # average time to cancer
beta = 0.11 # transmission scale factor

class People(sc.objdict):
    people_keys = ['age', 'sus', 'inf', 'can', 'scale', 'date_rec', 'date_can', 'date_remove', 'removed']

    def __init__(self, n_infected):

        self.ti = 0
        arr = lambda val=None: np.full(n, val)


        self.age = np.random.uniform(20, 40, size=n)
        self.sus = arr(True)
        self.inf = arr(False)
        self.can = arr(False)
        self.scale = arr(scale)
        self.date_rec = arr(np.nan)
        self.date_can = arr(np.nan)
        self.date_remove = arr(np.nan)
        self.removed = arr(False)

        self.res = sc.objdict()
        for k in ['n_sus', 'n_inf', 'n_can', 'new_inf', 'new_rec', 'new_can', 'cum_inf', 'cum_can', 'alive','age']:
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

        self.age += dt

        new_rec = self.check_recovery()
        self.res['new_rec_agents'][self.ti] = len(new_rec)
        self.res['new_rec_people'][self.ti] = len(new_rec) * scale

        new_inf = self.check_transmission()
        self.res['new_inf_agents'][self.ti] = len(new_inf)
        self.res['new_inf_people'][self.ti] = len(new_inf) * scale

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

    def __init__(self, *args, **kwargs):
        self.cancer_queue = [] # contains (time to create, index to create from)
        super().__init__(*args, **kwargs)

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
            n_cancers = np.random.binomial(int(np.round(scale/scale_can)),can_prob)
            date_can  = self.t + hpu.sample(**time_to_can, size=n_cancers)
            self.cancer_queue.extend([(d,ind) for d in date_can])

    def check_cancer(self):
        t = self.t
        cancers_today = Counter([x[1] for x in self.cancer_queue if x[0] >= t])
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
        self.cancer_queue = [x for x in self.cancer_queue if x[0] < t]
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

    def plot(self, fig=None, color='b'):
        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        else:
            plt.figure(fig)

        for i, key in enumerate(sim.df.columns):
            if key != 't':
                plt.subplot(5, 5, i + 1)
                plt.plot(sim.df.t, sim.df[key], color=color)
                plt.title(key)
        plt.tight_layout()
        return fig


sim = Sim(People(n_infected=100))
sim.run()
fig = sim.plot()

sim = Sim(MultiScalePeople(n_infected=100))
sim.run()
sim.plot(fig, 'r')


sim = Sim(People(n_infected=100))
sim.run()
fig = sim.plot()


#######################


#
# dur_to_peak_dys = sample(**dur_dyps, size=len(cin1_inds))
# prog_rate = genotype_pars[genotype_map[g]]['prog_rate']
# prog_time = genotype_pars[genotype_map[g]]['prog_time']
# mean_peaks = logf2(dur_to_peak_dys, prog_time, prog_rate)  # Apply a function that maps durations + genotype-specific progression to
#
# pars['severity_dist'] = dict(dist='lognormal', par1=None, par2=0.1) # Distribution of individual disease severity. Par1 is set to None because the mean is determined as a function of genotype and disease duration
#
# sev_dist = people.pars['severity_dist']['dist']
# sev_par2 = people.pars['severity_dist']['par2']
# peaks = np.minimum(1, sample(dist=sev_dist, par1=mean_peaks,par2=sev_par2))  # Evaluate peak dysplasia, which is a proxy for the clinical classification
#
# n = 100
#
# dur_dysp     = dict(dist='lognormal', par1=4.5, par2=4.0) # PLACEHOLDERS; INSERT SOURCE
# dur_to_peak_dys = hpu.sample(**dur_dysp, size=n)
#
# prog_rate    = 0.79 # Rate of progression of dysplasia once it is established. This parameter is used as the growth rate within a logistic function that maps durations to progression probabilities
# prog_time    = 4.4  # Point of inflection in logistic function
# mean_peaks = logf2(dur_to_peak_dys, prog_time, prog_rate)  # Apply a function that maps durations + genotype-specific progression to
#
# We want the INVERSE of this. Distribution of dur_dysp given that the peak is greater than cin3?
#
# def logf2(x, c, k):
#     '''
#     Logistic function, constrained to pass through 0,0 and with upper asymptote
#     at 1. Accepts 2 parameters: growth rate and point of inflexion.
#     '''
#     l_asymp = -1/(1+np.exp(k*c))
#     return l_asymp + 1/( 1 + np.exp(-k*(x-c)))
#
#
# def cutoff(t,c,k):
#     z = -1/(1+np.exp(k*c))
#     return (np.log(1/(t-z)-1)-k*c)/(-k)
