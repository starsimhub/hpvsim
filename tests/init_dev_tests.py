''' Dev tests '''

import numpy as np
import sciris as sc
import pylab as pl
import sys
import os

# Add module to paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def test_random():
    ''' Make the simplest possible sim with one kind of partnership '''
    from hpvsim.sim import Sim
    from hpvsim.analysis import snapshot
    pars = {'pop_size':20_000, 'rand_seed':100, 'location':'zimbabwe'}
    sim = Sim(pars=pars, analyzers=snapshot('2015', '2020'))
    sim.run()
    return sim

def test_basic():
    ''' Make a sim with two kinds of partnership, regular and casual '''
    from hpvsim.sim import Sim
    pars = {'network':'basic'}
    sim = Sim(pars=pars)
    sim.run()
    return sim


def test_genotypes():
    ''' Make a sim with two kinds of partnership, regular and casual and 2 HPV genotypes'''
    from hpvsim.sim import Sim
    from hpvsim.immunity import genotype

    hpv16 = genotype('HPV16')
    hpv18 = genotype('HPV18')
    pars = {
        'pop_size': 20e3,
        'network': 'basic',
        'genotypes': [hpv16, hpv18],
        'dt': .2,
        'end': 2025
    }
    sim = Sim(pars=pars)
    sim.run()

    # fig, ax = pl.subplots(2, 2, figsize=(10, 10))
    # timevec = sim.results['year']
    #
    # for i, genotype in sim['genotype_map'].items():
    #     ax[0,0].plot(timevec, sim.results['genotype']['hpv_incidence_by_genotype'].values[i,:], label=genotype)
    #     ax[0,1].plot(timevec, sim.results['genotype']['hpv_prevalence_by_genotype'].values[i, :])
    #     ax[1,0].plot(timevec, sim.results['genotype']['new_precancers_by_genotype'].values[i,:])
    #     ax[1,1].plot(timevec, sim.results['genotype']['cin_prevalence_by_genotype'].values[i, :])
    #
    # ax[0,0].legend()
    # ax[0,0].set_title('HPV incidence by genotype')
    # ax[0,1].set_title('HPV prevalence by genotype')
    # ax[1,0].set_title('New CIN by genotype')
    # ax[1,1].set_title('CIN prevalence by genotype')
    # fig.show()
    return sim



if __name__ == '__main__':

    # sim0 = test_random() # NOT WORKING
    # sim1 = test_basic() # NOT WORKING
    # sim2 = test_genotypes()

    # @nb.jit(parallel=True)
    # def isin(arr, vals):
    #     n = len(arr)
    #     result = np.full(n, False)
    #     set_vals = set(vals)
    #     for i in nb.prange(n):
    #         if arr[i] in set_vals:
    #             result[i] = True
    #     return result


    import numpy as np
    import numba as nb

    @nb.jit(parallel=True)
    def isinvals_cheating(arr, vals):
        n = len(arr)
        result = np.full(n, False)
        result_vals = np.full(n, np.nan)
        set_vals = set(vals[0,:])
        list_vals = list(vals[0,:])
        for i in nb.prange(n):
            if arr[i] in set_vals:
                ind = 0 #list_vals.index(arr[i]) ## THIS LINE IS WAY TOO SLOW
                result[i] = True
                result_vals[i] = vals[1,ind]
        return result, result_vals


    N = int(1e5)
    M = int(20e3)
    num_arr = 100e3
    num_vals = 20e3
    num_types = 6
    arr = np.random.randint(0, num_arr, N)
    vals_col1 = np.random.randint(0, num_vals, M)
    vals_col2 = np.random.randint(0, num_types, M)
    vals = np.array([vals_col1, vals_col2])

    result, result_vals = isinvals_cheating(arr,vals)