''' Dev tests '''

import numpy as np
import sciris as sc
import pylab as pl
import sys
import os

# Add module to paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Create genotypes - used in all subsequent tests
from hpvsim.sim import Sim
from hpvsim.immunity import genotype
hpv16 = genotype('HPV16')
hpv18 = genotype('HPV18')
hpvhi5 = genotype('hpvhi5')
hpv6 = genotype('hpv6')
hpv11 = genotype('hpv11')
hpv31 = genotype('hpv31')
hpv33 = genotype('hpv33')


def test_random():
    ''' Make the simplest possible sim with one kind of partnership '''
    from hpvsim.analysis import snapshot
    pars = {'pop_size':20_000,
            'rand_seed':100,
            'location':'zimbabwe',
            'genotypes': [hpv16, hpv18],
            }
    sim = Sim(pars=pars, analyzers=snapshot('2015', '2020'))
    sim.run()
    return sim


def test_basic(test_results=True):
    ''' Make a sim with two kinds of partnership, regular and casual and 2 HPV genotypes'''

    pop_size = 50e3
    pars = {
        'pop_size': pop_size,
        'network': 'basic',
        'genotypes': [hpv16, hpv18],#, hpv6],#, hpv11, hpv31, hpv33],
        'dt': .1,
        'end': 2035
    }
    sim = Sim(pars=pars)
    sim.run()

    return sim


def test_interventions():
    ''' Template to develop tests for interventions'''
    import hpvsim.interventions as hpint

    pars = {
        'network': 'basic',
        'genotypes': [hpv16, hpv18],
    }

    # Model an intervention to increase condom use
    condom_int = hpint.dynamic_pars(
        condoms=dict(timepoints=10, vals={'c': 0.9}))  # Increase condom use among casual partners to 90%

    # Model an intervention to increase the age of sexual debut
    debut_int = hpint.dynamic_pars(
        {'debut': {
            'timepoints': '2020',
            'vals': dict(f=dict(dist='normal', par1=20, par2=2.1), # Increase mean age of sexual debut
                         m=dict(dist='normal', par1=19.6,par2=1.8))
        }
        }
    )

    sim = Sim(pars=pars, interventions=[condom_int, debut_int])
    sim.run()
    return sim


def test_init_states():
    ''' Make a sim with two kinds of partnership, regular and casual and 2 HPV genotypes'''


    pop_size = 50e3
    pars = {
        'pop_size': pop_size,
        'init_hpv_prev': 0.08,
        'network': 'basic',
        'genotypes': [hpv16, hpv18],#, hpv6],#, hpv11, hpv31, hpv33],
        'dt': .2,
        'end': 2025
    }
    sim = Sim(pars=pars)
    sim.run()

    return sim


if __name__ == '__main__':

    # sim0 = test_random()
    sim = test_init_states()
    # to_plot = []
    # sim1.plot()
    # sim2 = test_interventions()
