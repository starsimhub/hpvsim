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
    hpvhi5 = genotype('hpvhi5')
    pars = {
        'pop_size': 100e3,
        'network': 'basic',
        'genotypes': [hpv16, hpv18, hpvhi5],
        'dt': .1,
        'end': 2035
    }
    sim = Sim(pars=pars)
    sim.run()

    fig, ax = pl.subplots(2, 2, figsize=(10, 10))
    timevec = sim.results['year']

    ax[0, 1].plot(timevec, sim.results['hpv_prevalence'].values)
    ax[1, 0].plot(timevec, sim.results['cin1_prevalence'].values, label='CIN1')
    ax[1, 0].plot(timevec, sim.results['cin2_prevalence'].values, label='CIN2')
    ax[1, 0].plot(timevec, sim.results['cin3_prevalence'].values, label='CIN3')
    ax[1,1].plot(timevec, sim.results['cancer_incidence'].values)
    for i, genotype in sim['genotype_map'].items():
        ax[0,0].plot(timevec, sim.results['hpv_incidence_by_genotype'].values[i,:], label=genotype)

    ax[0,0].legend()
    ax[1,0].legend()
    ax[0,0].set_title('HPV incidence by genotype')
    ax[0,1].set_title('HPV prevalence')
    ax[1,0].set_title('CIN prevalence')
    ax[1,1].set_title('Cancer incidence')
    fig.show()
    return sim


def test_interventions():
    ''' Template to develop tests for interventions'''
    from hpvsim.sim import Sim
    from hpvsim.immunity import genotype
    import hpvsim.interventions as hpint

    hpv16 = genotype('HPV16')
    hpv18 = genotype('HPV18')
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


if __name__ == '__main__':

    # sim0 = test_random() # NOT WORKING
    # sim1 = test_basic() # NOT WORKING
    sim2 = test_genotypes()
    # sim3 = test_interventions()





