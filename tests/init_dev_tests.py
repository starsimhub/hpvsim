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


def test_basic(doplot=False):
    ''' Make a sim with two kinds of partnership, regular and casual and 2 HPV genotypes'''

    pars = {
        'pop_size': 10e3,
        'network': 'basic',
        'genotypes': [hpv16, hpv18, hpvhi5],
        'dt': .1,
        'end': 2035
    }
    sim = Sim(pars=pars, location='zimbabwe')
    sim.run()

    if doplot:
        fig, ax = pl.subplots(2, 2, figsize=(10, 10))
        timevec = sim.results['year']

        for i, genotype in sim['genotype_map'].items():
            ax[0,0].plot(timevec, sim.results['genotype']['hpv_incidence_by_genotype'].values[i,:], label=genotype)
            ax[0,1].plot(timevec, sim.results['genotype']['hpv_prevalence_by_genotype'].values[i, :])
            ax[1,0].plot(timevec, sim.results['genotype']['new_precancers_by_genotype'].values[i,:])
            ax[1,1].plot(timevec, sim.results['genotype']['cin_prevalence_by_genotype'].values[i, :])

        ax[0,0].legend()
        ax[0,0].set_title('HPV incidence by genotype')
        ax[0,1].set_title('HPV prevalence by genotype')
        ax[1,0].set_title('New CIN by genotype')
        ax[1,1].set_title('CIN prevalence by genotype')
        fig.show()
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


if __name__ == '__main__':

    # sim0 = test_random()
    sim1 = test_basic()
    # sim2 = test_interventions()
