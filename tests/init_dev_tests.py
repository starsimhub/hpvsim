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



def test_basic(doplot=True):
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
    sim.plot()
    return sim





if __name__ == '__main__':

    sim = test_basic()
