''' Dev tests '''

import numpy as np
import sciris as sc
import pylab as pl
import sys
import os
import hpvsim as hpv

# Create genotypes - used in all subsequent tests

hpv16 = hpv.genotype('HPV16')
hpv18 = hpv.genotype('HPV18')
hpvhi5 = hpv.genotype('hpvhi5')
hpv6 = hpv.genotype('hpv6')
hpv11 = hpv.genotype('hpv11')
hpv31 = hpv.genotype('hpv31')
hpv33 = hpv.genotype('hpv33')



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
    sim = hpv.Sim(pars=pars)
    sim.run()
    sim.plot()
    return sim





if __name__ == '__main__':

    sim = test_basic()
