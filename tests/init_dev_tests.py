''' Dev tests '''

import numpy as np
import sciris as sc
import pylab as pl
import sys
import os

# Add module to paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))



if __name__ == '__main__':

    from hpvsim.sim import Sim
    pars = {}#'network':'basic'}
    sim = Sim(label='test1', pars=pars)
    print(sim.label)
    sim.run()

    # active_people = sim.people.filter(sim.people.age>sim.people.debut)



