'''
Benchmark the simulation
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc
import numpy as np

# Add module to paths and import hpvsim
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from test_sim import test_sim
import hpvsim.people as hpp
import hpvsim.population as hppop
import hpvsim.base as hpb
import hpvsim.utils as hpu


sim = test_sim(do_plot=False)
to_profile = 'step' # Must be one of the options listed below

func_options = {
    'make_contacts': hppop.make_random_contacts,
    'person':        hpb.Person.__init__,
    'make_people':   hppop.make_people,
    'init_people':   sim.init_people,
    'initialize':    sim.initialize,
    'run':           sim.run,
    'step':          sim.step,
    'get_pairs':     hpu.get_discordant_pairs,
    'pair_lookup':   hpu.pair_lookup_vals,
    'infect':        hpp.People.infect,
}

if not to_profile.startswith('plot'):
    sc.profile(run=sim.run, follow=func_options[to_profile])
else:
    sim.run()
    sc.profile(sim.plot, follow=func_options[to_profile])