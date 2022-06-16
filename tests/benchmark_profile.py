'''
Benchmark the simulation
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
from test_sim import test_sim


sim = test_sim(do_plot=False)
to_profile = 'update_pre' # Must be one of the options listed below

func_options = {
    'make_contacts': hpv.make_random_contacts,
    'person':        hpv.Person.__init__,
    'make_people':   hpv.make_people,
    'init_people':   sim.init_people,
    'initialize':    sim.initialize,
    'run':           sim.run,
    'step':          sim.step,
    'get_pairs':     hpv.utils.get_discordant_pairs,
    'pair_lookup':   hpv.utils.pair_lookup_vals,
    'update_pre':    hpv.people.People.update_states_pre,
    'infect':        hpv.people.People.infect,
}

if not to_profile.startswith('plot'):
    sc.profile(run=sim.run, follow=func_options[to_profile])
else:
    sim.run()
    sc.profile(sim.plot, follow=func_options[to_profile])