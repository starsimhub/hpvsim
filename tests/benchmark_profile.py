'''
Benchmark the simulation
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
from test_sim import test_sim


sim = test_sim(do_plot=False)
to_profile = 'apply_int' # Must be one of the options listed below

func_options = {
    'make_contacts': hpv.make_contacts,
    'person':        hpv.Person.__init__,
    'make_people':   hpv.make_people,
    'init_people':   sim.init_people,
    'initialize':    sim.initialize,
    'run':           sim.run,
    'step':          sim.step,
    'apply_int':     hpv.interventions.vaccinate_num.apply,
    'vaccinate':     hpv.interventions.vaccinate_num.vaccinate,
    'select_people': hpv.interventions.vaccinate_num.select_people,
    'get_pairs':     hpv.utils.get_discordant_pairs,
    'pair_lookup':   hpv.utils.pair_lookup_vals,
    'update_pre':    hpv.people.People.update_states_pre,
    'infect':        hpv.people.People.infect,
}

sc.profile(run=sim.run, follow=func_options[to_profile])
