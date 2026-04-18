'''
Tests for event scheduler
'''

import sciris as sc
import hpvsim as hpv
import functools


def test_schedule_scaleup():
    sc.heading('Test dynamics pars intervention')

    pars = {
        'n_agents': 10e3,
        'n_years': 50,
    }

    vx = hpv.BaseVaccination(product='bivalent', prob=0, label='vaccine_intervention')

    schedule = hpv.EventSchedule()
    schedule[10] = functools.partial(hpv.set_intervention_attributes, intervention_name='vaccine_intervention', prob=0.05)
    schedule['2030-01-01'] = functools.partial(hpv.set_intervention_attributes, intervention_name='vaccine_intervention', prob=0.2)
    schedule[2040.0] = functools.partial(hpv.set_intervention_attributes, intervention_name='vaccine_intervention', prob=0.4)
    schedule[2045.0] = functools.partial(hpv.set_intervention_attributes, intervention_name='vaccine_intervention', prob=0.6)

    sim = hpv.Sim(pars=pars, interventions=[vx, schedule])
    sim.run()

    sim.plot('new_doses')
    return sim


#%% Run as a script
if __name__ == '__main__':
    sim  = test_schedule_scaleup()
