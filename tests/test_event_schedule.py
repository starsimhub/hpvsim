'''
Tests for event scheduler
'''

import sciris as sc
import numpy as np
import hpvsim as hpv
import functools

class simple_vaccinate_prob(hpv.vaccinate_prob):
    # Simplified vaccination routine that delivers vaccine to everyone every day with probability prob
    def select_people(self, sim):
        vacc_probs = np.zeros(len(sim.people))
        vacc_probs[sim.people.alive & ~sim.people.vaccinated] = self.prob  # Assign equal vaccination probability to everyone
        print(self.prob)
        return hpv.true(hpv.binomial_arr(vacc_probs))  # Calculate who actually gets vaccinated


def test_schedule_scaleup():
    sc.heading('Test dynamics pars intervention')

    pars = {
        'n_agents': 1e5,
        'n_years': 50,
    }

    vx = simple_vaccinate_prob(vaccine='bivalent', label='vaccine_intervention', timepoints=0, prob=0)

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
