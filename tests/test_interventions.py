'''
Tests for single simulations
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc
import numpy as np
import hpvsim as hpv

do_plot = 1
do_save = 0


#%% Define the tests

def test_dynamic_pars():
    sc.heading('Test dynamics pars intervention')

    pars = {
        'pop_size': 10e3,
        'n_years': 10,
    }

    # Model an intervention to increase condom use
    condom_int = hpv.dynamic_pars(
        condoms=dict(timepoints=10, vals={'c': 0.9}))  # Increase condom use among casual partners to 90%

    # Model an intervention to increase the age of sexual debut
    debut_int = hpv.dynamic_pars(
        {'debut': {
            'timepoints': '2020',
            'vals': dict(f=dict(dist='normal', par1=20, par2=2.1), # Increase mean age of sexual debut
                         m=dict(dist='normal', par1=19.6,par2=1.8))
        }
        }
    )

    sim = hpv.Sim(pars=pars, interventions=[condom_int, debut_int])
    sim.run()
    return sim


def test_vaccines(do_plot=False):
    sc.heading('Test prophylactic vaccine intervention')

    hpv16 = hpv.genotype('HPV16')
    hpv18 = hpv.genotype('HPV18')

    pars = {
        'pop_size': 10e3,
        'n_years': 10,
        'genotypes': [hpv16, hpv18],
        'dt': .1,
    }

    # Model an intervention to roll out prophylactic vaccination
    def age_subtarget(sim):
        ''' Select people who are eligible for vaccination '''
        inds = sc.findinds((sim.people.age >= 9) & (sim.people.age <=14))
        return {'vals': [0.3 for _ in inds], 'inds': inds}

    bivalent_vx = hpv.vaccinate_prob(vaccine='bivalent', label='bivalent, 9-14', timepoints='2020',
                                       subtarget=age_subtarget)

    sim = hpv.Sim(pars=pars, interventions=[bivalent_vx])
    sim.run()
    if do_plot:
        sim.plot()
    return sim




#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # sim0 = test_dynamic_pars()
    sim1 = test_vaccines(do_plot=True)

    sc.toc(T)
    print('Done.')
