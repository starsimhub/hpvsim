'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
import pytest

do_plot = 1
do_save = 0


# % Define the tests

def test_microsim():
    sc.heading('Minimal sim test')

    # Define baseline parameters and initialize sim
    pars = {
        'n_agents': 500,
        'start': 2020,
        'stop': 2025,
    }
    sim = hpv.Sim(**pars)
    sim.run()
    return sim


def test_sim_options():
    sc.heading('Test equivalency of different ways of specifying sim parameters')

    # Option 1: flat par dict with a mixture of pars that belong in different modules
    pars = dict(
        start=2020,  # Sim par
        beta_m2f=0.05,  # HPV genotype par, applied to all genotypes
        dur_cancer=10,
        # prop_f0=0.45,
        cross_imm_med=0.7
    )
    pars['16'] = dict(dur_cin=4)  # HPV genotype-specific pars
    pars['18'] = dict(dur_cin=5)  # HPV genotype-specific pars

    sim = hpv.Sim(pars)
    sim.run()
    assert sim.pars.hpv16.beta_m2f == pars['beta_m2f']
    assert sim.pars.hpv16.dur_cin.pars['mean'] == pars['16']['dur_cin']

    # Option 2: pars in separate dictionaries
    sim_pars = dict(start=2020)
    hpv_pars = dict(beta_m2f=0.05, hpv16=dict(dur_cin=4))
    sim = hpv.Sim(sim_pars=sim_pars, hpv_pars=hpv_pars)
    sim.init()
    assert sim.pars.hpv16.beta_m2f == hpv_pars['beta_m2f']
    assert sim.pars.hpv16.dur_cin.pars['mean'] == hpv_pars['hpv16']['dur_cin']

    # Option 3: construct from scratch
    hpv16 = hpv.HPVType(genotype='hpv16', beta_m2f=0.05, dur_cin=4)
    hpv18 = hpv.HPVType(genotype='hpv18', beta_m2f=0.05, dur_cin=5)
    hpv_connector = hpv.HPV(genotypes=[hpv16, hpv18])
    sim = hpv.Sim(genotypes=[hpv16, hpv18], connectors=hpv_connector)
    sim.init()
    assert sim.pars.hpv16.beta_m2f == hpv16.pars['beta_m2f']
    assert sim.pars.hpv16.dur_cin.pars['mean'] == hpv16.pars['dur_cin'].pars['mean']

    return sim


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim0 = test_microsim()
    sim = test_sim_options()

    sc.toc(T)
    print('Done.')