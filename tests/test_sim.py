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
        # 'n_agents': 500,
        # 'start': 2020,
        # 'stop': 2025,
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


def test_epi():
    sc.heading('Test basic epi dynamics')

    # Define baseline parameters and initialize sim
    base_pars = dict(n_agents=3e3, start=2000, stop=2020, dt=6, genotypes=[16], beta=0.02, verbose=0)  #, eff_condoms=0.6)
    sim = hpv.Sim(pars=base_pars)
    sim.init()

    # Define the parameters to vary
    class ParEffects:
        def __init__(self, par, range, variables):
            self.par = par
            self.range = range
            self.variables = variables
            return

    par_effects = [
        ParEffects('beta',          [0.01, 0.99],   ['infections']),
        # ParEffects('condoms',       [0.90, 0.10],   ['infections']),
        # ParEffects('acts',          [1, 200],       ['infections']),
        # ParEffects('debut',         [25, 15],       ['infections']),
        ParEffects('init_prev',     [0.1, 0.8],     ['infections']),
    ]

    # Loop over each of the above parameters and make sure they affect the epi dynamics in the expected ways
    for par_effect in par_effects:
        if par_effect.par=='acts':
            bp = sc.dcp(sim[par_effect.par]['c'])
            lo = {lk:{**bp, 'par1': par_effect.range[0]} for lk in ['m','c']}
            hi = {lk:{**bp, 'par1': par_effect.range[1]} for lk in ['m','c']}
        elif par_effect.par=='condoms':
            lo = {lk:par_effect.range[0] for lk in ['m','c']}
            hi = {lk:par_effect.range[1] for lk in ['m','c']}
        elif par_effect.par=='debut':
            bp = sc.dcp(sim[par_effect.par]['f'])
            lo = {sk:{**bp, 'par1':par_effect.range[0]} for sk in ['f','m']}
            hi = {sk:{**bp, 'par1':par_effect.range[1]} for sk in ['f','m']}
        else:
            lo = par_effect.range[0]
            hi = par_effect.range[1]

        pars0 = sc.mergedicts(base_pars, {par_effect.par: lo})  # Use lower parameter bound
        pars1 = sc.mergedicts(base_pars, {par_effect.par: hi})  # Use upper parameter bound

        # Run the simulations and pull out the results
        s0 = hpv.Sim(pars0, label=f'{par_effect.par} {par_effect.range[0]}').run()
        s1 = hpv.Sim(pars1, label=f'{par_effect.par} {par_effect.range[1]}').run()

        # Check results
        # TODO: change hpv16 to hpv
        for var in par_effect.variables:
            v0 = s0.results.hpv16[var][:].sum()
            v1 = s1.results.hpv16[var][:].sum()
            print(f'Checking {var:10s} with varying {par_effect.par:10s} ... ', end='')
            assert v0 <= v1, f'Expected {var} to be lower with {par_effect.par}={lo} than with {par_effect.par}={hi}, but {v0} > {v1})'
            print(f'✓ ({v0} <= {v1})')

    return s0, s1


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim0 = test_microsim()
    sim = test_sim_options()
    s0, s1 = test_epi()

    sc.toc(T)
    print('Done.')