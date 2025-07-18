'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
import starsim as ss
import pytest

do_plot = 1
do_save = 0


# % Define the tests

def test_microsim():
    sc.heading('Minimal sim test')
    sim = hpv.Sim()
    sim.run()
    return sim


def test_sim_options():
    sc.heading('Test equivalency of different ways of specifying sim parameters')

    # Option 1: flat par dict with a mixture of pars that belong in different modules
    pars = dict(
        start=2020,  # Sim par
        beta_m2f=0.05,  # HPV genotype par, applied to all genotypes
        dur_cancer=10,
        prop_f0=0.45,
        location='india',
        cross_imm_sus_med=0.7,
        genotypes=[16, 18],  # HPV genotype list
    )
    pars['16'] = dict(dur_cin=4)  # HPV genotype-specific pars
    pars['18'] = dict(dur_cin=5)  # HPV genotype-specific pars

    sim = hpv.Sim(pars)
    sim.init()
    assert sim.pars.hpv16.beta_m2f == pars['beta_m2f']
    assert sim.pars.hpv16.dur_cin.pars['mean'] == pars['16']['dur_cin']

    # Option 2: pars in separate dictionaries
    sim_pars = dict(start=2020)
    hpv_pars = dict(beta_m2f=0.05, hpv16=dict(dur_cin=4))
    nw_pars = dict(prop_f0=0.45)
    sim = hpv.Sim(sim_pars=sim_pars, hpv_pars=hpv_pars, nw_pars=nw_pars)
    sim.init()
    assert sim.pars.hpv16.beta_m2f == hpv_pars['beta_m2f']
    assert sim.pars.hpv16.dur_cin.pars['mean'] == hpv_pars['hpv16']['dur_cin']

    # Option 3: construct from scratch
    hpv16 = hpv.Genotype(genotype='hpv16', beta_m2f=0.05, dur_cin=4)
    hpv18 = hpv.Genotype(genotype='hpv18', beta_m2f=0.05, dur_cin=5)
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
        ParEffects('beta',          [0.01, 0.99],   ['new_infections']),
        # ParEffects('condoms',       [0.90, 0.10],   ['infections']),
        # ParEffects('acts',          [1, 200],       ['infections']),
        # ParEffects('debut',         [25, 15],       ['infections']),
        ParEffects('init_prev',     [0.1, 0.8],     ['new_infections']),
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


def test_states():
    sc.heading('Test states')

    # Define baseline parameters and initialize sim
    base_pars = dict(start=2020, dt=6, beta=0.05)

    class check_states(ss.Analyzer):

        def __init__(self):
            super().__init__()
            self.okay = True
            return

        def step(self):
            """
            Checks states that should be mutually exlusive and collectively exhaustive
            """
            hpvc = self.sim.connectors.hpv  # HPV connector
            gtypes = hpvc.genotypes.values()

            for gtype in gtypes:

                # Infection states: people must be exactly one of susceptible/infectious/inactive
                s1 = (gtype.susceptible | gtype.infectious | gtype.inactive).all()
                if not s1:
                    errormsg = (f'States {{susceptible, infectious, inactive}} should be collectively exhaustive '
                                f'but are not for genotype {gtype.genotype}.')
                    raise ValueError(errormsg)
                s2 = ~(gtype.susceptible & gtype.infected).any()
                if not s2:
                    errormsg = (f'States {{susceptible, infected}} should be mutually exclusive '
                                f'but are not for genotype {gtype.genotype}.')
                    raise ValueError(errormsg)
                s3 = ~(gtype.susceptible & gtype.inactive).any()
                if not s3:
                    errormsg = (f'States {{susceptible, inactive}} should be mutually exclusive '
                                f'but are not for genotype {gtype.genotype}.')
                    raise ValueError(errormsg)
                s4 = ~(gtype.infectious & gtype.inactive).any()
                if not s4:
                    raise ValueError('States {infectious, inactive} should be mutually exclusive but are not.')

                # Dysplasia states:
                #   - people *without active infection* should not be in any dysplasia state (test d0)
                #   - people *with active infection* should be in exactly one dysplasia state (test d1)
                #   - people should either have no cellular changes (normal) or be in a dysplasia state (tests d2-d6)
                d0 = (~((~gtype.infectious) & gtype.cin)).any()
                if not d0:
                    raise ValueError('People without active infection should not have detectable cell changes')
                d1 = (gtype.susceptible | gtype.precin | gtype.cin | gtype.cancerous | gtype.dead_cancer).all()
                if not d1:
                    errormsg = (f'States {{precin, cin, cancerous, dead_cancer}} should be collectively exhaustive '
                                f'but are not for genotype {gtype.name}.')
                    raise ValueError(errormsg)
                d2 = ~(gtype.precin & gtype.cin).all()
                if not d2:
                    raise ValueError('States {precin, cin} should be mutually exclusive but are not.')
                d3 = ~(gtype.cin & gtype.cancerous).all()
                if not d3:
                    raise ValueError('States {cin, cancerous} should be mutually exclusive but are not.')

                # If there's anyone with abnormal cells & inactive infection, they must have cancer
                sd1inds = (gtype.abnormal & gtype.inactive).uids
                sd1 = True
                if len(sd1inds) > 0:
                    hpvc.cancerous[sd1inds].any()
                if not sd1:
                    raise ValueError('Anyne with abnormal cells and inactive infection should have cancer.')

                checkall = np.array([
                    s1, s2, s3, s4,
                    d0, d1, d2, d3,
                    sd1
                ])
                if not checkall.all():
                    self.okay = False

            return

    sim = hpv.Sim(pars=base_pars, analyzers=check_states())
    sim.run()
    a = sim.analyzers[0]
    assert a.okay

    return sim


def test_result_consistency():
    sc.heading('Check that results by genotype sum to the correct totals ')

    # Create sim
    n_agents = 1e3
    sim = hpv.Sim(n_agents=n_agents, stop=2030, dt=6, label='test_results')
    sim.run()

    # Check results by genotype sum up to the correct totals
    res_to_check = ['new_infections', 'n_infected', 'n_cin', 'n_cancerous']
    for res in res_to_check:
        gresults = np.array([gtype.results[res][:] for gtype in sim.genotypes.values()])
        print(f"Checking {res} ... ")
        by_gen = gresults.sum(axis=0)
        total = sim.results.hpv[res][:]
        assert (total <= by_gen).all(), f'{res} by genotype should be exceed total, but {by_gen}<{total}'
        print(f"✓ ({total.sum():.2f} < {by_gen.sum():.2f})")

    # Check that males don't have CINs or cancers
    males = sim.people.male
    males_with_cin = (males & sim.people.hpv.ti_cin.notnan).uids
    males_with_cancer = (males & sim.people.hpv.ti_cancer.notnan).uids
    assert len(males_with_cin) == 0, 'Should not have males with CINs'
    assert len(males_with_cancer) == 0, 'Should not have males with cancerss'
    print(f"✓ (No males with CINs or cancers)")

    # # Check that people younger than debut don't have HPV
    # virgin_inds = (sim.people.is_virgin).nonzero()[-1]
    # virgins_with_hpv = (~np.isnan(sim.people.date_infectious[:,virgin_inds])).nonzero()[-1]
    # assert len(virgins_with_hpv)==0

    return sim


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # sim0 = test_microsim()
    sim = test_sim_options()
    # s0, s1 = test_epi()
    # sim3 = test_states()
    # sim4 = test_result_consistency()

    sc.toc(T)
    print('Done.')