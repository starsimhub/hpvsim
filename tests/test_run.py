'''
Tests for run options (multisims and scenarios)
'''

#%% Imports and settings
import os
import numpy as np
import sciris as sc
import hpvsim as hpv

do_plot = 1
do_save = 0
debug   = 1
verbose = 0
n_agents = 1000
hpv.options.set(interactive=False) # Assume not running interactively


#%% Define the tests

def test_singlerun():
    sc.heading('Single run test')

    iterpars = {'beta': 0.035}
    sim = hpv.Sim(verbose=verbose)
    sim['n_years'] = 10
    sim['dt'] = 0.5
    sim['n_agents'] = 1000
    sim = hpv.single_run(sim=sim, **iterpars)

    return sim


def test_multirun(do_plot=do_plot): # If being run via pytest, turn off
    sc.heading('Multirun test')

    n_years = 20

    # Method 1 -- Note: this runs 3 simulations, not 3x3!
    iterpars = {'beta': [0.015, 0.025, 0.035],
                'hpv_control_prob': [0.0, 0.5, 1.0],
                }
    sim = hpv.Sim(n_years=n_years, n_agents=n_agents)
    sims = hpv.multi_run(sim=sim, iterpars=iterpars, verbose=verbose)

    # Method 2 -- run a list of sims
    simlist = []
    for i in range(len(iterpars['beta'])):
        sim = hpv.Sim(n_years=n_years, n_agents=n_agents, beta=iterpars['beta'][i], hpv_control_prob=iterpars['hpv_control_prob'][i])
        simlist.append(sim)
    sims2 = hpv.multi_run(sim=simlist, verbose=verbose)

    # Method 3 -- shortcut for parallelization
    s1 = hpv.Sim(n_years=n_years, n_agents=n_agents)
    s2 = s1.copy()
    s1,s2 = hpv.parallel(s1, s2).sims
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    # Run in serial for debugging
    hpv.multi_run(sim=hpv.Sim(n_years=n_years, n_agents=n_agents), n_runs=2, parallel=not(debug))

    if do_plot:
        for sim in sims + sims2:
            sim.plot()

    return sims


def test_multisim_reduce(do_plot=do_plot): # If being run via pytest, turn off
    sc.heading('Combine results test')

    n_runs = 3
    init_hpv_prev = 0.1

    sim = hpv.Sim(n_agents=n_agents, init_hpv_prev=init_hpv_prev)
    msim = hpv.MultiSim(sim, n_runs=n_runs, noise=0.1)
    msim.run(verbose=verbose, reduce=True)

    if do_plot:
        msim.plot()

    return msim


def test_multisim_combine(do_plot=do_plot): # If being run via pytest, turn off
    sc.heading('Combine results test')

    n_runs = 3
    init_hpv_prev = 0.1

    print('Running first sim...')
    sim = hpv.Sim(n_agents=n_agents, init_hpv_prev=init_hpv_prev, verbose=verbose)
    msim = hpv.MultiSim(sim)
    msim.run(n_runs=n_runs, keep_people=True)
    # sim1 = msim.combine(output=True) #CURRENTLY BROKEN
    # assert sim1['n_agents'] == n_agents*n_runs #CURRENTLY BROKEN

    print('Running second sim, results should be similar but not identical (stochastic differences)...')
    sim2 = hpv.Sim(n_agents=n_agents*n_runs, init_hpv_prev=init_hpv_prev)
    sim2.run(verbose=verbose)

    if do_plot:
        msim.plot()
        sim2.plot()

    return msim


def test_multisim_advanced():
    sc.heading('Advanced multisim options')

    # Settings
    msim_path = 'msim_test.msim'

    # Creat the sims/msims
    sims = sc.objdict()
    for i in range(4):
        sims[f's{i}'] = hpv.Sim(label=f'Sim {i}', n_agents=n_agents, beta=0.01*i)

    m1 = hpv.MultiSim(sims=[sims.s0, sims.s1])
    m2 = hpv.MultiSim(sims=[sims.s2, sims.s3])

    # Test methods
    m1.init_sims()
    m1.run()
    m2.run(reduce=True)
    m1.reduce()
    m1.mean()
    m1.median()
    m1.shrink()
    m1.disp()
    m1.summarize()
    m1.brief()

    # Check save/load
    m1.save(msim_path)
    m1b = hpv.MultiSim.load(msim_path)
    assert np.allclose(m1.summary[:], m1b.summary[:], rtol=0, atol=0, equal_nan=True)
    os.remove(msim_path)

    # Check merging/splitting
    merged1 = hpv.MultiSim.merge(m1, m2)
    merged2 = hpv.MultiSim.merge([m1, m2], base=True)
    m1c, m2c = merged1.split()
    m1d, m2d = merged1.split(chunks=[2,2])

    return merged1, merged2


def test_simple_scenarios(do_plot=do_plot):
    sc.heading('Simple scenarios test')
    basepars = {'n_agents':n_agents}

    json_path = 'scen_test.json'
    xlsx_path = 'scen_test.xlsx'

    scens = hpv.Scenarios(basepars=basepars)
    scens.run(verbose=verbose, parallel=not(debug))
    if do_plot:
        scens.plot()
    scens.to_json(json_path)
    scens.to_excel(xlsx_path)
    scens.disp()
    scens.summarize()
    scens.brief()

    for path in [json_path, xlsx_path]:
        print(f'Removing {path}')
        os.remove(path)

    return scens


def test_complex_scenarios(do_plot=do_plot, do_save=False, fig_path=None):
    sc.heading('Test impact of changing sexual behavior')

    n_runs = 3
    base_pars = {
      'n_agents': n_agents,
      'network': 'default',
      }

    base_sim = hpv.Sim(base_pars) # create sim object
    base_sim['n_years'] = 30

    # Define the scenarios
    scenarios = {
        'default': {
            'name': 'Default sexual behavior',
            'pars': {
            }
        },
        'high': {
            'name': 'Higher-risk sexual behavior',
            'pars': {
                'acts': dict(m=dict(dist='neg_binomial', par1=120, par2=40),
                             c=dict(dist='neg_binomial', par1=20, par2=5),
                             ),
                'condoms': dict(m=0, c=0.1),
                'debut': dict(f=dict(dist='normal', par1=14, par2=2),
                              m=dict(dist='normal', par1=14, par2=2))
            }
        },
        'low': {
            'name': 'Lower-risk sexual behavior',
            'pars': {
                'acts': dict(m=dict(dist='neg_binomial', par1=40, par2=10),
                             c=dict(dist='neg_binomial', par1=2, par2=1),
                             ),
                'condoms': dict(m=0.5, c=0.9),
                'debut': dict(f=dict(dist='normal', par1=20, par2=2),
                              m=dict(dist='normal', par1=21, par2=2))
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = hpv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)
    scens.compare()

    if do_plot:
        scens.plot(do_save=do_save, fig_path=fig_path)

    return scens


def test_sweeps(do_plot=do_plot):
    sc.heading('Test sweeps')
    sim = hpv.Sim(dt=1.0)
    sweep = hpv.Sweep(base_sim=sim,
                      sweep_pars={'beta': [0.01, 0.1], 'hpv_control_prob': [0, 0.5]},
                      sweep_vars=['cancers', 'infections'],
                      n_draws=4)
    sweep.run(reduce=True, from_year=2020)
    if do_plot:
        sweep.plot_heatmap(zscales=[1,1e6])

    return sweep


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    hpv.options.set(interactive=do_plot)
    T = sc.tic()

    sim1   = test_singlerun()
    sims2  = test_multirun(do_plot=do_plot)
    msim1  = test_multisim_reduce(do_plot=do_plot)
    msim2  = test_multisim_combine(do_plot=do_plot) #CURRENTLY PARTIALLY BROKEN
    m1,m2  = test_multisim_advanced()
    scens1 = test_simple_scenarios(do_plot=do_plot)
    scens2 = test_complex_scenarios(do_plot=do_plot)
    sweep = test_sweeps(do_plot=True)

    sc.toc(T)
    print('Done.')


