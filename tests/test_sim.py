'''
Tests for single simulations
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
import hpvsim.sim as hps

do_plot = 1
do_save = 0


#%% Define the tests

def test_microsim():
    sc.heading('Minimal sim test')

    sim = hps.Sim()
    pars = {
        'pop_size': 10,
        'init_hpv_prev': .1,
        'n_years': 2,
        }
    sim.update_pars(pars)
    sim.run()
    sim.disp()
    sim.summarize()
    sim.brief()

    return sim


def test_sim(do_plot=False, do_save=False): # If being run via pytest, turn off
    sc.heading('Basic sim test')

    # Settings
    seed = 1
    verbose = 1

    # Create and run the simulation
    sim = hps.Sim(end=2035)
    sim.set_seed(seed)
    sim.run(verbose=verbose)

    # Optionally plot
    if do_plot:
        sim.plot(do_save=do_save)

    return sim


def test_epi():
    sc.heading('Test basic epi dynamics')

    # Define baseline parameters and initialize sim
    base_pars = dict(n_years=10, dt=0.5)
    sim = hps.Sim()

    # Define the parameters to vary
    vary_pars   = ['beta',          'acts',         'condoms',      'debut',        'rel_cin1_prob',    'init_hpv_prev'] # Parameters
    vary_vals   = [[0.05, 0.5],     [10,200],       [0.1,0.9],      [15,25],        [0.1, 2],           [0.01,0.5]] # Values
    vary_rels   = ['pos',           'pos',          'neg',          'neg',          'pos',              'pos'] # Expected association with epi outcomes
    vary_what   = ['infections',    'infections',   'infections',   'infections',   'cin1s',            'cin1s'] # Epi outcomes to check

    # Loop over each of the above parameters and make sure they affect the epi dynamics in the expected ways
    for vpar,vval,vrel,vwhat in zip(vary_pars, vary_vals, vary_rels, vary_what):
        if vpar=='acts':
            bp = sc.dcp(sim[vpar]['a'])
            lo = {'a':{**bp, 'par1': vval[0]}}
            hi = {'a':{**bp, 'par1': vval[1]}}
        elif vpar=='condoms':
            lo = {'a':vval[0]}
            hi = {'a':vval[1]}
        elif vpar=='debut':
            bp = sc.dcp(sim[vpar]['f'])
            lo = {sk:{**bp, 'par1':vval[0]} for sk in ['f','m']}
            hi = {sk:{**bp, 'par1':vval[1]} for sk in ['f','m']}
        else:
            lo = vval[0]
            hi = vval[1]

        pars0 = sc.mergedicts(base_pars, {vpar:lo}) # Use lower parameter bound
        pars1 = sc.mergedicts(base_pars, {vpar:hi}) # Use upper parameter bound

        # Run the simulations and pull out the results
        s0 = hps.Sim(pars0, label=f'{vpar} {vval[0]}').run()
        s1 = hps.Sim(pars1, label=f'{vpar} {vval[1]}').run()
        res0 = s0.summary
        res1 = s1.summary

        # Check results
        key='cum_total_'+vwhat
        v0 = res0[key]
        v1 = res1[key]
        print(f'Checking {key:20s} ... ', end='')
        if vrel=='pos':
            assert v0 <= v1, f'Expected {key} to be lower with {vpar}={lo} than with {vpar}={hi}, but {v0} > {v1})'
            print(f'✓ ({v0} <= {v1})')
        elif vrel=='neg':
            assert v0 >= v1, f'Expected {key} to be higher with {vpar}={lo} than with {vpar}={hi}, but {v0} < {v1})'
            print(f'✓ ({v0} => {v1})')

    return


def test_flexible_inputs():
    sc.heading('Testing flexibility of sim inputs')

    # Test resetting layer parameters
    sim = hps.Sim(pop_size=100, label='test_label')
    sim.reset_layer_pars()
    sim.initialize()
    sim.reset_layer_pars()

    # Test validation
    sim['pop_size'] = 'invalid'
    with pytest.raises(ValueError):
        sim.validate_pars()
    sim['pop_size'] = 100 # Restore

    # Handle missing start
    sim['start'] = None
    sim.validate_pars()

    # Can't have an end before the start
    sim['end'] = 2014
    with pytest.raises(ValueError):
        sim.validate_pars()

    # Can't have both end_days and n_years None
    sim['end'] = None
    sim['n_years'] = None
    with pytest.raises(ValueError):
        sim.validate_pars()
    sim['n_years'] = 10 # Restore

    # Check different initial conditions
    sim['init_hpv_prev'] = [0.08, 0.2] # Can't accept an array without age brackets
    with pytest.raises(ValueError):
        sim.initialize()
    sim['init_hpv_prev'] = {'age_brackets': [15], 'tot': [0.05, 0.1]} # Array of age brackets sould be the same as array of prevalences
    with pytest.raises(ValueError):
        sim.initialize()

    #The following formats are OK
    sim['init_hpv_prev'] = {'age_brackets': [15, 99], 'tot': [0.05, 0.1]}
    sim.initialize()
    sim['init_hpv_prev'] = {'age_brackets': [15, 99], 'm': [0.05, 0.1], 'f': [0.05, 0.1]}
    sim.initialize(reset=True)

    # Check layer pars are internally consistent
    sim['condoms'] = {'invalid':30}
    with pytest.raises(sc.KeyNotFoundError):
        sim.validate_pars()
    sim.reset_layer_pars() # Restore

    # Check mismatch with population
    for key in ['acts', 'condoms']:
        sim[key] = {'invalid':1}
    with pytest.raises(sc.KeyNotFoundError):
        sim.validate_pars()
    sim.reset_layer_pars() # Restore

    return sim


def test_result_consistency():
    ''' Check that results by subgroup sum to the correct totals'''

    # Create sim
    pop_size = 10e3
    sim = hps.Sim(pop_size=pop_size, n_years=10, dt=0.5, label='test_results')
    sim.run()

    # Check that infections by age sum up the the correct totals
    assert (sim.results['cum_total_infections'][:] == sim.results['cum_total_infections_by_age'][:].sum(axis=0)).all() # Check cumulative results by age are equal to cumulative results
    assert (sim.results['new_total_infections'][:] == sim.results['new_total_infections_by_age'][:].sum(axis=0)).all() # Check new results by age are equal to new results

    # Check that infections by genotype sum up the the correct totals
    assert (sim.results['new_infections'][:].sum(axis=0)==sim.results['new_total_infections'][:]).all() # Check flows by genotype are equal to total flows
    assert (sim.results['n_infectious'][:].sum(axis=0)==sim.results['n_total_infectious'][:]).all() # Check flows by genotype are equal to total flows

    # Check that CINs by grade sum up the the correct totals
    assert ((sim.results['new_total_cin1s'][:] + sim.results['new_total_cin2s'][:] + sim.results['new_total_cin3s'][:]) == sim.results['new_total_cins'][:]).all()
    assert ((sim.results['new_cin1s'][:] + sim.results['new_cin2s'][:] + sim.results['new_cin3s'][:]) == sim.results['new_cins'][:]).all()

    # Check that cancers and CINs by age sum up the the correct totals
    assert (sim.results['new_total_cancers'][:] == sim.results['new_total_cancers_by_age'][:].sum(axis=0)).all()
    assert (sim.results['new_total_cins'][:] == sim.results['new_total_cins_by_age'][:].sum(axis=0)).all()
    assert (sim.results['n_total_cin_by_age'][:, :].sum(axis=0) == sim.results['n_total_cin'][:]).all()
    assert (sim.results['n_total_cancerous_by_age'][:, :].sum(axis=0) == sim.results['n_total_cancerous'][:]).all()

    # Check demographics
    assert (sim.results['n_alive_by_age'][:].sum(axis=0) == sim.results['n_alive'][:]).all()
    assert (sim.results['n_alive_by_sex'][0, :] == sim.results['f_alive_by_age'][:].sum(axis=0)).all()
    assert (sim.results['n_alive'][-1]+sim.results['cum_other_deaths'][-1]-sim.results['cum_births'][-1] == sim['pop_size'])
    assert (sim['pop_size'] == pop_size)

    # Check that males don't have CINs or cancers
    import hpvsim.utils as hpu
    male_inds = sim.people.is_male.nonzero()[0]
    males_with_cin = hpu.defined(sim.people.date_cin1[:,male_inds])
    males_with_cancer = hpu.defined(sim.people.date_cancerous[:,male_inds])
    assert len(males_with_cin)==0
    assert len(males_with_cancer)==0

    # Check that people younger than debut don't have HPV
    virgin_inds = (~sim.people.is_active).nonzero()[-1]
    virgins_with_hpv = (~np.isnan(sim.people.date_infectious[:,virgin_inds])).nonzero()[-1]
    assert len(virgins_with_hpv)==0

    return



def test_location_loading():
    ''' Check that data by location can be loaded '''

    sim0 = hps.Sim() # Default values
    sim1 = hps.Sim(location='zimbabwe') # Zimbabwe values
    sim2 = hps.Sim(location='Zimbabwe') # Location should not be case sensitive
    assert (sim0['birth_rates'][1] != sim1['birth_rates'][1]).all() # Values for Zimbabwe should be different to default values
    assert (sim1['birth_rates'][1] == sim2['birth_rates'][1]).all() # Values for Zimbabwe should be loaded regardless of capitalization
    with pytest.warns(RuntimeWarning): # If the location doesn't exist, should use defaults
        sim3 = hps.Sim(location='penelope') # Make sure a warning message is raised
    assert (sim0['birth_rates'][1] == sim3['birth_rates'][1]).all() # Check that defaults have been used

    return


def test_resuming():
    sc.heading('Test that resuming a run works')

    pop_size = 10e3
    s0 = hps.Sim(pop_size=pop_size, n_years=10, dt=0.5, label='test_resume')
    s1 = s0.copy()
    s0.run()

    # Cannot run the same simulation multiple times
    with pytest.raises(hps.AlreadyRunError):
        s0.run()

    # If until=0 then no timesteps will be taken
    with pytest.raises(hps.AlreadyRunError):
        s1.run(until='2015', reset_seed=False)
    assert s1.initialized # It should still have been initialized though
    with pytest.raises(RuntimeError):
        s1.compute_summary(require_run=True) # Not ready yet

    s1.run(until='2020', reset_seed=False)
    with pytest.raises(hps.AlreadyRunError):
        s1.run(until=10, reset_seed=False) # Error if running up to the same value
    with pytest.raises(hps.AlreadyRunError):
        s1.run(until=5, reset_seed=False) # Error if running until a previous timestep

    s1.run(until='2023', reset_seed=False)
    s1.run(reset_seed=False)
    with pytest.raises(hps.AlreadyRunError):
        s1.finalize() # Can't re-finalize a finalized sim

    assert np.all(s0.results['cum_total_infections'].values == s1.results['cum_total_infections']) # Results should be identical

    return s1


# def test_fileio():
#     sc.heading('Test file saving')
#
#     json_path = 'test_hpvsim.json'
#     xlsx_path = 'test_hpvsim.xlsx'
#
#     # Create and run the simulation
#     sim = hps.Sim()
#     sim['n_years'] = 5
#     sim['pop_size'] = 1000
#     sim.run(verbose=0)
#
#     # Create objects
#     json = sim.to_json()
#     xlsx = sim.to_excel()
#     print(xlsx)
#
#     # Save files
#     sim.to_json(json_path)
#     sim.to_excel(xlsx_path)
#
#     for path in [json_path, xlsx_path]:
#         print(f'Removing {path}')
#         os.remove(path)
#
#     return json





#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim0 = test_microsim()
    sim1 = test_sim(do_plot=do_plot, do_save=do_save)
    sim2 = test_epi()
    sim3 = test_flexible_inputs()
    sim4 = test_result_consistency()
    sim5 = test_location_loading()
    sim6 = test_resuming()
    # json = test_fileio()

    sc.toc(T)
    print('Done.')
