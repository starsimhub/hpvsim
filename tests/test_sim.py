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
import hpvsim.utils as hpu

do_plot = 1
do_save = 0


#%% Define the tests

def test_microsim():
    sc.heading('Minimal sim test')

    sim = hpv.Sim()
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
    hpv16 = hpv.genotype('HPV16')
    hpv18 = hpv.genotype('HPV18')

    pars = {
        'pop_size': 50e3,
        'start': 1990,
        'burnin': 30,
        'end': 2030,
        'location': 'tanzania',
        'dt': .5,
    }

    # age_target = {'inds': lambda sim: hpu.true((sim.people.age < 9)+(sim.people.age > 14)), 'vals': 0}  # Only give boosters to people who have had 2 doses
    # doses_per_year = 2e3
    # bivalent_2_dose = hpv.vaccinate_num(vaccine='bivalent_2dose', num_doses=doses_per_year,
    #                                     timepoints=['2020', '2021', '2022', '2023', '2024'],
    #                                     label='bivalent 2 dose, 9-14', subtarget=age_target)

    sim = hpv.Sim(pars=pars, genotypes=[hpv16,hpv18])
    sim.set_seed(seed)

    # Optionally plot
    if do_plot:
        sim.run(verbose=verbose)
        sim.plot(do_save=do_save)

    return sim


def test_epi():
    sc.heading('Test basic epi dynamics')

    # Define baseline parameters and initialize sim
    base_pars = dict(n_years=10, dt=0.5)
    sim = hpv.Sim()

    # Define the parameters to vary
    vary_pars   = ['beta',          'acts',             'condoms',          'debut',            'init_hpv_prev'] # Parameters
    vary_vals   = [[0.01, 0.99],    [10,200],           [0.1,1.0],         [15,25],             [0.01,0.8]] # Values
    vary_rels   = ['pos',           'pos',              'neg',              'neg',              'pos'] # Expected association with epi outcomes
    vary_what   = ['hpv_incidence', 'hpv_incidence',    'hpv_incidence',    'hpv_incidence',    'cancer_prevalence'] # Epi outcomes to check

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
        s0 = hpv.Sim(pars0, label=f'{vpar} {vval[0]}').run()
        s1 = hpv.Sim(pars1, label=f'{vpar} {vval[1]}').run()
        res0 = s0.summary
        res1 = s1.summary

        # Check results
        key='total_'+vwhat
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
    sim = hpv.Sim(pop_size=100, label='test_label')
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
    sim = hpv.Sim(pop_size=pop_size, n_years=10, dt=0.5, label='test_results')
    sim.run()

    # Check that infections by genotype sum up the the correct totals
    assert (sim.results['infections'][:].sum(axis=0)==sim.results['total_infections'][:]).all() # Check flows by genotype are equal to total flows
    assert (sim.results['n_infectious'][:].sum(axis=0)==sim.results['n_total_infectious'][:]).all() # Check flows by genotype are equal to total flows

    # Check that CINs by grade sum up the the correct totals
    assert ((sim.results['total_cin1s'][:] + sim.results['total_cin2s'][:] + sim.results['total_cin3s'][:]) == sim.results['total_cins'][:]).all()
    assert ((sim.results['cin1s'][:] + sim.results['cin2s'][:] + sim.results['cin3s'][:]) == sim.results['cins'][:]).all()

    # Check demographics
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

    sim0 = hpv.Sim() # Default values
    sim1 = hpv.Sim(location='zimbabwe') # Zimbabwe values
    sim2 = hpv.Sim(location='Zimbabwe') # Location should not be case sensitive
    assert not np.array_equal( sim0['birth_rates'][1],sim1['birth_rates'][1]) # Values for Zimbabwe should be different to default values
    assert np.array_equal(sim1['birth_rates'][1], sim2['birth_rates'][1]) # Values for Zimbabwe should be loaded regardless of capitalization
    with pytest.warns(RuntimeWarning): # If the location doesn't exist, should use defaults
        sim3 = hpv.Sim(location='penelope') # Make sure a warning message is raised
    assert np.array_equal(sim0['birth_rates'][1], sim3['birth_rates'][1]) # Check that defaults have been used

    return sim1


def test_resuming():
    sc.heading('Test that resuming a run works')

    pop_size = 10e3
    s0 = hpv.Sim(pop_size=pop_size, n_years=10, dt=0.5, label='test_resume')
    s1 = s0.copy()
    s0.run()

    # Cannot run the same simulation multiple times
    with pytest.raises(hpv.AlreadyRunError):
        s0.run()

    # If until=0 then no timesteps will be taken
    with pytest.raises(hpv.AlreadyRunError):
        s1.run(until='2015', reset_seed=False)
    assert s1.initialized # It should still have been initialized though
    with pytest.raises(RuntimeError):
        s1.compute_summary(require_run=True) # Not ready yet

    s1.run(until='2020', reset_seed=False)
    with pytest.raises(hpv.AlreadyRunError):
        s1.run(until=10, reset_seed=False) # Error if running up to the same value
    with pytest.raises(hpv.AlreadyRunError):
        s1.run(until=5, reset_seed=False) # Error if running until a previous timestep

    s1.run(until='2023', reset_seed=False)
    s1.run(reset_seed=False)
    with pytest.raises(hpv.AlreadyRunError):
        s1.finalize() # Can't re-finalize a finalized sim

    assert np.all(s0.results['total_infections'].values == s1.results['total_infections']) # Results should be identical

    return s1




#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # sim0 = test_microsim()
    sim1 = test_sim(do_plot=do_plot, do_save=do_save)
    # sim2 = test_epi()
    # sim3 = test_flexible_inputs()
    # sim4 = test_result_consistency() # CURRENTLY BROKEN: CINs by grade to not sum to total CINs
    # sim5 = test_location_loading()
    # sim6 = test_resuming()

    sc.toc(T)
    print('Done.')