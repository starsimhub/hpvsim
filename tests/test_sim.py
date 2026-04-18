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


#%% Define the tests

def test_microsim():
    sc.heading('Minimal sim test')

    # Define baseline parameters and initialize sim
    sim = hpv.Sim()
    pars = {
        'n_agents': 500, # CK: values smaller than this fail
        'init_hpv_prev': .1,
        'n_years': 2,
        'burnin': 0,
        'genotypes': [16,18],
        }
    sim.update_pars(pars)
    sim.run()
    sim.summarize()
    sim.brief()

    return sim


def test_sim(do_plot=False, do_save=False, do_run=True, **kwargs): # If being run via pytest, turn off
    sc.heading('Basic sim test')

    # Settings
    seed = 1
    verbose = 0.1

    # Create and run the simulation
    pars = {
        'n_agents': 5e3,
        'start': 1970,
        'genotypes': [16, 18, 'hi5', 'ohr'],
        'burnin': 30,
        'end': 2030,
        'ms_agent_ratio': 100
    }
    pars = sc.mergedicts(pars, kwargs)

    # Create some genotype pars
    genotype_pars = {
        16: {
            'cin_fn': dict(form='logf2', k=0.25, x_infl=0, ttc=49)
        }
    }

    sim = hpv.Sim(pars=pars, genotype_pars=genotype_pars, rand_seed=seed)
    if do_run:
        sim.run(verbose=verbose)

    # Optionally plot
    if do_plot:
        sim.plot(do_save=do_save)

    return sim


def test_epi():
    sc.heading('Test basic epi dynamics')

    # Define baseline parameters and initialize sim
    base_pars = dict(n_agents=3e3, n_years=20, dt=0.5, genotypes=[16], beta=0.02, verbose=0, eff_condoms=0.6)
    sim = hpv.Sim(pars=base_pars)
    sim.initialize()

    # Define the parameters to vary
    class ParEffects():
        def __init__(self, par, range, variables):
            self.par = par
            self.range = range
            self.variables = variables
            return

    par_effects = [
        ParEffects('beta',          [0.01, 0.99],   ['infections']),
        ParEffects('condoms',       [0.90, 0.10],   ['infections']),
        ParEffects('acts',          [1, 200],       ['infections']),
        ParEffects('debut',         [25, 15],       ['infections']),
        ParEffects('init_hpv_prev', [0.1, 0.8],     ['infections']),
    ]

    # Loop over each of the above parameters and make sure they affect the epi dynamics in the expected ways
    for par_effect in par_effects:
    # for vpar,vval,vrel,vwhat in zip(vary_pars, vary_vals, vary_rels, vary_what):
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

        if par_effect.par == 'model_hiv':
            base_pars['location'] = 'south africa'
            base_pars['n_years'] = 30
            base_pars['dt'] = 0.25
            hiv_datafile = 'test_data/hiv_incidence_south_africa.csv'
            art_datafile = 'test_data/art_coverage_south_africa.csv'
        else:
            hiv_datafile = None
            art_datafile = None

        pars0 = sc.mergedicts(base_pars, {par_effect.par: lo})  # Use lower parameter bound
        pars1 = sc.mergedicts(base_pars, {par_effect.par: hi})  # Use upper parameter bound

        # Run the simulations and pull out the results
        s0 = hpv.Sim(pars0, art_datafile=art_datafile, hiv_datafile=hiv_datafile, label=f'{par_effect.par} {par_effect.range[0]}').run()
        s1 = hpv.Sim(pars1, art_datafile=art_datafile, hiv_datafile=hiv_datafile, label=f'{par_effect.par} {par_effect.range[1]}').run()

        # Check results
        for var in par_effect.variables:
            v0 = s0.results[var][:].sum()
            v1 = s1.results[var][:].sum()
            print(f'Checking {var:10s} with varying {par_effect.par:10s} ... ', end='')
            assert v0 <= v1, f'Expected {var} to be lower with {par_effect.par}={lo} than with {par_effect.par}={hi}, but {v0} > {v1})'
            print(f'✓ ({v0} <= {v1})')

    return s0, s1


def test_states():
    sc.heading('Test states')

    # Define baseline parameters and initialize sim
    base_pars = dict(n_years=20, dt=0.5, network='random', beta=0.05)

    class check_states(hpv.Analyzer):

        def __init__(self):
            self.okay = True
            return

        def apply(self, sim):
            '''
            Checks states that should be mutually exlusive and collectively exhaustive
            '''
            people = sim.people
            ng = sim['n_genotypes']
            removed = people.dead_cancer[:] | people.dead_other[:] | people.emigrated[:]
            for g in range(ng):

                # Infection states: people must be exactly one of susceptible/infectious/inactive
                s1  = (people.susceptible[g,:] | people.infectious[g,:] | people.inactive[g,:] | removed ).all()
                if not s1:
                    raise ValueError('States {susceptible, infectious, inactive} should be collectively exhaustive but are not.')
                s2  = ~(people.susceptible[g,:] & people.infectious[g,:]).any()
                if not s2:
                    raise ValueError('States {susceptible, infectious} should be mutually exclusive but are not.')
                s3  = ~(people.susceptible[g,:] & people.inactive[g,:]).any()
                if not s3:
                    raise ValueError('States {susceptible, inactive} should be mutually exclusive but are not.')
                s4  = ~(people.infectious[g,:] & people.inactive[g,:]).any()
                if not s4:
                    raise ValueError('States {infectious, inactive} should be mutually exclusive but are not.')

                # Dysplasia states:
                #   - people *without active infection* should not be in any dysplasia state (test d0)
                #   - people *with active infection* should be in exactly one dysplasia state (test d1)
                #   - people should either have no cellular changes (normal) or be in a dysplasia state (tests d2-d6)
                d0 = (~((~people.infectious[g,:]) & people.cin[g,:])).any()
                if not d0:
                    raise ValueError('People without active infection should not have detectable cell changes/')
                d1 = (people.normal[g,:] | people.precin[g,:] | people.cin[g,:] | people.cancerous[g,:] | removed).all()
                if not d1:
                    raise ValueError('States {normal, precin, cin, cancerous} should be collectively exhaustive but are not.')
                d2 = ~(people.precin[g,:] & people.cin[g,:]).all()
                if not d2:
                    raise ValueError('States {precin, cin} should be mutually exclusive but are not.')
                d3 = ~(people.cin[g,:] & people.cancerous[g,:]).all()
                if not d3:
                    raise ValueError('States {cin, cancerous} should be mutually exclusive but are not.')

                # Cellular transformation states:
                c1 = (people.normal[g,:] | people.cin[g,:] | people.cancerous[:,:].any(axis=0) | removed).all()
                if not c1:
                    raise ValueError('States {normal, cin, cancerous} should be collectively exhaustive but are not.')
                c2 = ~(people.normal[g,:] & people.cin[g,:]).all()
                if not c2:
                    raise ValueError('States {normal, cin} should be mutually exclusive but are not.')
                c3 = ~(people.cin[g,:] & people.cancerous[:,:].any(axis=0)).all()
                if not c3:
                    raise ValueError('States {cin, cancerous} should be mutually exclusive but are not.')

                # Check combinations of cell states & infection states:
                sc1 = ~(people.normal[g,:] & people.infectious[g,:]).all() # No-one can be infectious without any cell changes
                if not sc1:
                    raise ValueError('Everyone infectious should have abnormal cells, but they do not.')
                sc2 = ~((people.cin[g,:] ) & people.susceptible[g,:]).all() # No-one can be susceptible and have cell changes
                if not sc2:
                    raise ValueError('No-one susceptible should have abnormal cells.')

                # If there's anyone with abnormal cells & inactive infection, they must have cancer
                sd1inds = hpv.true(people.abnormal[g,:] & people.inactive[g,:])
                sd1 = True
                if len(sd1inds)>0:
                    sd1 = people.cancerous[:,sd1inds].any(axis=0).all()
                if not sd1:
                    raise ValueError('Anyne with abnormal cells and inactive infection should have cancer.')

                checkall = np.array([
                    s1, s2, s3, s4,
                    d0, d1, d2, d3,
                    c1, c2, c3,
                    sc1, sc2, sd1
                ])
                if not checkall.all():
                    self.okay = False

            return

    sim = hpv.Sim(pars=base_pars, analyzers=check_states())
    sim.run()
    a = sim.get_analyzer()
    assert a.okay

    return sim


def test_flexible_inputs():
    sc.heading('Testing flexibility of sim inputs')

    # Test resetting layer parameters
    sim = hpv.Sim(n_agents=100, ms_agent_ratio=1, genotypes=[16], label='test_label')
    sim.reset_layer_pars()
    sim.initialize()
    sim.reset_layer_pars()

    # Test validation
    sim['n_agents'] = 'invalid'
    with pytest.raises(ValueError):
        sim.validate_pars()
    sim['n_agents'] = 100 # Restore

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
    n_agents = 1e3
    sim = hpv.Sim(n_agents=n_agents, n_years=10, dt=0.5, label='test_results')
    sim.run()

    # Check that infections by genotype sum up the the correct totals
    # This test works because sim.results['infections'] holds the total number of infections
    # of any genotype that occured each period. Thus, sim.results['infections'] can technically
    # be greater than the total population size, for example if half the population got infected
    # with 2 genotypes simultaneously.
    assert np.allclose(sim.results['infections_by_genotype'][:].sum(axis=0),sim.results['infections'][:]) # Check flows by genotype are equal to total flows
    assert np.allclose(sim.results['infections_by_age'][:].sum(axis=0),sim.results['infections'][:]) # Check flows by genotype are equal to total flows

    # The test below was faulty, but leaving it here (commented out) is instructive.
    # Specifically, the total number of people infectious by genotype (sim.results['n_infectious'])
    # doesn't necessarily sum to the number of infectious people in total (sim.results['n_total_infectious'])
    # because of the possibility of coinfections within a single person.
    # So sim.results['n_total_infectious'] represents the total number of people who have 1+ infections
    # whereas sim.results['n_infectious'] represents the total number of people infected with each genotype.
    # assert (sim.results['n_infectious'][:].sum(axis=0)==sim.results['n_total_infectious'][:]).all() # Check stocks by genotype are equal to stocks flows
    # assert np.allclose(sim.results.age['n_infectious_by_age'][:].sum(axis=0),sim.results['n_infectious'][:]) # Check stocks by age are equal to stocks flows


    # # Check that CINs by grade sum up the the correct totals
    # assert np.allclose((sim.results['cin1s'][:] + sim.results['cin2s'][:] + sim.results['cin3s'][:]),sim.results['dysplasias'][:])
    # assert np.allclose((sim.results['cin1s_by_genotype'][:] + sim.results['cin2s_by_genotype'][:] + sim.results['cin3s_by_genotype'][:]), sim.results['dysplasias_by_genotype'][:])

    # Check that results by age sum to the correct totals
    assert np.allclose(sim.results['cancers_by_age'][:].sum(axis=0),sim.results['cancers'][:])
    assert np.allclose(sim.results['infections_by_age'][:].sum(axis=0),sim.results['infections'][:])

    # Check demographics
    assert (sim['n_agents'] == n_agents)

    # Check that males don't have CINs or cancers
    male_inds = sim.people.is_male.nonzero()[0]
    males_with_cin = hpv.defined(sim.people.date_cin[:,male_inds])
    males_with_cancer = hpv.defined(sim.people.date_cancerous[:,male_inds])
    assert len(males_with_cin)==0
    assert len(males_with_cancer)==0

    # Check that people younger than debut don't have HPV
    virgin_inds = (sim.people.is_virgin).nonzero()[-1]
    virgins_with_hpv = (~np.isnan(sim.people.date_infectious[:,virgin_inds])).nonzero()[-1]
    assert len(virgins_with_hpv)==0

    return sim



def test_location_loading():
    ''' Check that data by location can be loaded '''

    sim0 = hpv.Sim() # Default values
    sim1 = hpv.Sim(location='zimbabwe') # Zimbabwe values
    sim2 = hpv.Sim(location='Zimbabwe') # Location should not be case sensitive
    assert not np.array_equal( sim0['birth_rates'][1],sim1['birth_rates'][1]) # Values for Zimbabwe should be different to default values
    assert np.array_equal(sim1['birth_rates'][1], sim2['birth_rates'][1]) # Values for Zimbabwe should be loaded regardless of capitalization

    return sim1


def test_resuming():
    sc.heading('Test that resuming a run works')

    n_agents = 5e3
    s0 = hpv.Sim(n_agents=n_agents, genotypes=[16], dt=0.5, label='test_resume', start=2015, n_years=15, burnin=5)
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

    assert np.all(s0.results['infections'].values == s1.results['infections']) # Results should be identical

    return s1


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim0 = test_microsim()
    sim1 = test_sim(do_plot=do_plot, do_save=do_save)
    s0, s1 = test_epi()
    sim3 = test_states()
    sim4 = test_flexible_inputs()
    sim5 = test_result_consistency()
    sim6 = test_location_loading()
    sim7 = test_resuming()

    sc.toc(T)
    print('Done.')