'''
Tests for single simulations
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc

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
    sim = hps.Sim()
    sim.set_seed(seed)
    sim.run(verbose=verbose)

    # Optionally plot
    if do_plot:
        sim.plot(do_save=do_save)

    return sim


def test_epi(do_plot=False, do_save=False): # If being run via pytest, turn off
    sc.heading('Test basic epi dynamics')

    # Define parameters specific to this test
    base_pars = dict(n_years=10, dt=0.5)
    pars0 = sc.mergedicts(base_pars, dict(beta=0.05))
    pars1 = sc.mergedicts(base_pars, dict(beta=0.5))

    # Run the simulations and pull out the results
    s0 = hps.Sim(pars0, label='Beta 0.05').run()
    s1 = hps.Sim(pars1, label='Beta 0.5').run()
    res0 = s0.summary
    res1 = s1.summary

    # Check results
    for key in ['cum_total_infections', 'cum_total_cins', 'cum_total_cancers', 'n_total_infectious']:
        v0 = res0[key]
        v1 = res1[key]
        print(f'Checking {key:20s} ... ', end='')
        assert v0 <= v1, f'Expected {key} to be lower with low beta ({v0}) than high ({v1})'
        print(f'✓ ({v0} <= {v1})')

    return


def test_sim_inputs():
    sc.heading('Testing sim inputs')

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
    sim['init_hpv_prev'] = {'age_brackets': [15], 'tot': [0.05, 0.1]}
    with pytest.raises(ValueError):
        sim.initialize()
    sim['init_hpv_prev'] = {'age_brackets': [15, 99], 'tot': [0.05, 0.1]}
    sim.initialize()
    sim.run()

    # # Check layer pars are internally consistent
    # sim['condoms'] = {'invalid':30}
    # with pytest.raises(sc.KeyNotFoundError):
    #     sim.validate_pars()
    # sim.reset_layer_pars() # Restore

    # # Check mismatch with population
    # for key in ['acts', 'contacts', 'quar_factor']:
    #     sim[key] = {'invalid':1}
    # with pytest.raises(sc.KeyNotFoundError):
    #     sim.validate_pars()
    # sim.reset_layer_pars() # Restore

    # # Convert interventions dict to intervention
    # sim['interventions'] = {'which': 'change_beta', 'pars': {'days': 10, 'changes': 0.5}}
    # sim.validate_pars()

    return





def test_init_conditions():
    sc.heading('Test initial conditions')

    # Define parameters specific to this test

    # Run the simulations and pull out the results
    s0 = hps.Sim(pars0, label='Beta 0.05').run()
    s1 = hps.Sim(pars1, label='Beta 0.5').run()
    res0 = s0.summary
    res1 = s1.summary

    # Check results
    for key in ['cum_total_infections', 'cum_total_cins', 'cum_total_cancers', 'n_total_infectious']:
        v0 = res0[key]
        v1 = res1[key]
        print(f'Checking {key:20s} ... ', end='')
        assert v0 <= v1, f'Expected {key} to be lower with low beta ({v0}) than high ({v1})'
        print(f'✓ ({v0} <= {v1})')

    return



# def test_fileio():
#     sc.heading('Test file saving')
#
#     json_path = 'test_covasim.json'
#     xlsx_path = 'test_covasim.xlsx'
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


# def test_sim_data(do_plot=False):
#     sc.heading('Data test')
#
#     pars = dict(
#         pop_size = 2000,
#         start_day = '2020-02-25',
#         )
#
#     # Create and run the simulation
#     sim = cv.Sim(pars=pars, datafile=os.path.join(sc.thisdir(__file__), 'example_data.csv'))
#     sim.run()
#
#     # Optionally plot
#     if do_plot:
#         sim.plot()
#
#     return sim


# def test_dynamic_resampling(do_plot=False): # If being run via pytest, turn off
#     sc.heading('Test dynamic resampling')
#
#     pop_size = 1000
#     sim = cv.Sim(pop_size=pop_size, rescale=1, pop_scale=1000, n_days=180, rescale_factor=2)
#     sim.run()
#
#     # Optionally plot
#     if do_plot:
#         sim.plot()
#
#     # Create and run a basic simulation
#     assert sim.results['cum_infections'][-1] > pop_size  # infections at the end of sim should be much more than internal pop
#     return sim



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # sim0 = test_microsim()
    # sim1 = test_sim(do_plot=do_plot, do_save=do_save)
    # sim2 = test_epi()
    sim3 = test_sim_inputs()
    # json = test_fileio()
    # sim2 = test_sim_data(do_plot=do_plot)
    # sim3 = test_dynamic_resampling(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
