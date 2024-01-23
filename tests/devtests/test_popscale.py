'''
Script for simulating a population with demographic inputs
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv

do_plot = 1
do_run = 1

#%% Define the tests

def test_scaled_sim(do_plot=False, do_run=True):
    sc.heading('Sim with custom demographics')

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
        'ms_agent_ratio': 100,
        'location': 'india',
        'age_datafile': 'devtest_data/mah_age_data.csv',
        'pop_datafile': 'devtest_data/mah_pop_data.csv',
        'popage_datafile': 'devtest_data/mah_popage_data.csv',
    }

    sim = hpv.Sim(pars=pars, rand_seed=seed)

    if do_run:
        sim.run(verbose=verbose)

    # Optionally plot
    if do_plot:
        sim.plot()
        sim.plot("n_alive")

    return sim


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim = test_scaled_sim(do_plot=do_plot, do_run=do_run)

    sc.toc(T)
    print('Done.')