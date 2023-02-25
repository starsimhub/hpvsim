'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv

do_plot = 0
do_save = 0
debug = 0

n_agents = [50e3,500][debug] # Swap between sizes
start = [1970,1990][debug]


#%% Define the tests

def test_hiv(model_hiv=True):
    sc.heading('Testing hiv')

    pars = {
        'n_agents': n_agents,
        'location': 'south africa',
        'model_hiv': model_hiv,
        'start': start,
        'end': 2030
    }

    if model_hiv:
        hiv_datafile='hiv_incidence_south_africa.csv'
        art_datafile = 'art_coverage_south_africa.csv'
    else:
        hiv_datafile=None
        art_datafile=None

    sim = hpv.Sim(
        pars=pars,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile
    )
    sim.run()
    sim.plot(to_plot=['hiv_prevalence_by_age'])
    return sim


def test_impact_on_cancer():
    sc.heading('Testing hiv')

    pars = {
        'n_agents': n_agents,
        'location': 'south africa',
        'start': start,
        'end': 2030
    }

    base_sim = hpv.Sim(
        pars=pars,
        hiv_datafile='hiv_incidence_south_africa.csv',
        art_datafile='art_coverage_south_africa.csv'
    )

    scenarios = {
        'no_hiv': {
            'name': 'No HIV',
            'pars': {
                'model_hiv': False
            }
        },
        'hiv_baseline': {
            'name': 'HIV',
            'pars': {
                'model_hiv': True
            }
        }
    }

    metapars = {'n_runs': 2}
    scens = hpv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(debug=debug)
    to_plot = {
        'HIV prevalence': [
            'hiv_prevalence',
        ],
        'HPV prevalence': [
            'hpv_prevalence',
        ],
        'Age standardized cancer incidence (per 100,000 women)': [
            'asr_cancer_incidence',
        ]
    }
    scens.plot(to_plot=to_plot)
    return scens

#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()
    sim0 = test_hiv(model_hiv=True)
    # sim1 = test_impact_on_cancer()
    sc.toc(T)
    print('Done.')
