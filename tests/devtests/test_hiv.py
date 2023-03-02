'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv
import numpy as np

do_plot = 0
do_save = 0
debug = 1

n_agents = [50e3,500][debug] # Swap between sizes
start = [1950,1990][debug]
ms_agent_ratio = [100,10][debug]
hiv_datafile = '../test_data/hiv_incidence_south_africa.csv'
art_datafile = '../test_data/art_coverage_south_africa.csv'


#%% Define the tests

def test_calibration_hiv():
    sc.heading('Testing calibration with hiv pars')
    pars = {
        'n_agents': n_agents,
        'location': 'south africa',
        'model_hiv': True,
        'start': start,
        'end': 2020,
    }

    sim = hpv.Sim(
        pars=pars,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile
    )

    calib_pars = dict(
        beta=[0.05, 0.010, 0.20],
        dur_transformed=dict(par1=[5, 3, 10]),
    )
    genotype_pars = dict(
        hpv16=dict(
            sev_fn=dict(k=[0.5, 0.2, 1.0],)
        ),
        hpv18=dict(
            sev_fn=dict(k=[0.5, 0.2, 1.0],)
        )
    )

    hiv_pars = dict(
        rel_sus= dict(
            cat1=dict(value=[3, 2,4])
        )
    )


    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars, hiv_pars=hiv_pars,
                            datafiles=[
                                '../test_data/south_africa_hpv_data.csv',
                                '../test_data/south_africa_cancer_data_2020.csv',
                                '../test_data/south_africa_cancer_data_hiv_2020.csv',
                            ],
                            total_trials=3, n_workers=1)
    calib.calibrate(die=True)
    calib.plot(res_to_plot=4)
    return sim, calib


def test_hiv():
    sc.heading('Testing hiv')

    partners = dict(m=dict(dist='poisson', par1=0.1),
                    c=dict(dist='poisson', par1=0.5),
                    o=dict(dist='poisson', par1=0.0),
                    )

    pars = {
        'n_agents': n_agents,
        'location': 'south africa',
        'model_hiv': True,
        'start': start,
        'end': 2020,
        'ms_agent_ratio': ms_agent_ratio,
        'partners': partners,
        'cross_layer': 0.1  # Proportion of females who have crosslayer relationships
        # 'hiv_pars': {
        # 'rel_sus': dict(
        #     cat1=dict(value=3)
        # )
        # }
    }


    sim = hpv.Sim(
        pars=pars,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile
    )
    sim.run()
    to_plot = {
        'ART Coverage': [
            'art_coverage',
        ],
        'HPV prevalence by HIV status': [
            'hpv_prevalence_by_age_with_hiv',
            'hpv_prevalence_by_age_no_hiv'
        ],
        'Age standardized cancer incidence (per 100,000 women)': [
            'asr_cancer_incidence',
            'cancer_incidence_with_hiv',
            'cancer_incidence_no_hiv',
        ],
        'Cancers by age and HIV status': [
            'cancers_by_age_with_hiv',
            'cancers_by_age_no_hiv'
        ]
    }
    sim.plot()
    sim.plot(to_plot=to_plot)
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
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile
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
        'HPV prevalence': [
            'hpv_prevalence',
        ],
        'Age standardized cancer incidence (per 100,000 women)': [
            'asr_cancer_incidence',
        ],

    }
    scens.plot(to_plot=to_plot)
    return scens

#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()
    sim0 = test_hiv()
    sim1 = test_impact_on_cancer()
    sim, calib = test_calibration_hiv()
    sc.toc(T)
    print('Done.')
