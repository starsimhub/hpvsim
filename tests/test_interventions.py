'''
Tests for single simulations
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc
import numpy as np
import seaborn as sns
import hpvsim as hpv
import hpvsim.parameters as hpvpar
import hpvsim.utils as hpu

do_plot = 1
do_save = 0
hpv16 = hpv.genotype('HPV16')
hpv18 = hpv.genotype('HPV18')

base_pars = {
    'pop_size': 50e3,
    'start': 1985,
    'burnin': 30,
    'end': 2050,
    'genotypes': [hpv16, hpv18],
    'location': 'tanzania',
    'dt': .2,
}


#%% Define the tests

def test_dynamic_pars():
    sc.heading('Test dynamics pars intervention')

    pars = {
        'pop_size': 10e3,
        'n_years': 10,
    }

    # Model an intervention to increase condom use
    condom_int = hpv.dynamic_pars(
        condoms=dict(timepoints=10, vals={'c': 0.9}))  # Increase condom use among casual partners to 90%

    # Model an intervention to increase the age of sexual debut
    debut_int = hpv.dynamic_pars(
        {'debut': {
            'timepoints': '2020',
            'vals': dict(f=dict(dist='normal', par1=20, par2=2.1), # Increase mean age of sexual debut
                         m=dict(dist='normal', par1=19.6,par2=1.8))
        }
        }
    )

    sim = hpv.Sim(pars=pars, interventions=[condom_int, debut_int])
    sim.run()
    return sim


def test_vaccinate_prob(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test prophylactic vaccine intervention')

    verbose = .1
    debug = 0

    # Model an intervention to roll out prophylactic vaccination
    vx_prop = 0.5
    def age_subtarget(sim):
        ''' Select people who are eligible for vaccination '''
        inds = sc.findinds((sim.people.age >= 9) & (sim.people.age <=14))
        return {'vals': [vx_prop for _ in inds], 'inds': inds}

    def faster_age_subtarget(sim):
        ''' Select people who are eligible for vaccination '''
        inds = sc.findinds((sim.people.age >= 9) & (sim.people.age <=24))
        return {'vals': [vx_prop for _ in inds], 'inds': inds}

    years = np.arange(2020, base_pars['end'], dtype=int)
    bivalent_vx = hpv.vaccinate_prob(vaccine='bivalent', label='bivalent, 9-14', timepoints=years,
                                       subtarget=age_subtarget)
    bivalent_vx_faster = hpv.vaccinate_prob(vaccine='bivalent', label='bivalent, 9-24', timepoints=years,
                                       subtarget=faster_age_subtarget)

    n_runs = 3
    sim = hpv.Sim(pars=base_pars)

    # Define the scenarios
    scenarios = {
        'no_vx': {
            'name': 'No vaccination',
            'pars': {
            }
        },
        'vx': {
            'name': f'Vaccinate {vx_prop*100}% of 9-14y girls starting in 2020',
            'pars': {
                'interventions': [bivalent_vx]
            }
        },
        'faster_vx': {
            'name': f'Vaccinate {vx_prop * 100}% of 9-24y girls starting in 2020',
            'pars': {
                'interventions': [bivalent_vx_faster]
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = hpv.Scenarios(sim=sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)
    scens.compare()

    if do_plot:
        to_plot = {
            'HPV incidence': [
                'total_hpv_incidence',
            ],
            'CIN prevalence': [
                'total_cin_prevalence',
            ],
            'Number vaccinated': [
                'cum_total_vaccinated',
            ],
        }
        scens.plot(do_save=do_save, to_plot=to_plot, fig_path=fig_path)

    return scens


def test_vaccinate_num(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test vaccinate_num intervention')

    verbose = .1
    debug = 0

    # Model an intervention to roll out prophylactic vaccination with a given number of doses over time
    age_target = {'inds': lambda sim: hpu.true((sim.people.age < 9)+(sim.people.age > 14)), 'vals': 0}  # Only give vaccine to people who have had 2 doses
    doses_per_year = 6e3
    bivalent_1_dose = hpv.vaccinate_num(vaccine='bivalent_1dose', num_doses=doses_per_year, timepoints=['2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029'], label='bivalent 1 dose, 9-14', subtarget=age_target)
    bivalent_2_dose = hpv.vaccinate_num(vaccine='bivalent_2dose', num_doses=doses_per_year, timepoints=['2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029'], label='bivalent 2 dose, 9-14', subtarget=age_target)
    # bivalent_3_dose = hpv.vaccinate_num(vaccine='bivalent_3dose', num_doses=doses_per_year, timepoints=['2020', '2021', '2022', '2023', '2024'], label='bivalent 3 dose, 9-14', subtarget=age_target)

    sim = hpv.Sim(pars=base_pars)
    n_runs = 3

    # Define the scenarios
    scenarios = {
        'no_vx': {
            'name': 'No vaccination',
            'pars': {
            }
        },
        'vx2': {
            'name': f'Double dose, 9-14y girls, {int(doses_per_year)} doses available per year',
            'pars': {
                'interventions': [bivalent_2_dose]
            }
        },
        'vx1': {
            'name': f'Single dose, 9-14y girls, {int(doses_per_year)} doses available per year',
            'pars': {
                'interventions': [bivalent_1_dose]
            }
        }
    }

    metapars = {'n_runs': n_runs}

    scens = hpv.Scenarios(sim=sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)
    scens.compare()

    if do_plot:
        to_plot = {
            'HPV incidence': [
                'total_hpv_incidence',
            ],
            'CIN prevalence': [
                'total_cin_prevalence',
            ],
            'Number vaccinated': [
                'cum_total_vaccinated',
            ],
        }
        scens.plot(do_save=do_save, to_plot=to_plot, fig_path=fig_path)
        scens.plot_age_results()

    return scens


def test_screening(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test screening intervention')

    hpv16 = hpv.genotype('HPV16')
    hpv18 = hpv.genotype('HPV18')
    verbose = .1
    debug = 0
    n_agents = 50e3

    pars = {
        'pop_size': n_agents,
        'n_years': 60,
        'burnin': 30,
        'start': 1970,
        'genotypes': [hpv16, hpv18],
        'pop_scale' : 25.2e6 / n_agents,
        'location': 'tanzania',
        'dt': .2,
    }

    # Model an intervention to screen 50% of 30 year olds with hpv DNA testing and treat immediately
    screen_prop = .7
    compliance = .9
    cancer_compliance = 0.2
    hpv_screening = hpv.Screening(primary_screen_test='hpv', treatment='via_triage', screen_start_age=30,
                                  screen_stop_age=50, screen_interval=5, timepoints='2010',
                                  screen_compliance=screen_prop, triage_compliance=compliance, cancer_compliance=cancer_compliance)


    hpv_via_screening = hpv.Screening(primary_screen_test='hpv', triage_screen_test='via', treatment='via_triage', screen_start_age=30,
                                  screen_stop_age=50, screen_interval=10, timepoints='2010', label='hpv primary, via triage',
                                  screen_compliance=screen_prop, triage_compliance=compliance, cancer_compliance=cancer_compliance)

    hpv_hpv1618_screening = hpv.Screening(primary_screen_test='hpv', triage_screen_test='hpv1618', treatment='via_triage',
                                        screen_start_age=30,screen_stop_age=50, screen_interval=10, timepoints='2010',
                                        label='hpv primary, hpv1618 triage', screen_compliance=screen_prop,
                                          triage_compliance=compliance, cancer_compliance=cancer_compliance)

    az = hpv.age_results(
        timepoints=['2020'],
        result_keys=['total_hpv_prevalence']
    )


    sim = hpv.Sim(pars=pars, analyzers=[az])
    n_runs = 3

    # Define the scenarios
    scenarios = {
        'no_screening_rsa': {
            'name': 'No screening',
            'pars': {
            }
        },
        # 'hpv_screening': {
        #     'name': f'Screen {screen_prop * 100}% of 30-50y women with {hpv_screening.label}',
        #     'pars': {
        #         'interventions': [hpv_screening],
        #     }
        # },
        # 'hpv_via_screening': {
        #     'name': f'Screen {screen_prop * 100}% of 30-50y women with {hpv_via_screening.label}',
        #     'pars': {
        #         'interventions': [hpv_via_screening],
        #     }
        # },
        # 'hpv_hpv1618_screening': {
        #     'name': f'Screen {screen_prop * 100}% of 30-50y women with {hpv_hpv1618_screening.label}',
        #     'pars': {
        #         'interventions': [hpv_hpv1618_screening],
        #     }
        # },
    }

    metapars = {'n_runs': n_runs}

    scens = hpv.Scenarios(sim=sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)
    scens.compare()

    if do_plot:
        to_plot = {
            'HPV prevalence': [
                'total_hpv_prevalence',
            ],
            'CIN prevalence': [
                'total_cin_prevalence',
            ],
            'Cancers per 100,000 women': [
                'total_cancer_incidence',
            ],
        }
        scens.plot(to_plot=to_plot)
        scens.plot_age_results(plot_type=sns.boxplot)

    return scens


def test_screening_ltfu(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test screening LTFU params')

    hpv16 = hpv.genotype('HPV16')
    hpv18 = hpv.genotype('HPV18')
    verbose = .1
    debug = 0
    n_agents = 50e3

    pars = {
        'pop_size': n_agents,
        'n_years': 60,
        'burnin': 30,
        'start': 1970,
        'genotypes': [hpv16, hpv18],
        'pop_scale' : 25.2e6 / n_agents,
        'location': 'tanzania',
        'dt': .2,
    }

    # Model an intervention to screen 50% of 30 year olds with hpv DNA testing and treat immediately
    hpv_screening = hpv.Screening(primary_screen_test='hpv', treatment='via_triage', screen_start_age=30,
                                  screen_stop_age=50, screen_interval=5, timepoints='2010',
                                  screen_compliance=0.7, triage_compliance=0.9, cancer_compliance=0.2,
                                  excision_compliance=0.2, ablation_compliance=0.7)


    hpv_via_screening = hpv.Screening(primary_screen_test='hpv', triage_screen_test='via', treatment='via_triage', screen_start_age=30,
                                  screen_stop_age=50, screen_interval=10, timepoints='2010', label='hpv primary, via triage',
                                      screen_compliance=0.7, triage_compliance=0.9, cancer_compliance=0.2,
                                      excision_compliance=0.2, ablation_compliance=0.7)

    hpv_via_screening_more_ltfu = hpv.Screening(primary_screen_test='hpv', triage_screen_test='via', treatment='via_triage', screen_start_age=30,
                                  screen_stop_age=50, screen_interval=10, timepoints='2010', label='hpv primary, via triage, more LTFU',
                                      screen_compliance=0.7, triage_compliance=0.6, cancer_compliance=0.2,
                                      excision_compliance=0.1, ablation_compliance=0.5)

    az = hpv.age_results(
        timepoints=['2030'],
        result_keys=['total_cin_prevalence']
    )


    sim = hpv.Sim(pars=pars, analyzers=[az])
    n_runs = 3

    # Define the scenarios
    scenarios = {
        'hpv_screening': {
            'name': f'Screen 70% of 30-50y women with {hpv_screening.label}',
            'pars': {
                'interventions': [hpv_screening],
            }
        },
        'hpv_via_screening': {
            'name': f'Screen 70% of 30-50y women with {hpv_via_screening.label}',
            'pars': {
                'interventions': [hpv_via_screening],
            }
        },
        'hpv_via_screening_more_ltfu': {
            'name': f'Screen 70% of 30-50y women with {hpv_via_screening_more_ltfu.label}',
            'pars': {
                'interventions': [hpv_via_screening_more_ltfu],
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = hpv.Scenarios(sim=sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)
    scens.compare()

    if do_plot:
        to_plot = {
            'HPV prevalence': [
                'total_hpv_prevalence',
            ],
            'CIN prevalence': [
                'total_cin_prevalence',
            ],
            'Cancers per 100,000 women': [
                'total_cancer_incidence',
            ],
        }
        scens.plot(to_plot=to_plot)
        # scens.plot_age_results(plot_type=sns.boxplot)

    return scens

#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim0 = test_dynamic_pars()
    scens1 = test_vaccinate_prob(do_plot=True)
    scens2 = test_vaccinate_num(do_plot=True)
    scens3 = test_screening(do_plot=True)
    scens4 = test_screening_ltfu(do_plot=True)

    sc.toc(T)
    print('Done.')
