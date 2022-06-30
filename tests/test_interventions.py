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
import hpvsim.parameters as hpvpar
import hpvsim.utils as hpu

do_plot = 1
do_save = 0
hpv16 = hpv.genotype('HPV16')
hpv18 = hpv.genotype('HPV18')

base_pars = {
    'pop_size': 50e3,
    'start': 1990,
    'burnin': 20,
    'end': 2025,
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


def test_vaccines(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test prophylactic vaccine intervention')

    verbose = .1
    debug = 0

    # Model an intervention to roll out prophylactic vaccination
    vx_prop = 1.0
    def age_subtarget(sim):
        ''' Select people who are eligible for vaccination '''
        inds = sc.findinds((sim.people.age >= 9) & (sim.people.age <=14))
        return {'vals': [vx_prop for _ in inds], 'inds': inds}

    def faster_age_subtarget(sim):
        ''' Select people who are eligible for vaccination '''
        inds = sc.findinds((sim.people.age >= 9) & (sim.people.age <=24))
        return {'vals': [vx_prop for _ in inds], 'inds': inds}

    bivalent_vx = hpv.vaccinate_prob(vaccine='bivalent', label='bivalent, 9-14', timepoints='2020',
                                       subtarget=age_subtarget)

    bivalent_vx_faster = hpv.vaccinate_prob(vaccine='bivalent', label='bivalent, 9-24', timepoints='2020',
                                       subtarget=faster_age_subtarget)

    sim = hpv.Sim(pars=base_pars)

    n_runs = 3

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
            'HPV infections': [
                'total_infections',
            ],
            'Cancers per 100,000 women': [
                'total_cancer_incidence',
            ],
        }
        scens.plot(do_save=do_save, to_plot=to_plot, fig_path=fig_path)

    return scens


def test_vaccinate_num(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test vaccinate_num intervention')

    hpv16 = hpv.genotype('HPV16')
    hpv18 = hpv.genotype('HPV18')
    verbose = .1
    debug = 0

    # Model an intervention to roll out prophylactic vaccination with a given number of doses over time
    age_target = {'inds': lambda sim: hpu.true((sim.people.age < 9)+(sim.people.age > 14)), 'vals': 0}  # Only give boosters to people who have had 2 doses
    doses_per_year = 2e3
    bivalent_1_dose = hpv.vaccinate_num(vaccine='bivalent_1dose', num_doses=doses_per_year, timepoints=['2020', '2021', '2022', '2023', '2024'], label='bivalent 1 dose, 9-14', subtarget=age_target)
    bivalent_2_dose = hpv.vaccinate_num(vaccine='bivalent_2dose', num_doses=doses_per_year, timepoints=['2020', '2021', '2022', '2023', '2024'], label='bivalent 2 dose, 9-14', subtarget=age_target)
    bivalent_3_dose = hpv.vaccinate_num(vaccine='bivalent_3dose', num_doses=doses_per_year, timepoints=['2020', '2021', '2022', '2023', '2024'], label='bivalent 3 dose, 9-14', subtarget=age_target)

    # sim = hpv.Sim(pars=base_pars, interventions=[bivalent_1_dose])
    # sim.run()
    # sim.plot()
    # return sim

    sim = hpv.Sim(pars=base_pars)
    n_runs = 3

    # Define the scenarios
    scenarios = {
        'no_vx': {
            'name': 'No vaccination',
            'pars': {
            }
        },
        'vx1': {
            'name': f'Single dose, 9-14y girls, {int(doses_per_year)} doses available per year',
            'pars': {
                'interventions': [bivalent_1_dose]
            }
        },
        'vx2': {
            'name': f'Double dose, 9-14y girls, {int(doses_per_year)} doses available per year',
            'pars': {
                'interventions': [bivalent_2_dose]
            }
        },
        'vx3': {
            'name': f'Triple dose, 9-14y girls, {int(doses_per_year)} doses available per year',
            'pars': {
                'interventions': [bivalent_3_dose]
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
            'HPV infections': [
                'total_infections',
            ],
            'Cancers per 100,000 women': [
                'total_cancer_incidence',
            ],
        }
        scens.plot(do_save=do_save, to_plot=to_plot, fig_path=fig_path)


    return scens





#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # sim0 = test_dynamic_pars()
    # scens = test_vaccines(do_plot=True)
    scens = test_vaccinate_num(do_plot=True)

    sc.toc(T)
    print('Done.')
