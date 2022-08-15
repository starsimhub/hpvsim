'''
Create a demo project
'''

#%% Imports and settings
import os
import pytest
import sys
import sciris as sc
import numpy as np
import hpvsim as hpv


do_plot = 1
do_save = 0
figfolder = 'figs'
location = 'kenya'

#%% Define the tests
def test_demo(datafile=None, do_plot=True, do_save=False):

    sc.heading('Creating a demo project')

    pars = dict(n_agents=50e3,
                start=1950,
                end=2020,
                dt=.5,
                network='default',
                location=location,
                genotypes=[16,18,35,58,45,18,52],
                verbose=0.1
                )

    # Initial conditions
    pars['init_hpv_dist'] = {'hpv16': .347, 'hpv35': .174, 'hpv58': .121,
                             'hpv45': .116, 'hpv18': .114, 'hpv52': .097}
    pars['init_hpv_prev'] = {
        'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
        'm'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
        'f'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
    }

    pars['sero'] = 2.5
    pars['hpv_control_prob'] = 0.4
    pars['beta'] = 0.217

    # Initialize genotypes
    hpv33   = hpv.genotype('HPV33')
    hpv45   = hpv.genotype('HPV45')
    hpv18    = hpv.genotype('HPV18')
    hpv16   = hpv.genotype('HPV16')
    hpv31   = hpv.genotype('HPV31')
    hpv52   = hpv.genotype('HPV52')

    # # Set up genotype pars
    # HPV 16 #
    # hpv16.p.prog_rate = .567
    # hpv16.p.prog_time = 6.23
    # hpv16.p.dur_dysp['par1'] = 7.59
    # hpv16.p.dur_none['par1'] = 2.35
    #
    # # HPV 18 #
    # hpv18.p.dysp_rate = 1.2
    # hpv18.p.prog_rate = 0.956
    # hpv18.p.prog_time = 5.189
    # hpv18.p.dur_dysp['par1'] = 5.627
    # hpv18.p.dur_none['par1'] = 2.76
    # hpv18.p.rel_beta = 0.7228
    #
    # # HPV 31 #
    # hpv31.p.prog_rate = 0.17948958373918267
    # hpv31.p.prog_time = 14.12136294913353
    # hpv31.p.dur_dysp['par1'] = 1.353471867503286
    # hpv31.p.rel_beta = 0.9355728046226204
    #
    # # HPV 33 #
    # hpv33.p.prog_rate = 0.23565745824841802
    # hpv33.p.prog_time = 8.4648
    # hpv33.p.dur_dysp['par1'] = 14.12136294913353
    # hpv33.p.rel_beta = 0.26010781379830755
    #
    # # HPV 45 #
    # hpv45.p.dysp_rate = 1.2
    # hpv45.p.prog_rate = 0.9499
    # hpv45.p.prog_time = 7.3
    # hpv45.p.dur_dysp['par1'] = 3.575
    # hpv45.p.dur_none['par1'] = 1.639
    # hpv45.p.rel_beta = 0.77
    #
    # # HPV 52 #
    # hpv52.p.prog_rate = 0.484
    # hpv52.p.prog_time = 4.24
    # hpv52.p.dur_dysp['par1'] = 2.35
    # hpv52.p.rel_beta = 0.623


    az = hpv.age_results(
        result_keys=sc.objdict(
            hpv_prevalence=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            ),
            cin_prevalence=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            ),
            cancer_mortality=sc.objdict(
                datafile=f'test_data/{location}_cancer_mortality.csv',
            ),
            cancer_incidence=sc.objdict(
                datafile=f'test_data/{location}_cancer_incidence.csv',
            )
        )
    )

    analyzers = [az]
    genotypes = [hpv16, hpv18, hpv31, hpv33, hpv45, hpv52]
    interventions = []

    # Create sim
    sim = hpv.Sim(pars=pars,
                  genotypes=genotypes,
                  analyzers=analyzers,
                  interventions=interventions,
                  datafile=datafile)

    # Run sim
    sim.run()
    az = sim.get_analyzer(0)

    # Check plot
    if do_plot:
        to_plot = {
            'HPV prevalence': [
                'hpv_prevalence',
            ],
            'HPV type distribution in cancer': [
                'cancer_types',
            ],
            'Cervical cancer mortality': [
                'cancer_mortality',
            ],
            'Cervical cancer incidence': [
                'cancer_incidence',
            ],
        }
        # sim.plot('demographics', do_save=True, do_show=False, fig_path=f'{figfolder}/{location}_dem.png')
        # sim.plot(do_save=do_save, do_show=True, fig_path=f'{figfolder}/{location}_basic_epi.png')
        sim.plot(do_save=do_save, do_show=True, to_plot=to_plot, fig_path=f'{figfolder}/{location}_basic_epi.png')
        # ap.plot(do_save=True, do_show=False, fig_path=f'{figfolder}/{location}_age_pyramids.png')
        az.plot(do_save=do_save, do_show=True, fig_path=f'{figfolder}/{location}_cancer_by_age.png')

    return sim



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim = test_demo(datafile=f'test_data/{location}_data.csv')

    sc.toc(T)
    print('Done.')
