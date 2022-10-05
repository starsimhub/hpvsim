'''
Create a demo project
'''

#%% Imports and settings
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

    pars = dict(n_agents=5e3,
                start=1950,
                end=2020,
                dt=.5,
                network='default',
                location=location,
                genotypes=[16,18,35,45,52,58],
                verbose=0.1
                )

    # Initial conditions
    pars['init_hpv_dist'] = {'hpv16': .347, 'hpv18': .114, 'hpv35': .174,
                             'hpv45': .116, 'hpv52': .097, 'hpv58': .121}
    pars['init_hpv_prev'] = {
        'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
        'm'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
        'f'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
    }

    pars['hpv_control_prob'] = 0.4
    pars['beta'] = 0.217

    # Initialize genotypes
    hpv16   = hpv.genotype('HPV16')
    hpv18   = hpv.genotype('HPV18')
    hpv35   = hpv.genotype('HPV35')
    hpv45   = hpv.genotype('HPV45')
    hpv52   = hpv.genotype('HPV52')
    hpv58   = hpv.genotype('HPV58')

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
            total_cancer_incidence=sc.objdict(
                datafile=f'test_data/{location}_cancer_incidence.csv',
            )
        )
    )

    analyzers = [az]
    genotypes = [hpv16, hpv18, hpv35, hpv45, hpv52, hpv58]
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
                'total_cancer_incidence',
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
