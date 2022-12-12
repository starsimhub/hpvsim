'''
Tests for analyzers
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
import matplotlib.pyplot as plt

do_plot = 1
do_save = 0
n_agents = 5e3


#%% Define the tests
def test_snapshot():

    sc.heading('Testing snapshot analyzer')

    pars = dict(n_years=10, dt=0.5)

    sim = hpv.Sim(pars, analyzers=hpv.snapshot(['2016', '2019']))
    sim.run()
    snapshot = sim.get_analyzer()
    people1 = snapshot.snapshots[0]         # Option 1
    people2 = snapshot.snapshots['2016.0']  # Option 2
    people3 = snapshot.snapshots['2019.0']
    people4 = snapshot.get()                # Option 3

    assert people1 == people2, 'Snapshot options should match but do not'
    assert people3 != people4, 'Snapshot options should not match but do'
    return people4


def test_age_pyramids(do_plot=True):

    sc.heading('Testing age pyramids')

    pars = dict(n_agents=n_agents, start=2000, n_years=30, dt=0.5)

    # Loop over countries and their population sizes in the year 2000
    for country in ['south_africa', 'australia']:

        age_pyr = hpv.age_pyramid(
            timepoints=['2010', '2020'],
            datafile=f'test_data/{country}_age_pyramid.csv',
            edges=np.linspace(0, 100, 21))

        sim = hpv.Sim(
            pars,
            location = country.replace('_',' '),
            analyzers=age_pyr)

        sim.run()
        a = sim.get_analyzer()

        # Check plot()
        if do_plot:
            fig = a.plot(percentages=True)

    return sim, a


def test_age_results(do_plot=True):

    sc.heading('Testing by-age results')

    pars = dict(n_agents=n_agents, start=1970, n_years=50, dt=0.5, network='default', location='tanzania')
    pars['beta'] = .5

    pars['init_hpv_prev'] = {
        'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
        'm'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
        'f'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
    }
    az1 = hpv.age_results(
        result_keys=sc.objdict(
            hpv_prevalence=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            ),
            infections=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            ),
            cancer_incidence=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0.,20.,25.,30.,40.,45.,50.,55.,65.,100.]),
            ),
            cancer_mortality=sc.objdict(
                timepoints=['2019'],
                edges=np.array([0., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            )
        )
    )

    sim = hpv.Sim(pars, genotypes=[16, 18], analyzers=[az1])
    sim.run()
    a = sim.get_analyzer('age_results')

    # Check plot()
    if do_plot: a.plot()

    return sim, a


def test_reduce_analyzers():

    sc.heading('Test reducing analyzers')

    # Test averaging
    locations = ['kenya', 'tanzania']
    age_pyramids = []
    age_results = []
    pars = dict(n_agents=5e3, start=2000, n_years=30, dt=0.5)

    for location in locations:

        age_pyr = hpv.age_pyramid(
            timepoints=['2020'],
            datafile=f'test_data/{location}_age_pyramid.csv',
            edges=np.linspace(0, 100, 21))

        az = hpv.age_results(
            result_keys=sc.objdict(
                cancer_incidence=sc.objdict(
                    timepoints=['2020'],
                    edges=np.array([0.,15.,20.,25.,30.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.]),
                ),
                cancer_mortality=sc.objdict(
                    timepoints=['2020'],
                    edges=np.array([0.,15.,20.,25.,30.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.]),
                )
            )
        )

        sim = hpv.Sim(
            pars,
            location=location,
            analyzers=[age_pyr, az])

        sim.run()

        age_pyr = sim.get_analyzer('age_pyramid')
        age_pyramids.append(age_pyr)
        age_res = sim.get_analyzer('age_results')
        age_results.append(age_res)

    # reduced_analyzer = hpv.age_pyramid.reduce(age_pyramids)
    reduced_analyzer = hpv.age_results.reduce(age_results)

    return sim, reduced_analyzer


def test_age_causal_analyzer(do_plot=True):
    sc.heading('Test age causal infection analyzer')

    pars = {
        'n_agents': n_agents,
        'n_years': 70,
        'burnin': 50,
        'start': 1950,
        'genotypes': [16, 18],
        'location': 'tanzania',
        'network': 'default',
        'hpv_control_prob': 0.9,
        'debut': dict(f=dict(dist='normal', par1=14.0, par2=2.0),
                      m=dict(dist='normal', par1=16.0, par2=2.5)),
        'dt': 0.5,
    }
    pars['init_hpv_prev'] = {
        'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
        'm'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
        'f'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
    }

    sim = hpv.Sim(pars=pars, analyzers=hpv.age_causal_infection(start_year=2000))
    sim.run()

    a = sim.get_analyzer('age_causal_infection')
    count, bins_count = np.histogram(a.age_causal, bins=10)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    if do_plot:
        plt.figure()
        plt.plot(bins_count[1:], cdf)
        plt.title('Distribution of age of causal HPV infection')
        plt.legend()
        plt.xlabel('Age')
        plt.show()

    return sim, a


def test_detection():
    sc.heading('Test detection analyzers')

    pars = {
        'n_agents': n_agents,
        'n_years': 50,
        'burnin': 20,
        'start': 1980,
        'genotypes': [16, 18],
        'location': 'tanzania',
        'dt': 1.,
    }

    sim = hpv.Sim(pars=pars, analyzers='cancer_detection')
    sim.run()
    a = sim.get_analyzer(hpv.cancer_detection)

    return sim, a


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    people      = test_snapshot()
    sim0, a0    = test_age_pyramids()
    sim1, a1    = test_age_results()
    sim2, a2    = test_reduce_analyzers()
    sim3, a3    = test_age_causal_analyzer()
    sim4, a4    = test_detection()

    sc.toc(T)
    print('Done.')
