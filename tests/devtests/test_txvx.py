'''
Tests
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv


do_plot = 0
do_save = 0

n_agents = [1e3,50e3][1] # Swap between sizes

base_pars = {
    'n_agents': n_agents,
    'start': 1990,
    'end': 2050,
    'genotypes': [16, 18],
    'location': 'tanzania',
    'dt': 0.5,
    'hpv_control_prob': 0.0,
    'use_multiscale': True,
    'ms_agent_ratio': 100
}


#%% Define the tests
def make_mv_ints():
    sc.heading('Making mass vaccination interventions')

    ### Create interventions
    mass_vac_campaign_txvx_dose1 = hpv.campaign_txvx(
        annual_prob=False,
        prob=1,
        years=[2030],
        age_range=[25, 50],
        product='txvx1',
        label='campaign txvx'
    )

    second_dose_eligible = lambda sim: (sim.people.txvx_doses == 1)
    mass_vac_campaign_txvx_dose2 = hpv.campaign_txvx(
        annual_prob=False,
        prob=1,
        years=[2030],
        product='txvx2',
        eligibility=second_dose_eligible,
        label='campaign txvx 2nd dose'
    )

    mass_vac_routine_txvx_dose1 = hpv.routine_txvx(
        annual_prob=False,
        prob=1,
        start_year=2031,
        age_range=[25, 26],
        product='txvx1',
        label='routine txvx'
    )

    mass_vac_routine_txvx_dose2 = hpv.routine_txvx(
        annual_prob=False,
        prob=1,
        start_year=2031,
        age_range=[25, 26],
        product='txvx2',
        eligibility=second_dose_eligible,
        label='routine txvx 2nd dose'
    )
    mv_ints = [
        mass_vac_campaign_txvx_dose1,
        mass_vac_campaign_txvx_dose2,
        mass_vac_routine_txvx_dose1,
        mass_vac_routine_txvx_dose2
    ]

    return mv_ints


def make_tnv_ints(product=None):
    sc.heading('Making test & vaccination interventions')

    if product=='perf_sens':
        import pandas as pd
        hpv_test_pars = pd.read_csv('hpv_test_pars.csv')
        test_product = hpv.dx(hpv_test_pars, hierarchy=['positive', 'inadequate', 'negative'])
    else:
        test_product = 'hpv'


    # test and vaccinate
    # Run a one-time campaign to test & vaccinate everyone aged 25-50
    test_eligible = lambda sim: ((sim.people.txvx_doses==0) & (sim.people.screens==0))
    test_and_vac_txvx_campaign_testing = hpv.campaign_screening(
        annual_prob=False,
        product=test_product,
        prob=1.0,
        eligibility=test_eligible,
        age_range=[25,50],
        years=[2030],
        label='txvx_campaign_testing'
    )

    # In addition, run routine vaccination of everyone aged 25
    test_eligible = lambda sim: ((sim.people.txvx_doses==0) & (sim.people.screens==0))
    test_and_vac_txvx_routine_testing = hpv.routine_screening(
        annual_prob=False,
        product=test_product,
        prob=1.0,
        eligibility=test_eligible,
        age_range=[25,26],
        start_year=2031,
        label='txvx_routine_testing'
    )

    screened_pos = lambda sim: list(set(sim.get_intervention('txvx_routine_testing').outcomes['positive'].tolist()
                                        + sim.get_intervention('txvx_campaign_testing').outcomes['positive'].tolist()))
    test_and_vac_deliver_txvx = hpv.linked_txvx(
        annual_prob=False,
        prob=1.0,
        product='txvx1',
        eligibility=screened_pos,
        label='txvx'
    )

    second_dose_eligible = lambda sim: (sim.people.txvx_doses == 1)
    test_and_vac_txvx_dose2 = hpv.linked_txvx(
        prob=1,
        annual_prob=False,
        product='txvx2',
        eligibility=second_dose_eligible,
        label='txvx 2nd dose'
    )

    tnv_ints = [
        test_and_vac_txvx_campaign_testing,
        test_and_vac_txvx_routine_testing,
        test_and_vac_deliver_txvx,
        test_and_vac_txvx_dose2,
    ]

    return tnv_ints



def test_single(which='tnv', product=None, do_plot=False, do_save=False, fig_path=None):
    sc.heading(f'Testing {which}')

    if which == 'tnv':
        ints = make_tnv_ints(product=product)
    elif which == 'mv':
        ints = make_mv_ints()

    sim = hpv.Sim(pars=base_pars, interventions=ints)
    sim.run()
    to_plot = {
        'Total tx vaccinated': ['n_tx_vaccinated'],
        'Newly tx vaccinated': ['new_tx_vaccinated'],
        'Cumulative tx vaccinated': ['cum_tx_vaccinated'],
        'Tx_doses': ['new_txvx_doses']
    }
    sim.plot(to_plot=to_plot)
    return sim


def test_both(debug_scens=0):
    sc.heading('Testing T&V and MV')

    tnv_ints = make_tnv_ints()
    mv_ints = make_mv_ints()
    tvn_perf_sens_ints = make_tnv_perf_sens_ints()
    base_sim = hpv.Sim(pars=base_pars)

    scenarios = {
        # 'No vaccination': {
        #     'name': 'No vaccination',
        #     'pars': {
        #         'interventions': []
        #     }
        # },
        # 'Mass vaccination': {
        #     'name': 'Mass vaccination',
        #     'pars': {
        #         'interventions': mv_ints
        #     }
        # },
        # 'Test and vaccinate': {
        #     'name': 'Test and vaccinate',
        #     'pars': {
        #         'interventions': tnv_ints
        #     }
        # },
        'Test and vaccinate perf sens': {
            'name': 'Test and vaccinate perf sens',
            'pars': {
                'interventions': tvn_perf_sens_ints
            }
        },
    }

    metapars = {'n_runs': 1}
    scens = hpv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(debug=debug_scens, keep_people=True)
    to_plot = {
        'Total tx vaccinated': ['n_tx_vaccinated'],
        'Newly tx vaccinated': ['new_tx_vaccinated'],
        'CINs': ['total_cins'],
        'Cancers': ['total_cancers'],
    }
    scens.plot(to_plot=to_plot)
    return scens



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # sim = test_single(which='mv')
    sim = test_single(which='tnv', product='perf_sens')
    # scens0 = test_both(debug_scens = 0)


    sc.toc(T)
    print('Done.')
