'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import seaborn as sns
import hpvsim as hpv
import pytest

do_plot = 0
do_save = 0

n_agents = [2e3,50e3][0] # Swap between sizes

base_pars = {
    'n_agents': n_agents,
    'start': 1990,
    'burnin': 30,
    'end': 2050,
    'genotypes': [16, 18],
    'location': 'tanzania',
    'dt': .5,
}


#%% Define the tests


def test_new_interventions(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test new intervention implementation')

    verbose = .1
    debug = 0

    ### Create interventions
    # Screen, triage, assign treatment, treat
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    routine_screen = hpv.routine_screening(
        product='via',  # pass in string or product
        prob=0.03,  # 3% annual screening probability/year over 30-50 implies ~60% of people will get a screen
        eligibility=screen_eligible,  # pass in valid state of People OR indices OR callable that gets indices
        age_range=[30, 50],
        start_year=2020,
        label='routine screening',
    )

    campaign_screen = hpv.campaign_screening(
        product='via',
        prob=0.3,
        age_range=[30, 70],
        years=2030,
        label='campaign screening',
    )

    # SOC: use a secondary diagnostic to determine how to treat people who screen positive
    to_triage = lambda sim: sim.get_intervention('routine screening').outcomes['positive']
    soc_triage = hpv.routine_triage(
        years = [2020,2029],
        prob = 0.5, # acceptance rate
        product = 'via_triage',
        eligibility = to_triage,
        label = 'VIA triage (pre-txvx)'
    )

    #### New protocol: for those who screen positive, decide whether to immediately offer TxVx or refer them for further testing
    screened_pos = lambda sim: list(set(sim.get_intervention('routine screening').outcomes['positive'].tolist() + sim.get_intervention('campaign screening').outcomes['positive'].tolist()))
    pos_screen_assesser = hpv.routine_triage(
        start_year=2030,
        prob = 1.0,
        product = 'txvx_assigner',
        eligibility = screened_pos,
        label = 'txvx assigner'
    )

    # Do further testing for those who were referred for further testing
    to_triage_new = lambda sim: sim.get_intervention('txvx assigner').outcomes['triage']
    new_triage = hpv.routine_triage(
        start_year = 2030,
        prob = 0.3,
        product = 'via_triage',
        eligibility = to_triage_new,
        label = 'VIA triage (post-txvx)'
    )

    # Get people who've been classified as txvx eligible based on the positive screen assessment, and deliver txvx to them
    txvx_eligible = lambda sim: sim.get_intervention('txvx assigner').outcomes['txvx']
    deliver_txvx = hpv.deliver_txvx(
        accept_prob = 0.8,
        product = 'txvx1',
        eligibility = txvx_eligible,
        label = 'txvx'
    )

    # New and old protocol: for those who've been confirmed positive in their secondary diagnostic, determine what kind of treatment to offer them
    confirmed_positive = lambda sim: list(set(sim.get_intervention('VIA triage (pre-txvx)').outcomes['positive'].tolist() + sim.get_intervention('VIA triage (post-txvx)').outcomes['positive'].tolist()))
    assign_treatment = hpv.routine_triage(
        prob = 1.0,
        product = 'tx_assigner',
        eligibility = confirmed_positive,
        label = 'tx assigner'
    )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        accept_prob = 0.5,
        max_capacity = 100,
        product = 'ablation',
        eligibility = ablation_eligible,
        label = 'ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'] + sim.get_intervention('ablation').outcomes['unsuccessful']))
    excision = hpv.treat_delay(
        accept_prob = 0.5,
        delay = 0.5,
        product = 'excision',
        eligibility = excision_eligible,
        label = 'excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_delay(
        accept_prob = 0.01,
        delay = 1.0,
        product = hpv.radiation(),
        eligibility = radiation_eligible,
        label = 'radiation'
    )

    soc_screen = [routine_screen, campaign_screen, soc_triage]
    new_screen = [pos_screen_assesser, new_triage,  deliver_txvx]
    triage_treat = [assign_treatment, ablation, excision, radiation]
    st_interventions = soc_screen + new_screen + triage_treat

    ## Vaccination interventions
    routine_years = np.arange(2020, base_pars['end']+1, dtype=int)
    routine_values = np.array([0,0,0,.1,.2,.3,.4,.5,.6,.7,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8])

    routine_vx = hpv.routine_vx(
        prob = routine_values,
        years = routine_years,
        product = 'bivalent',
        age_range=(9,10),
        label = 'routine vx'
    )

    campaign_vx = hpv.campaign_vx(
        prob = 0.9,
        years = 2023,
        product = 'bivalent',
        age_range=(9,14),
        label = 'campaign vx'
    )

    second_dose_eligible = lambda sim: (sim.people.doses == 1) | (sim.t > (sim.people.date_vaccinated + 0.5 / sim['dt']))
    second_dose = hpv.routine_vx(
        prob = 0.1,
        product = 'bivalent2',
        eligibility = second_dose_eligible,
        label = '2nd dose routine'
    )

    vx_interventions = [routine_vx, campaign_vx, second_dose]

    interventions =  st_interventions + vx_interventions
    for intv in interventions: intv.do_plot=False

    sim = hpv.Sim(pars=base_pars, interventions=interventions)
    sim.run()
    to_plot = {
        'Screens': ['resources_routine screening', 'resources_campaign screening'],
        'Vaccines': ['resources_routine vx', 'resources_campaign vx'],
        'Therapeutic vaccine': ['resources_txvx'],
        'Treatments': ['resources_ablation', 'resources_excision', 'resources_radiation'],
    }
    sim.plot(to_plot=to_plot)

    return sim




#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim = test_new_interventions(do_plot=do_plot)


    sc.toc(T)
    print('Done.')
