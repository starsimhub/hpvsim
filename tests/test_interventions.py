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

    # Define products
    import pandas as pd
    dfdx = pd.read_csv('../hpvsim/screen_products1.csv')
    dfvx = pd.read_csv('../hpvsim/vx_products.csv')

    # Create screening and triage products
    via_primary = hpv.dx(dfdx[dfdx.name == 'via'],              hierarchy=['positive', 'inadequate', 'negative'])
    via_triage  = hpv.dx(dfdx[dfdx.name == 'via_triage'],       hierarchy=['positive', 'inadequate', 'negative'])
    tx_assigner = hpv.dx(dfdx[dfdx.name == 'treatment_triage'], hierarchy=['radiation', 'excision', 'ablation', 'none'])

    # Create vaccines - first and second dose
    bivalent    = hpv.vx(genotype_pars = dfvx[dfvx.name == 'bivalent'], imm_init=dict(dist='beta', par1=30, par2=2))
    bivalent2   = hpv.vx(genotype_pars = dfvx[dfvx.name == 'bivalent'], imm_boost=1.2)

    # Create treatment products
    abl_prod    = hpv.tx(efficacy = dict(precin=0, cin1=0.936, cin2=0.936, cin3=0.936))
    exc_prod    = hpv.tx(efficacy = dict(precin=0, cin1=0.81,  cin2=0.81,  cin3=0.81))

    ### Create interventions
    # Screen, triage, assign treatment, treat
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    routine_screen = hpv.routine_screening(
        product=via_primary,  # pass in string or product
        prob=0.03,  # 3% annual screening probability/year over 30-50 implies ~60% of people will get a screen
        eligibility=screen_eligible,  # pass in valid state of People OR indices OR callable that gets indices
        age_range=[30, 50],
        start_year=2020,
        label='screening',
    )

    campaign_screen = hpv.campaign_screening(
        product=via_primary,
        prob=0.3,
        age_range=[30, 70],
        years=2025,
        label='campaign screening',
    )

    triage_eligible = lambda sim: sim.get_intervention('screening').outcomes['positive']
    triage = hpv.triage(
        product = via_triage,
        triage_prob = .5,
        eligibility = triage_eligible,
        label='triage',
    )

    confirmed_positive = lambda sim: sim.get_intervention('triage').outcomes['positive']
    assign_treatment = hpv.triage(
        triage_prob = 1.0,
        product = tx_assigner,
        eligibility = confirmed_positive,
        label = 'treatment_triage'
    )

    ablation_eligible = lambda sim: sim.get_intervention('treatment_triage').outcomes['ablation']
    ablation = hpv.treat_num(
        treat_prob = 0.5,
        max_capacity = 100,
        product = abl_prod,
        eligibility = ablation_eligible,
        label = 'ablation'
    )

    excision_eligible = lambda sim: sim.get_intervention('treatment_triage').outcomes['excision'] | sim.get_intervention('ablation').outcomes['unsuccessful']
    excision = hpv.treat_delay(
        treat_prob = 0.5,
        delay = 0.5,
        product = exc_prod,
        eligibility = excision_eligible,
        label = 'excision'
    )

    st_interventions  = [campaign_screen, routine_screen, triage, assign_treatment, ablation, excision]
    # TODO: cancer treatment not included yet<<<<<

    ## Vaccination interventions
    routine_years = np.arange(2020, base_pars['end']+1, dtype=int)
    routine_values = np.array([0,0,0,.1,.2,.3,.4,.5,.6,.7,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8])

    routine_vx = hpv.routine_vx(
        prob = routine_values,
        years = routine_years,
        product = bivalent,
        age_range=(9,10),
        label = 'routine vx'
    )

    campaign_vx = hpv.campaign_vx(
        prob = 0.9,
        years = 2023,
        product = bivalent,
        age_range=(9,14),
        label = 'campaign vx'
    )

    second_dose_eligible = lambda sim: (sim.people.doses == 1) | (sim.t > (sim.people.date_vaccinated + 0.5 / sim['dt']))
    second_dose = hpv.routine_vx(
        prob = 0.1,
        product = bivalent2,
        eligibility = second_dose_eligible,
        label = '2nd dose'
    )

    vx_interventions = [routine_vx, campaign_vx, second_dose]

    interventions = st_interventions + vx_interventions

    sim = hpv.Sim(pars=base_pars, interventions=interventions)
    sim.run()

    return sim




#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim = test_new_interventions(do_plot=do_plot)


    sc.toc(T)
    print('Done.')
