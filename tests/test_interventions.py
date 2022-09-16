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


def test_complex_vax(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test complex roll-out of prophylactic vaccine')

    verbose = .1
    debug = 0

    # # Model an intervention to roll out prophylactic vaccination
    #
    # # Campaign vaccination
    # campaign_years = np.arange(2020, 2022, dtype=int)
    # campaign_values = 0.5
    # campaign_vx = hpv.RoutineVaccination(vaccine='bivalent', label='Campaign', age_range=(9, 24), coverage=campaign_values, timepoints=campaign_years)
    # interventions = [routine_vx, campaign_vx]

    # # Screening
    # ablation_compliance=0.5
    # excision_compliance=0.2
    # cancer_compliance = 0.1
    #
    # screen_years    = np.arange(2020, base_pars['end'], dtype=int)
    # screen_coverage = np.array([0,0,0,.1,.2,.3,.4,.5,.6,.7,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8])

    #### PROPOSED NEW ALGO

    # Define products
    import pandas as pd
    dfdx = pd.read_csv('../hpvsim/screen_products1.csv')
    dfvx = pd.read_csv('../hpvsim/vx_products.csv.csv')

    via_primary = hpv.dx(dfdx[dfdx.name == 'via'],              hierarchy=['positive', 'inadequate', 'negative'])
    via_triage  = hpv.dx(dfdx[dfdx.name == 'via_triage'],       hierarchy=['positive', 'inadequate', 'negative'])
    tx_assigner = hpv.dx(dfdx[dfdx.name == 'treatment_triage'], hierarchy=['radiation', 'excision', 'ablation', 'none'])

    bivalent    = hpv.vx(genotype_pars = dfvx[dfvx.name == 'bivalent'], imm_init=dict(dist='beta', par1=30, par2=2))
    bivalent2   = hpv.vx(genotype_pars = dfvx[dfvx.name == 'bivalent'], boost=1.2)

    abl_prod    = hpv.tx(efficacy = dict(precin=0, cin1=0.936, cin2=0.936, cin3=0.936))
    exc_prod    = hpv.tx(efficacy = dict(precin=0, cin1=0.81,  cin2=0.81,  cin3=0.81))

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    routine = hpv.RoutineScreening(
        product=via_primary,  # pass in string or product
        screen_prob=0.03,  # 3% annual screening probability/year over 30-50 implies ~60% of people will get a screen
        eligibility=screen_eligible,  # pass in valid state of People OR indices OR callable that gets indices
        age_range=[30, 50],
        start_year=2020,
        label='screening',
    )

    triage_eligible = lambda sim: sim.get_intervention('screening').outcomes['positive']
    triage = hpv.Triage(
        product = via_triage,
        triage_prob = .5,
        eligibility = triage_eligible,
        label='triage',
    )

    confirmed_positive = lambda sim: sim.get_intervention('triage').outcomes['positive']
    assign_treatment = hpv.Triage(
        triage_prob = 1.0,
        product = tx_assigner,
        eligibility = confirmed_positive,
        label = 'treatment_triage'
    )

    ablation_eligible = lambda sim: sim.get_intervention('treatment_triage').outcomes['ablation']
    ablation = hpv.NumTreat(
        treat_prob = 0.5,
        max_capacity = 100,
        product = abl_prod,
        eligibility = ablation_eligible,
        label = 'ablation'
    )

    excision_eligible = lambda sim: sim.get_intervention('treatment_triage').outcomes['excision'] | sim.get_intervention('ablation').outcomes['unsuccessful']
    excision = hpv.DelayTreat(
        treat_prob = 0.5,
        delay = 0.5,
        product = exc_prod,
        eligibility = excision_eligible,
        label = 'excision'
    )

    # Routine vaccination
    routine_years = np.arange(2020, base_pars['end'], dtype=int)
    routine_values = np.array([0,0,0,.1,.2,.3,.4,.5,.6,.7,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8,.8])
    routine_vx = hpv.RoutineVaccination(vaccine='bivalent', label='Routine', age_range=(9, 10), coverage=routine_values, timepoints=routine_years)

    second_dose = hpv.RoutineVaccination(
        coverage = routine_values,
        years = routine_years,
        product = bivalent,
        age_range=(9,10),
        eligibility = second_dose_eligible,
        label = '2nd dose'
    )

    second_dose_eligible = lambda sim: sim.get_intervention('vaccination').outcomes['vaccinated'] #| sim.get_intervention('ablation').outcomes['unsuccessful']
    second_dose = hpv.Vaccination(
        prob = 0.5,
        delay = 0.5,
        product = bivalent,
        eligibility = second_dose_eligible,
        label = '2nd dose'
    )


    interventions  = [routine, triage, assign_treatment, ablation, excision]

    sim = hpv.Sim(pars=base_pars, interventions=interventions)
    sim.run()

    return sim


    #
    #
    # triage_eligible = lambda sim: (sim.get_intervention('routine').states('positive')) & (not already_triaged)
    #
    # excision_eligible = lambda sim: sim.get_intervention('triage').get_states('needs_excision') | ablation.get_states('unsuccessful')
    # ablation = hpv.PrecancerTreatment(
    #     product='excision',
    #     eligibility=excision_eligible,
    #     treat_prob=0.1,
    #     start_year=2020,
    # )
    #
    # campaign = hpv.CampaignScreening(
    #     product=hpv,
    #     screen_prob=0.2,
    #     age_range=[30, 70],
    #     years=2030,
    #     states=['positive', 'negative']
    # )
    #


   #
    # txvx = hpv.TherapeuticVaccination(
    #     product = txvx,
    #     eligibility = campaign.get_inds('positive'), # Returns inds
    # )


    # class MyCampaign(hpv.Intervention):
    #     '''TBC'''
    #     def __init__(self):
    #         return
    #
    #     def apply(self, args):
    #         # do screening
    #         self.all_inds = {}
    #         self.all_inds['positive'] = [] # fill in
    #         self.all_inds['negative'] = [] # ditto
    #
    #         return all_inds
    #
    #     def get_inds(self, which, t):
    #         return self.all_inds[which]
    #
    #
    # txvx = hpv.TherapeuticVaccination(
    #     product = txvx,
    #     eligibility = campaign.get_inds('positive'), # Returns inds
    #     etc = {},
    # )
    #
    # campaign = hpv.CampaignScreening(
    #     product='hpv',
    #     eligibility='screen_eligible',
    #     screen_prob=0.3, # 3% annual screening probability/year over 30-50 implies ~60% of people will get a screen
    #     age_range=[30,50],
    #     years=[2020, 2025],
    #     next_step_pos = {'state': 'triage_eligible', 'interval': 0},
    #     next_step_neg = {'state': 'screen_eligible', 'interval': 5}
    # )




#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim = test_complex_vax(do_plot=do_plot)
    # from collections import defaultdict
    # aa = defaultdict(set)


    sc.toc(T)
    print('Done.')
