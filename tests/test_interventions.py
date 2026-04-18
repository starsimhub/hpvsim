'''
Tests for single simulations
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import hpvsim as hpv
import pandas as pd
import matplotlib.pyplot as plt

do_plot = 0
do_save = 0

n_agents = [500,50e3][0] # Swap between sizes

base_pars = {
    'n_agents': n_agents,
    'start': 1990,
    'end': 2050,
    'genotypes': [16, 18],
    'location': 'tanzania',
    'dt': 0.5,
}


#%% Define the tests

def test_screen_prob():
    sc.heading('Test that screening selects the right number of people')

    target_lifetime_prob = 0.6
    target_triage = 0.8
    target_treat = 1
    age_range = [30, 50]
    len_age_range = age_range[1]-age_range[0]
    model_annual_prob = 1 - (1 - target_lifetime_prob)**(1/len_age_range)
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) # Only model a single lifetime screen

    screen = hpv.routine_screening(
        product='via',
        prob=model_annual_prob,  # This looks like it means that we screen 50% of the population each year
        eligibility=screen_eligible,  # pass in valid state of People OR indices OR callable that gets indices
        age_range=age_range,
        start_year=2000,
        label='screen'
    )

    to_triage = lambda sim: sim.get_intervention('screen').outcomes['positive']
    triage = hpv.routine_triage(
        product='via',  # pass in string or product
        prob=target_triage,
        annual_prob=False,
        eligibility=to_triage,  # pass in valid state of People OR indices OR callable that gets indices
        label='triage'
    )

    to_treat = lambda sim: sim.get_intervention('triage').outcomes['positive']
    assign_tx = hpv.routine_triage(
        product='tx_assigner',
        prob=target_treat,
        annual_prob=False,
        eligibility=to_treat,
        label='assign_tx'
    )

    to_ablate = lambda sim: sim.get_intervention('assign_tx').outcomes['ablation']
    ablation = hpv.treat_num(
        product='ablation',  # pass in string or product
        prob=1,
        # annual_prob=False,
        eligibility=to_ablate,
        label='ablation'
    )

    to_excise = lambda sim: sim.get_intervention('assign_tx').outcomes['excision']
    excision = hpv.treat_delay(
        product='excision',  # pass in string or product
        prob=1,
        # annual_prob=False,
        eligibility=to_excise,
        label='excision'
    )

    # Check that the right number of people are getting screened, triaged, and treated
    class intv_counter(hpv.Analyzer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_screened = []
            self.n_screened_pos = []
            self.n_triaged = []
            self.n_triaged_pos = []
            self.n_assigned_tx = []
            self.n_assigned_ablation = []
            self.n_assigned_excision = []
            self.n_ablation = []
            self.n_excision = []

        def initialize(self, sim):
            super().initialize()
            self.yearvec = sim.yearvec

        def apply(self, sim):
            self.n_screened.append(sum([len(v) for v in sim.get_intervention('screen').outcomes.values()]))
            self.n_screened_pos.append(len(sim.get_intervention('screen').outcomes['positive']))
            self.n_triaged.append(sum([len(v) for v in sim.get_intervention('triage').outcomes.values()]))
            self.n_triaged_pos.append(len(sim.get_intervention('triage').outcomes['positive']))
            self.n_assigned_tx.append(sum([len(v) for v in sim.get_intervention('assign_tx').outcomes.values()]))
            self.n_assigned_ablation.append(len(sim.get_intervention('assign_tx').outcomes['ablation']))
            self.n_assigned_excision.append(len(sim.get_intervention('assign_tx').outcomes['excision']))
            self.n_ablation.append(sum([len(v) for v in sim.get_intervention('ablation').outcomes.values()]))
            self.n_excision.append(sum([len(v) for v in sim.get_intervention('excision').outcomes.values()]))

        def finalize(self, sim=None):
            for att in ['n_screened','n_screened_pos','n_triaged','n_triaged_pos','n_assigned_tx',
                        'n_assigned_ablation','n_assigned_excision',
                        'n_ablation','n_excision',]:
                setattr(self,att,sc.promotetoarray(getattr(self,att)))

    sim = hpv.Sim(pars=base_pars, interventions=[screen, triage, assign_tx, ablation, excision], analyzers=[intv_counter()])
    sim.run(verbose=0)

    # Check that the right number of people are getting screened
    n_eligible = len(hpv.true((sim.people.age >= 50) & (sim.people.age <= 70) & (sim.people.is_female) ))
    n_screened = len(hpv.true((sim.people.age >= 50) & (sim.people.age <= 70) & (sim.people.is_female) & (sim.people.screened)))
    print(f'Proportion screened ({n_screened/n_eligible:.2f}) vs target: ({target_lifetime_prob})')

    # Check that the right number of people are getting triaged and assigned treatment
    a = sim.get_analyzer()
    if target_triage==1:
        assert np.array_equal(a.n_screened_pos, a.n_triaged)
        assert np.array_equal(a.n_triaged_pos, a.n_assigned_tx)
    print(f'Proportion triaged ({sum(a.n_triaged)/sum(a.n_screened_pos):.2f}) vs target: ({target_triage})')
    print(f'Proportion assigned treatment ({sum(a.n_assigned_tx)/sum(a.n_triaged_pos):.2f}) vs target: ({target_treat})')
    print(f'Proportion actually treated ({(sum(a.n_ablation)+sum(a.n_excision))/sum(a.n_assigned_tx):.2f}), target depends on product')

    print('Note, mismatches in any of the above are inevitable with small pop sizes, but should be equal for large pop sizes.')

    return sim


def test_all_interventions(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Test all interventions together')

    ### Create interventions
    # Screen, triage, assign treatment, treat
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    routine_screen = hpv.routine_screening(
        product='hpv',
        prob=0.1,
        eligibility=screen_eligible,
        age_range=[30, 50],
        start_year=2020,
        label='routine screening',
    )

    campaign_screen = hpv.campaign_screening(
        product='hpv',
        prob=0.3,
        age_range=[30, 70],
        years=2030,
        label='campaign screening',
    )

    # SOC: use a secondary diagnostic to determine how to treat people who screen positive
    to_triage = lambda sim: sim.get_intervention('routine screening').outcomes['positive']
    soc_triage = hpv.routine_triage(
        years = [2020,2029],
        prob = 0.9, # acceptance rate
        annual_prob=False, # This probability is per timestep, not annual
        product = 'via',
        eligibility = to_triage,
        label = 'VIA triage (pre-txvx)'
    )

    #### New protocol: for those who screen positive, decide whether to immediately offer TxVx or refer them for further testing
    screened_pos = lambda sim: list(set(sim.get_intervention('routine screening').outcomes['positive'].tolist() + sim.get_intervention('campaign screening').outcomes['positive'].tolist()))
    pos_screen_assesser = hpv.routine_triage(
        start_year=2030,
        prob = 1.0,
        annual_prob=False,
        product = 'txvx_assigner',
        eligibility = screened_pos,
        label = 'txvx assigner'
    )

    # Do further testing for those who were referred for further testing
    to_triage_new = lambda sim: sim.get_intervention('txvx assigner').outcomes['triage']
    new_triage = hpv.routine_triage(
        start_year = 2030,
        prob = 1.0,
        product = 'via',
        eligibility = to_triage_new,
        label = 'VIA triage (post-txvx)'
    )

    # Get people who've been classified as txvx eligible based on the positive screen assessment, and deliver txvx to them
    txvx_eligible = lambda sim: sim.get_intervention('txvx assigner').outcomes['txvx']
    deliver_txvx = hpv.linked_txvx(
        prob = 0.8,
        product = 'txvx1',
        eligibility = txvx_eligible,
        label = 'txvx'
    )

    # New and old protocol: for those who've been confirmed positive in their secondary diagnostic, determine what kind of treatment to offer them
    confirmed_positive = lambda sim: list(set(sim.get_intervention('VIA triage (pre-txvx)').outcomes['positive'].tolist() + sim.get_intervention('VIA triage (post-txvx)').outcomes['positive'].tolist()))
    assign_treatment = hpv.routine_triage(
        prob = 1.0,
        annual_prob=False,
        product = 'tx_assigner',
        eligibility = confirmed_positive,
        label = 'tx assigner'
    )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        prob = 0.5,
        max_capacity = 100,
        product = 'ablation',
        eligibility = ablation_eligible,
        label = 'ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() + sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_delay(
        prob = 0.5,
        delay = 0.5,
        product = 'excision',
        eligibility = excision_eligible,
        label = 'excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_delay(
        prob = 0.01,
        delay = 1.0,
        product = hpv.radiation(),
        eligibility = radiation_eligible,
        label = 'radiation'
    )

    soc_screen = [routine_screen, campaign_screen, soc_triage]
    new_screen = [pos_screen_assesser, new_triage,  deliver_txvx]
    triage_treat = [assign_treatment, ablation, excision, radiation]
    st_interventions = soc_screen + triage_treat + new_screen

    ## Vaccination interventions
    routine_years = np.arange(2020, base_pars['end'], dtype=int)
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

    sim0 = hpv.Sim(pars=base_pars)
    sim0.run()
    sim = hpv.Sim(pars=base_pars, interventions=interventions)
    sim.run()
    to_plot = {
        'Screens': ['new_screens', 'new_screened'],
        'Vaccines': ['new_vaccinated', 'new_doses'],
        'Therapeutic vaccine': ['new_tx_vaccinated', 'new_txvx_doses'],
        'Treatments': ['new_cin_treatments', 'new_cin_treated'],
    }
    sim.plot(to_plot=to_plot)

    fig, ax = plt.subplots(1, 2)
    for i, result in enumerate(['cancers', 'cins']):
        ax[i].plot(sim0.results['year'], sim0.results[result].values, label='No Screening')
        ax[i].plot(sim.results['year'], sim.results[result].values, label='Screening')
        ax[i].set_ylabel(result)
        ax[i].legend()
    fig.show()


    return sim



def test_txvx_noscreen(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Testing TxVx rollout without screening')

    ### Create interventions
    # Campaign txvx
    campaign_txvx_dose1 = hpv.campaign_txvx(
        prob = 0.9,
        years = 2030,
        age_range = [30,50],
        product = 'txvx1',
        label = 'campaign txvx'
    )

    second_dose_eligible = lambda sim: (sim.people.txvx_doses == 1) | (sim.t > (sim.people.date_tx_vaccinated + 0.5 / sim['dt']))
    campaign_txvx_dose2 = hpv.campaign_txvx(
        prob = 0.7,
        years=2030,
        age_range=[30, 70],
        product = 'txvx2',
        eligibility = second_dose_eligible,
        label = 'campaign txvx 2nd dose'
    )

    routine_txvx_dose1 = hpv.routine_txvx(
        prob = 0.9,
        start_year = 2031,
        age_range = [30,31],
        product = 'txvx2',
        label = 'routine txvx'
    )

    second_dose_eligible = lambda sim: (sim.people.txvx_doses == 1) | (sim.t > (sim.people.date_tx_vaccinated + 0.5 / sim['dt']))
    routine_txvx_dose2 = hpv.routine_txvx(
        prob = 0.8,
        start_year = 2031,
        age_range = [30,31],
        product = 'txvx1',
        eligibility=second_dose_eligible,
        label = 'routine txvx 2nd dose'
    )

    interventions = [campaign_txvx_dose1, campaign_txvx_dose2, routine_txvx_dose1, routine_txvx_dose2]
    for intv in interventions: intv.do_plot=False

    sim = hpv.Sim(pars=base_pars, interventions=interventions)
    sim.run()
    to_plot = {
        'Number vaccinated': ['new_tx_vaccinated', 'cum_tx_vaccinated'],
    }
    sim.plot(to_plot=to_plot)

    return sim



def test_vx_effect(do_plot=False, do_save=False, fig_path=None):
    sc.heading('Testing effect of prophylactic vaccination')

    debug_scens = 1

    ### Create interventions
    routine_vx_dose1 = hpv.routine_vx(
        prob = 0.9,
        start_year = 2023,
        age_range = [9,10],
        product = 'bivalent',
        label = 'Bivalent dose 1'
    )

    second_dose_eligible = lambda sim: (sim.people.doses == 1) | (sim.t > (sim.people.date_vaccinated + 0.5 / sim['dt']))
    routine_vx_dose2 = hpv.routine_vx(
        prob = 0.8,
        start_year = 2023,
        product = 'bivalent2',
        eligibility=second_dose_eligible,
        label = 'Bivalent dose 2'
    )

    interventions = [routine_vx_dose1, routine_vx_dose2]
    for intv in interventions: intv.do_plot=False

    base_sim = hpv.Sim(pars=base_pars)

    scenarios = {
        'baseline': {
            'name': 'Baseline',
            'pars': {
            }
        },
        'vx scaleup': {
            'name': 'Vaccinate 90% of 9yos',
            'pars': {
                'interventions': interventions
            }
        },
    }

    metapars = {'n_runs': 3}
    scens = hpv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(debug=debug_scens)
    to_plot = {
        'HPV prevalence': ['hpv_prevalence'],
        'Age standardized cancer incidence (per 100,000 women)': ['asr_cancer_incidence'],
        'Cancer deaths per 100,000 women': ['cancer_mortality'],
    }
    scens.plot(to_plot=to_plot)
    return scens


def test_screening():
    sc.heading('Test new screening implementation')

    ### Create interventions
    # Screen, triage, assign treatment, treat
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    routine_screen = hpv.routine_screening(
        product='via',
        prob=1.0,
        eligibility=screen_eligible,
        age_range=[0, 100],
        start_year=2020,
        label='routine screening',
    )

    # New and old protocol: for those who've been confirmed positive in their secondary diagnostic, determine what kind of treatment to offer them
    screen_positive = lambda sim: sim.get_intervention('routine screening').outcomes['positive']
    assign_treatment = hpv.routine_triage(
        prob = 1.0,
        product = 'tx_assigner',
        eligibility = screen_positive,
        label = 'tx assigner'
    )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        prob = 1.0,
        product = 'ablation',
        eligibility = ablation_eligible,
        label = 'ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() + sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_num(
        prob = 1.0,
        product = 'excision',
        eligibility = excision_eligible,
        label = 'excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_num(
        prob = 1.0,
        product = hpv.radiation(),
        eligibility = radiation_eligible,
        label = 'radiation'
    )

    soc_screen = [routine_screen]
    triage_treat = [assign_treatment, ablation, excision, radiation]
    interventions = soc_screen + triage_treat


    for intv in interventions: intv.do_plot=False

    sim0 = hpv.Sim(pars=base_pars)
    sim0.run()
    sim = hpv.Sim(pars=base_pars, interventions=interventions)
    sim.run()
    to_plot = {
        'CINs': ['cins'],
        # 'Screens': ['screens'],
    }
    sim.plot(to_plot=to_plot)

    fig, ax = plt.subplots(1, 2)
    for i, result in enumerate(['cancers', 'cins']):
        ax[i].plot(sim0.results['year'], sim0.results[result].values, label='No Screening')
        ax[i].plot(sim.results['year'], sim.results[result].values, label='Screening')
        ax[i].set_ylabel(result)
        ax[i].legend()
    fig.show()


    return sim


def test_cytology():
    sc.heading('Test cytology vs HPV')

    # Algorithm 3 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened)# Single lifetime screen
    cytology = hpv.routine_screening(
        product='lbc',
        prob=1.0,
        eligibility=screen_eligible,
        start_year=2020,
        label='cytology',
    )

    # Triage ASCUS with HPV test
    ascus = lambda sim: sim.get_intervention('cytology').outcomes['ascus']
    hpv_triage = hpv.routine_triage(
        product='hpv',
        prob=.5,
        annual_prob=False,
        eligibility=ascus,
        label='hpv triage'
    )

    # Send abnormal cytology results, plus ASCUS results that were HPV+, for colpo
    to_colpo = lambda sim: list(set(sim.get_intervention('cytology').outcomes['abnormal'].tolist() + sim.get_intervention('hpv triage').outcomes['positive'].tolist()))
    colpo = hpv.routine_triage(
        product='colposcopy',
        prob = 0.5,
        annual_prob=False,
        eligibility=to_colpo,
        label = 'colposcopy'
    )

    # After colpo, treat HSILs with excision and LSILs with ablation
    lsils = lambda sim: sim.get_intervention('colposcopy').outcomes['lsil']
    ablation = hpv.treat_num(
        prob = 0.5,
        product = 'ablation',
        eligibility = lsils,
        label = 'ablation'
    )

    hsils = lambda sim: sim.get_intervention('colposcopy').outcomes['hsil']
    excision = hpv.treat_num(
        prob = 0.2,
        product = 'excision',
        eligibility = hsils,
        label = 'excision'
    )

    interventions = [cytology, hpv_triage, colpo, ablation, excision]

    for intv in interventions: intv.do_plot=False

    base_sim = hpv.Sim(pars=base_pars)

    scenarios = {
        'baseline': {
            'name': 'Baseline',
            'pars': {
            }
        },
        'algo3': {
            'name': 'Algorithm 3',
            'pars': {
                'interventions': interventions
            }
        },
    }

    metapars = {'n_runs': 3}
    scens = hpv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(debug=0)
    to_plot = {
        'HPV prevalence': ['hpv_prevalence'],
        'Age standardized cancer incidence (per 100,000 women)': ['asr_cancer_incidence'],
        'Cancer deaths per 100,000 women': ['cancer_mortality'],
    }
    scens.plot(to_plot=to_plot)
    return scens


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim0 = test_screen_prob()
    sim1 = test_all_interventions(do_plot=do_plot)
    sim2 = test_txvx_noscreen()
    sim3 = test_screening()
    scens0 = test_vx_effect()
    scens1 = test_cytology()

    sc.toc(T)
    print('Done.')
