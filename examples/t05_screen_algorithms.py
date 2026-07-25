'''
Construct the 7 screen and treat algorithms recommended by the WHO
See documentation here: https://www.ncbi.nlm.nih.gov/books/NBK572308/
'''

import numpy as np
import starsim as ss
import hpvsim as hpv

debug = 1

def make_sim(seed=0):
    ''' Make a single sim '''

    # Parameters (v3: `stop` not `end`; pass as keyword arguments)
    pars = dict(
        n_agents        = [50_000, 5_000][debug],
        dt              = [0.5, 1.0][debug],
        start           = [1975, 2000][debug],
        stop            = 2060,
        ms_agent_ratio  = 10,
        rand_seed       = seed,
    )
    sim = hpv.Sim(**pars)
    return sim


def make_algorithms(sim=None, seed=0, debug=debug):

    if sim is None: sim = make_sim(seed=seed)

    # Shared parameters
    primary_screen_prob = 0.2
    triage_screen_prob = 0.9
    ablate_prob = 0.9
    start_year = 2025

    # Screening is delivered at `prob` each active year. A re-screen interval
    # (e.g. skip anyone screened in the last 5 years) can be built from the
    # screening module's `ti_screened` state.

    def union(*uid_arrays):
        ''' Combine outcome uid arrays into a single deduplicated ss.uids '''
        out = np.array([], dtype=int)
        for u in uid_arrays:
            out = np.union1d(out, np.asarray(u))
        return ss.uids(out)


    ####################################################################
    #### Algorithm 1 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # Visual inspection with acetic acid (VIA) as the primary screening test, followed by treatment
    ####################################################################

    via_primary = hpv.routine_screening(
        product='via',
        prob=primary_screen_prob,
        start_year=start_year,
        name='via primary',
    )

    via_positive = lambda sim: sim.interventions['via primary'].outcomes['positive']
    ablation1 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = via_positive,
    )

    algo1 = [via_primary, ablation1]

    ####################################################################
    #### Algorithm 2 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV testing, then immediate ablation for anyone eligible
    ####################################################################

    hpv_primary = hpv.routine_screening(
        product='hpv',
        prob=primary_screen_prob,
        start_year=start_year,
        name='hpv primary',
    )

    hpv_positive = lambda sim: sim.interventions['hpv primary'].outcomes['positive']
    ablation2 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = hpv_positive,
    )

    algo2 = [hpv_primary, ablation2]

    ####################################################################
    #### Algorithm 3 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # Cytology testing, triage ASCUS results with HPV, triage all HPV+ and
    # abnormal cytology results with colposcopy/biopsy, then ablation for all
    # HSILs
    ####################################################################

    cytology = hpv.routine_screening(
        product='lbc',
        prob=primary_screen_prob,
        start_year=start_year,
        name='cytology',
    )

    # Triage ASCUS with HPV test
    ascus = lambda sim: sim.interventions['cytology'].outcomes['ascus']
    hpv_triage = hpv.routine_triage(
        product='hpv',
        prob=triage_screen_prob,
        eligibility=ascus,
        name='hpv triage',
    )

    # Send abnormal cytology results, plus ASCUS results that were HPV+, for colpo
    to_colpo = lambda sim: union(sim.interventions['cytology'].outcomes['abnormal'],
                                 sim.interventions['hpv triage'].outcomes['positive'])
    colpo = hpv.routine_triage(
        product='colposcopy',
        prob = triage_screen_prob,
        eligibility=to_colpo,
        name = 'colpo',
    )

    # After colpo, treat HSILs with ablation
    hsils = lambda sim: sim.interventions['colpo'].outcomes['hsil']
    ablation3 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = hsils,
    )

    algo3 = [cytology, hpv_triage, colpo, ablation3]


    ####################################################################
    #### Algorithm 4 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV DNA as the primary screening test, followed by HPV16/18 triage
    # (when already part of the HPV test), followed by treatment,
    # and using VIA triage for those who screen negative for HPV16/18
    ####################################################################

    hpv_primary4 = hpv.routine_screening(
        product='hpv_type',
        prob=primary_screen_prob,
        start_year=start_year,
        name='hpv primary',
    )

    # Those who test + for OHR types are triaged with VIA
    pos_ohr = lambda sim: sim.interventions['hpv primary'].outcomes['positive_ohr']
    via_triage = hpv.routine_triage(
        product='via',
        prob=triage_screen_prob,
        eligibility=pos_ohr,
        name='via triage',
    )

    # Determine ablation eligibility for people with 16/18, plus those who test positive from VIA
    to_assign = lambda sim: union(sim.interventions['hpv primary'].outcomes['positive_1618'],
                                  sim.interventions['via triage'].outcomes['positive'])
    tx_assigner = hpv.routine_triage(
        product='tx_assigner',
        prob = triage_screen_prob,
        eligibility=to_assign,
        name = 'tx assigner',
    )

    # Ablate anyone eligible for ablation
    to_ablate = lambda sim: sim.interventions['tx assigner'].outcomes['ablation']
    ablation4 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = to_ablate,
    )

    algo4 = [hpv_primary4, via_triage, tx_assigner, ablation4]


    ####################################################################
    #### Algorithm 5 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV DNA as the primary screening test, followed by VIA triage,
    # followed by treatment
    ####################################################################

    hpv_primary5 = hpv.routine_screening(
        product='hpv',
        prob=primary_screen_prob,
        start_year=start_year,
        name='hpv primary',
    )

    # Those who test + are triaged with VIA
    screen_pos = lambda sim: sim.interventions['hpv primary'].outcomes['positive']
    via_triage5 = hpv.routine_triage(
        product='via',
        prob=triage_screen_prob,
        eligibility=screen_pos,
        name='via triage',
    )

    # Determine ablation eligibility
    to_assign5 = lambda sim: sim.interventions['via triage'].outcomes['positive']
    tx_assigner5 = hpv.routine_triage(
        product='tx_assigner',
        prob = triage_screen_prob,
        eligibility=to_assign5,
        name = 'tx assigner',
    )

    # Ablate anyone eligible for ablation
    to_ablate = lambda sim: sim.interventions['tx assigner'].outcomes['ablation']
    ablation5 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = to_ablate,
    )

    algo5 = [hpv_primary5, via_triage5, tx_assigner5, ablation5]


    ####################################################################
    #### Algorithm 6 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # High-risk HPV DNA as the primary screening test, followed by
    # colposcopy triage, followed by treatment
    ####################################################################

    hpv_primary6 = hpv.routine_screening(
        product='hpv',
        prob=primary_screen_prob,
        start_year=start_year,
        name='hpv primary',
    )

    # Send HPV+ women for colpo
    to_colpo = lambda sim: sim.interventions['hpv primary'].outcomes['positive']
    colpo6 = hpv.routine_triage(
        product='colposcopy',
        prob = triage_screen_prob,
        eligibility=to_colpo,
        name = 'colpo',
    )

    # After colpo, treat HSILs with ablation
    hsils = lambda sim: sim.interventions['colpo'].outcomes['hsil']
    ablation6 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = hsils,
    )

    algo6 = [hpv_primary6, colpo6, ablation6]

    ####################################################################
    #### Algorithm 7 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV DNA as the primary screening test, followed by cytology triage,
    # followed by colposcopy and treatment
    ####################################################################

    hpv_primary7 = hpv.routine_screening(
        product='hpv',
        prob=primary_screen_prob,
        start_year=start_year,
        name='hpv primary',
    )

    # Send HPV+ women for cytology
    to_cytology = lambda sim: sim.interventions['hpv primary'].outcomes['positive']
    cytology7 = hpv.routine_triage(
        product='lbc',
        prob=triage_screen_prob,
        eligibility=to_cytology,
        name='cytology',
    )

    # Send ASCUS and abnormal cytology results for colpo
    to_colpo = lambda sim: union(sim.interventions['cytology'].outcomes['abnormal'],
                                 sim.interventions['cytology'].outcomes['ascus'])
    colpo7 = hpv.routine_triage(
        product='colposcopy',
        prob=triage_screen_prob,
        eligibility=to_colpo,
        name='colpo',
    )

    # After colpo, treat HSILs with ablation
    hsils = lambda sim: sim.interventions['colpo'].outcomes['hsil']
    ablation7 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = hsils,
    )

    algo7 = [hpv_primary7, cytology7, colpo7, ablation7]


    ####################################################################
    #### Set up scenarios to compare algoriths
    ####################################################################

    # Create, run, and plot the simulations
    sim0 = hpv.Sim(label='No screening')
    sim1 = hpv.Sim(interventions=algo1, label='Algorithm 1')
    sim2 = hpv.Sim(interventions=algo2, label='Algorithm 2')
    sim3 = hpv.Sim(interventions=algo3, label='Algorithm 3')
    sim4 = hpv.Sim(interventions=algo4, label='Algorithm 4')
    sim5 = hpv.Sim(interventions=algo5, label='Algorithm 5')
    sim6 = hpv.Sim(interventions=algo6, label='Algorithm 6')
    sim7 = hpv.Sim(interventions=algo7, label='Algorithm 7')
    msim = ss.MultiSim([sim0, sim1, sim2, sim3, sim4, sim5, sim6, sim7]).run()

    return msim




#%% Run as a script
if __name__ == '__main__':

    msim = make_algorithms()
