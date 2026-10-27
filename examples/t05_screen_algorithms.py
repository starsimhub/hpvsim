'''
Construct the 7 screen and treat algorithms recommended by the WHO
See documentation here: https://www.ncbi.nlm.nih.gov/books/NBK572308/

Each step of an algorithm is a delivery of an HPVsim product (``ss.screen``,
``ss.treat``), and each later step is triggered by a result of an earlier one
(e.g. ``who=cytology.ascus``), on the same step.
'''

import starsim as ss
import hpvsim as hpv

debug = 1

# Shared parameters
women = ss.ppl.female & (ss.ppl.age >= 30) & (ss.ppl.age < 50) # Who is screened
primary_screen_prob = 0.2 # Each year
triage_screen_prob = 0.9 # Of those referred
ablate_prob = 0.9 # Of those referred
start_year = 2025


def make_sim(seed=0, **kwargs):
    ''' Make a single sim '''
    pars = dict(
        n_agents        = [50_000, 5_000][debug],
        dt              = [0.5, 1.0][debug],
        start           = [1975, 2000][debug],
        stop            = 2060,
        ms_agent_ratio  = 10,
        rand_seed       = seed,
    )
    return hpv.Sim(**(pars | kwargs))


# Each step is named (name=), since steps made in a function cannot take the name of a variable, and later steps refer to them
def primary(product, name='screening'):
    ''' Primary screening: the women eligible, each year '''
    return ss.screen(hpv.dx(product), where=women, coverage=primary_screen_prob, interval=ss.years(1), start=start_year, name=name)


def triage(product, who, name):
    ''' A test for those referred by an earlier step '''
    return ss.screen(hpv.dx(product), who=who, coverage=triage_screen_prob, name=name)


def ablate(who):
    ''' Ablation for those referred, unless they have cancer '''
    return ss.treat(hpv.tx('ablation'), who=who & hpv.no_cancer, coverage=ablate_prob, name='ablation')


def algorithm1():
    ''' VIA as the primary screening test, followed by treatment '''
    via = primary('via', name='via_screening')
    return [via, ablate(via.positive)]


def algorithm2():
    ''' HPV testing, then immediate ablation for anyone eligible '''
    screening = primary('hpv')
    return [screening, ablate(screening.positive)]


def algorithm3():
    ''' Cytology, triage of ASCUS with HPV, colposcopy for HPV+ and abnormal cytology, then ablation of HSILs '''
    cytology = primary('lbc', name='cytology')
    hpv_triage = triage('hpv', cytology.ascus, name='hpv_triage')
    colpo = triage('colposcopy', cytology.abnormal | hpv_triage.positive, name='colpo')
    return [cytology, hpv_triage, colpo, ablate(colpo.hsil)]


def algorithm4():
    ''' HPV DNA with HPV16/18 typing, VIA triage for the other types, then treatment as assigned '''
    screening = primary('hpv_type')
    via_triage = triage('via', screening.positive_ohr, name='via_triage')
    assigner = triage('tx_assigner', screening.positive_1618 | via_triage.positive, name='assigner')
    return [screening, via_triage, assigner, ablate(assigner.ablation)]


def algorithm5():
    ''' HPV DNA, then VIA triage, then treatment as assigned '''
    screening = primary('hpv')
    via_triage = triage('via', screening.positive, name='via_triage')
    assigner = triage('tx_assigner', via_triage.positive, name='assigner')
    return [screening, via_triage, assigner, ablate(assigner.ablation)]


def algorithm6():
    ''' HPV DNA, then colposcopy triage, then ablation of HSILs '''
    screening = primary('hpv')
    colpo = triage('colposcopy', screening.positive, name='colpo')
    return [screening, colpo, ablate(colpo.hsil)]


def algorithm7():
    ''' HPV DNA, then cytology triage, then colposcopy, then ablation of HSILs '''
    screening = primary('hpv')
    cytology = triage('lbc', screening.positive, name='cytology')
    colpo = triage('colposcopy', cytology.abnormal | cytology.ascus, name='colpo')
    return [screening, cytology, colpo, ablate(colpo.hsil)]


algorithms = [algorithm1, algorithm2, algorithm3, algorithm4, algorithm5, algorithm6, algorithm7]


def make_algorithms(seed=0):
    ''' Run the seven algorithms and a baseline with no screening '''
    sims = [make_sim(seed=seed, label='No screening')]
    sims += [make_sim(seed=seed, interventions=algo(), label=f'Algorithm {i+1}') for i,algo in enumerate(algorithms)]
    return ss.parallel(sims)


#%% Run as a script
if __name__ == '__main__':
    msim = make_algorithms()
    print(msim.summarize())
