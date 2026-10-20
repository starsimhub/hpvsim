"""
HPV programs as Starsim deliveries: routine vaccination, and screen-and-treat algorithms
"""

import starsim as ss

from .products import vaccine, diagnostic, treatment

__all__ = ['routine_vx', 'screen_treat']


def routine_vx(product='bivalent', genotypes=('hpv16', 'hpv18'), age=9, coverage=0.9, sex='female', start=None, interval=ss.months(6), next_coverage=0.9, name='routine_vx', **kwargs):
    """
    Routine vaccination of girls (or everyone) as they reach an age, with the later doses of a multi-dose vaccine

    Args:
        product (str/ss.Vaccine): a vaccine, or the name of one in HPVsim's table; for several doses, give one with `doses=`, e.g. `hpv.vaccine('bivalent', doses=[0.8, 1.0])`
        genotypes (list): the genotypes in the sim
        age (float): the age at the first dose, in years
        coverage (float): the share given the first dose
        sex (str): 'female', 'male', or None for both
        start (float): the year vaccination starts
        interval (dur): the shortest time between doses
        next_coverage (float): the share of those due a later dose who are given it
        kwargs (dict): passed to the delivery of the first dose

    Returns:
        The delivery of the first dose, or a list of it and the delivery of the later doses (named `<name>_next`)
    """
    vx = vaccine(product, genotypes) if isinstance(product, str) else product
    where = None if sex is None else getattr(ss.ppl, sex)
    first = ss.vaccinate(vx, at_age=age, where=where, coverage=coverage, start=start, name=name, **kwargs)
    if not vx.doses or len(vx.doses) < 2:
        return first
    later = ss.vaccinate(vx, who=vx.ever & (vx.since >= interval), coverage=next_coverage, start=start, name=f'{name}_next')
    return [first, later]


def screen_treat(primary='hpv', triage='via', treat='ablation', genotypes=('hpv16', 'hpv18'), age=(30, 50),
                 interval=ss.years(5), coverage=ss.probperyear(0.2), start=None, triage_coverage=1.0, treat_coverage=1.0):
    """
    A screen-and-treat algorithm: women of screening age screened at most once per interval, those positive
    triaged (optionally), and those positive at triage treated

    WHO's algorithms: `screen_treat('hpv', None, 'ablation')` (screen and treat), `screen_treat('hpv', 'via', 'ablation')`
    (screen, triage, and treat), `screen_treat('via', None, 'ablation')`.

    Args:
        primary (str/ss.Diagnostic): the screening test
        triage (str/ss.Diagnostic): the triage test for those positive at screening, or None to treat them all
        treat (str/ss.Treatment): the treatment
        genotypes (list): the genotypes in the sim
        age (tuple): the ages screened, [lower, upper) in years
        interval (dur): the shortest time between screens
        coverage (float/rate): the share of eligible women screened (a typed probability is per unit time)
        start (float): the year screening starts
        triage_coverage (float): the share of those positive at screening who come for triage
        treat_coverage (float): the share of those to treat who are treated

    Returns:
        A list of deliveries: screening, triage (if any), and treatment, named so
    """
    test = diagnostic(primary, genotypes) if isinstance(primary, str) else primary
    women = ss.ppl.female & (ss.ppl.age >= age[0]) & (ss.ppl.age < age[1])
    screening = ss.screen(test, who=women & (test.since >= interval), coverage=coverage, start=start, name='screening')
    steps = [screening]
    to_treat = screening.positive
    if triage is not None:
        tri = diagnostic(triage, genotypes) if isinstance(triage, str) else triage
        triaging = ss.screen(tri, who=screening.positive, coverage=triage_coverage, name='triage')
        steps.append(triaging)
        to_treat = triaging.positive
    tx = treatment(treat, genotypes) if isinstance(treat, str) else treat
    steps.append(ss.treat(tx, who=to_treat, coverage=treat_coverage, name='treatment'))
    return steps
