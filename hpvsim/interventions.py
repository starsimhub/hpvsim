"""
HPV interventions with HPVsim's names, as Starsim 4 deliveries

Each function returns an ``ss.vaccinate``, ``ss.screen``, or ``ss.treat``, so anything those take can
be given too (``capacity=``, ``queue=``, ``repeat=``, ``priority=``, ``on_result=``, ...).

Algorithms are trees of follow-ups: a triage or treatment is given as ``on_positive=`` (or
``on_result=``) of the screen it follows, instead of an ``eligibility=`` function reading the
screen's ``outcomes``::

    ablation = hpv.treat_num(product='ablation', prob=0.9, max_capacity=100)
    via = hpv.routine_triage(product='via', prob=0.9, on_positive=ablation)
    screen = hpv.routine_screening(product='hpv', prob=0.2, start_year=2025, age_range=(30, 50), on_positive=via)

Routine delivery at an annual probability (``prob``) is a rate among those eligible at every step
(``coverage=ss.probperyear(prob)``, ``repeat=True``), as in HPVsim 3; a campaign (``years=``) offers
the product once to everyone eligible in each of those years.
"""
import numpy as np
import sciris as sc
import starsim as ss

from . import products as hpp

__all__ = [
    'routine_vx', 'campaign_vx',
    'routine_screening', 'campaign_screening',
    'routine_triage', 'campaign_triage',
    'treat_num', 'treat_delay',
    'routine_txvx', 'campaign_txvx', 'linked_txvx',
    'dynamic_pars', 'any_cancer',
]

ppl = ss.people


def _cast_sex(sex):
    """ 'f', 'm', 'fm', 0, 1, or a list of them, as a set of 0 (female) and 1 (male); None for both """
    if sex is None:
        return None
    if isinstance(sex, str):
        out = set()
        for char in sex:
            if char not in 'fm':
                raise ValueError(f"sex must contain only 'f' and 'm', not {sex!r}")
            out.add(0 if char == 'f' else 1)
        return out or None
    if isinstance(sex, (list, tuple, set, np.ndarray)):
        out = set()
        for s in sex:
            out |= _cast_sex(s)
        return out
    if int(sex) not in (0, 1):
        raise ValueError(f'sex must be 0 or 1, not {sex!r}')
    return {int(sex)}


def any_cancer(genotypes=None):
    """
    Having cancer of any genotype, as a condition: of the genotypes given, or (as a function of the sim, so agents only) of all in the sim

    Args:
        genotypes (list): the genotype diseases, by name
    """
    if genotypes is not None:
        return ss.any(ss.state(f'{g}.X') for g in genotypes)
    return _any_cancer


def _no_cancer(sim):
    """ Who is alive with no cancer of any genotype """
    auids = sim.people.auids
    return auids[~np.isin(auids, _any_cancer(sim))]


def _any_cancer(sim):
    """ Who has cancer of any genotype in the sim """
    from .hpv import HPV
    return ss.uids.concatenate([d.X.uids for d in sim.diseases.values() if isinstance(d, HPV)]).unique()


def _eligible(age_range=None, sex=None, eligibility=None, debut_age=None, extra=None):
    """ Who is eligible, as a condition: an age range [lo, hi), a sex, a minimum age, and any further condition (or a function of the sim) """
    conds = []
    if age_range is not None:
        lo, hi = age_range
        conds += [ppl.age >= lo, ppl.age < hi]
    if debut_age is not None:
        conds.append(ppl.age >= debut_age)
    sexes = _cast_sex(sex)
    if sexes is not None and len(sexes) == 1:
        conds.append(ppl.female if 0 in sexes else ~ppl.female)
    if extra is not None:
        conds.append(extra)
    if eligibility is not None: # A condition, or a function of the sim (which makes the model agent-based)
        conds.append(eligibility)
    if not conds:
        return None
    out = conds[0]
    for c in conds[1:]:
        out = out & c if isinstance(out, ss.Cond) or isinstance(c, ss.Cond) else ss.conditions.as_cond(out) & c
    return out


def _timing(start_year=None, end_year=None, years=None):
    return dict(start=start_year, stop=end_year, years=None if years is None else sc.tolist(years))


def _rate(prob):
    if isinstance(prob, (list, tuple, np.ndarray)) or isinstance(prob, dict):
        raise NotImplementedError('A probability that varies by year is not supported yet as a rate (TODO(v4-rs): ss.probperyear(ss.timeseries(...)))')
    return ss.probperyear(prob)


def _product(product, cls):
    return cls(name=product) if isinstance(product, str) else product


def routine_vx(product=None, prob=None, age_range=None, sex='f', eligibility=None, start_year=None, end_year=None, years=None, name=None, **kwargs):
    """
    Routine prophylactic vaccination: an annual probability among those eligible, every step

    Args:
        product (str/hpv.vx): 'bivalent', 'quadrivalent', 'nonavalent', or a product
        prob (float): annual probability of vaccination among those eligible
        age_range (tuple): ages [lo, hi)
        sex (str): 'f' (default), 'm', or 'fm'
        eligibility (Cond/func): who else must hold; a function of the sim makes it agents only
        start_year, end_year (float): when it runs
        kwargs (dict): passed to `ss.vaccinate` (e.g. `dose=`)

    Examples:
        ```python
        vx = hpv.routine_vx(product='bivalent', prob=0.9, age_range=(9, 10), start_year=2020)
        ```
    """
    return ss.vaccinate(_product(product, hpp.vx), coverage=_rate(prob), eligible=_eligible(age_range, sex, eligibility),
                        **_timing(start_year, end_year), name=name, **kwargs)


def campaign_vx(product=None, prob=None, years=None, age_range=None, sex='f', eligibility=None, name=None, **kwargs):
    """ A vaccination campaign: in each of `years`, everyone eligible is offered it, and a share `prob` are vaccinated """
    return ss.vaccinate(_product(product, hpp.vx), coverage=prob, eligible=_eligible(age_range, sex, eligibility), years=sc.tolist(years), name=name, **kwargs)


def routine_screening(product=None, prob=None, age_range=None, sex='f', eligibility=None, debut_age=None, start_year=None, end_year=None, years=None, name=None, **kwargs):
    """
    Routine screening: an annual probability among those eligible, every step

    Args:
        product (str/hpv.dx): e.g. 'hpv', 'via', 'lbc', or a product
        prob (float): annual probability of being screened among those eligible
        age_range (tuple): ages [lo, hi)
        sex (str): 'f' (default), 'm', or 'fm'
        eligibility (Cond/func): who else must hold
        debut_age (float): a minimum age
        kwargs (dict): passed to `ss.screen`: `on_positive=`, `on_negative=`, `on_result=`, `repeat=` (e.g. `ss.years(5)`), ...

    Examples:
        ```python
        screen = hpv.routine_screening(product='hpv', prob=0.2, age_range=(30, 50), start_year=2025, repeat=ss.years(5),
                                       on_positive=hpv.treat_num(product='ablation', prob=0.9))
        ```
    """
    kwargs.setdefault('repeat', True)
    return ss.screen(_product(product, hpp.dx), coverage=_rate(prob), eligible=_eligible(age_range, sex, eligibility, debut_age),
                     **_timing(start_year, end_year), name=name, **kwargs)


def campaign_screening(product=None, prob=None, years=None, age_range=None, sex='f', eligibility=None, debut_age=None, name=None, **kwargs):
    """ A screening campaign: in each of `years`, a share `prob` of those eligible are screened """
    return ss.screen(_product(product, hpp.dx), coverage=prob, eligible=_eligible(age_range, sex, eligibility, debut_age), years=sc.tolist(years), name=name, **kwargs)


def routine_triage(product=None, prob=None, age_range=None, sex=None, eligibility=None, name=None, **kwargs):
    """
    A triage test, given as a follow-up (`on_positive=`, `on_result=`) of a screen: a share `prob` of those sent to it come

    Args:
        product (str/hpv.dx): e.g. 'via', 'colposcopy', 'tx_assigner'
        prob (float): the share of those sent who come
        kwargs (dict): passed to `ss.screen`: its own `on_positive=`, `on_result=`, ...
    """
    return ss.screen(_product(product, hpp.dx), coverage=prob, eligible=_eligible(age_range, sex, eligibility), name=name, **kwargs)


campaign_triage = routine_triage


def _tx_eligible(product, age_range, sex, eligibility, genotypes=None):
    """ Treatments for lesions are for people with no cancer of any genotype; radiation is for people with cancer """
    cancer = any_cancer(genotypes)
    if isinstance(product, hpp.radiation):
        extra = cancer
    else:
        extra = ~cancer if isinstance(cancer, ss.Cond) else _no_cancer
    return _eligible(age_range, sex, eligibility, extra=extra)


def treat_num(product=None, prob=None, max_capacity=None, age_range=None, sex='f', eligibility=None, genotypes=None, name=None, **kwargs):
    """
    Treatment, usually as a follow-up of a test: a share `prob` of those sent are treated, up to `max_capacity` a step

    Those over capacity wait (first come, first served) and leave the queue if they are no longer eligible,
    as in HPVsim 3; give `queue=ss.years(1)` for the longest wait, or `queue=False` to drop them.

    Args:
        product (str/hpv.tx/hpv.radiation): 'ablation', 'excision', 'radiation', or a product
        prob (float): the share treated
        max_capacity (float): the most treated in a step (None for no limit)
        genotypes (list): the genotypes (for who has cancer; default all four)
        kwargs (dict): passed to `ss.treat`
    """
    product = hpp.radiation() if product == 'radiation' else _product(product, hpp.tx)
    if max_capacity is not None:
        kwargs.setdefault('queue', True)
    return ss.treat(product, coverage=prob, capacity=max_capacity, eligible=_tx_eligible(product, age_range, sex, eligibility, genotypes), name=name, **kwargs)


def treat_delay(*args, **kwargs):
    """ Not ported: give the test the delay (`ss.Test(delay=...)`), or queue the treatment (`queue=`) """
    raise NotImplementedError('hpv.treat_delay is not ported to Starsim 4: give the test before it a delay (hpv.dx(..., delay=ss.months(6))) or use treat_num(queue=...)')


def routine_txvx(product='txvx1', prob=None, age_range=None, sex='f', eligibility=None, start_year=None, end_year=None, genotypes=None, name=None, **kwargs):
    """ Routine therapeutic vaccination: an annual probability among those eligible (with no cancer), every step """
    product = _product(product, hpp.txvx)
    kwargs.setdefault('repeat', True)
    return ss.treat(product, coverage=_rate(prob), eligible=_tx_eligible(product, age_range, sex, eligibility, genotypes), **_timing(start_year, end_year), name=name, **kwargs)


def campaign_txvx(product='txvx1', prob=None, years=None, age_range=None, sex='f', eligibility=None, genotypes=None, name=None, **kwargs):
    """ A therapeutic vaccination campaign """
    product = _product(product, hpp.txvx)
    return ss.treat(product, coverage=prob, eligible=_tx_eligible(product, age_range, sex, eligibility, genotypes), years=sc.tolist(years), name=name, **kwargs)


def linked_txvx(product='txvx1', prob=None, age_range=None, sex='f', eligibility=None, genotypes=None, name=None, **kwargs):
    """ Therapeutic vaccination as a follow-up (`on_positive=`, `on_result=`) of a test """
    product = _product(product, hpp.txvx)
    return ss.treat(product, coverage=prob, eligible=_tx_eligible(product, age_range, sex, eligibility, genotypes), name=name, **kwargs)


def dynamic_pars(*args, **kwargs):
    """ Not ported: a parameter that varies in time is an `ss.timeseries`, or an `ss.modify(..., when=)` """
    raise NotImplementedError('hpv.dynamic_pars is not ported to Starsim 4: give the parameter as an ss.timeseries, or use ss.modify(target, multiply=ss.timeseries(...))')
