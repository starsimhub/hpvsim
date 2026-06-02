"""HPV-specific Starsim interventions.

Currently contains the prophylactic vaccination intervention API:
``hpv.BaseVaccination`` (a subclass of ``ss.BaseVaccination`` that
accepts v2-compatible ``age_range`` / ``sex`` / ``eligibility``
constructor args and composes them into a single Starsim eligibility
callable) and the ``hpv.routine_vx`` / ``hpv.campaign_vx`` leaf classes
that combine it with Starsim's RoutineDelivery / CampaignDelivery.

M06 will add screening (routine_screening / campaign_screening), triage,
treatment (treat_num / treat_delay / radiation), dynamic_pars, and the
txvx family (BaseTxVx / routine_txvx / campaign_txvx / linked_txvx).
"""
import numpy as np
import starsim as ss

from hpvsim.products import vx as _vx

__all__ = ['BaseVaccination', 'routine_vx', 'campaign_vx']


def _cast_sex(sex):
    """Cast v2-style sex input into a set of allowed sex ints (0=F, 1=M).

    Accepts:
      - None: no sex filter (returns None)
      - 'f' / 'm': single sex
      - 0 / 1 (int): single sex by int convention
      - list of 'f'/'m'/0/1: union of sexes

    Anything else raises ValueError.
    """
    if sex is None:
        return None
    if isinstance(sex, str):
        if sex == 'f':
            return {0}
        if sex == 'm':
            return {1}
        raise ValueError(f"sex string must be 'f' or 'm', got {sex!r}")
    if isinstance(sex, (list, tuple, set, np.ndarray)):
        if len(sex) == 0:
            raise ValueError("sex list must not be empty; pass None for 'no sex filter'")
        out = set()
        for s in sex:
            out |= _cast_sex(s)
        return out
    try:
        s_int = int(sex)
    except (TypeError, ValueError):
        raise ValueError(f"sex must be 'f', 'm', 0, 1, or a list thereof, got {sex!r}")
    if s_int not in (0, 1):
        raise ValueError(f"sex int must be 0 or 1, got {sex!r}")
    return {s_int}


def _as_boolarr(extra_result, people):
    """Coerce an eligibility-callback return value into a BoolArr.

    Starsim eligibility callbacks may return either a BoolArr/BoolState or
    an ss.uids. We need a BoolArr so we can intersect with our own conditions
    via ``&``. Clone people.alive (same shape, same linkage) all-False and
    fill True at the returned uids.
    """
    if isinstance(extra_result, ss.BoolArr):
        return extra_result
    # Assume ss.uids or array-like of ints — build a blank BoolArr from alive
    out = people.alive.asnew()
    out.values[:] = False
    out[extra_result] = True
    return out


def _compose_eligibility(age_range, sex, extra):
    """Compose v2-style targeting into a Starsim eligibility callable.

    Returns ``elig(sim) -> ss.uids`` that intersects:
      - sim.people.alive
      - sim.people.age in [age_range[0], age_range[1]) if age_range is set
      - sim.people.sex matches sex if sex is set to a single sex
      - extra(sim) if extra is provided (callable returning BoolArr or uids)
    """
    sex_set = _cast_sex(sex)

    def elig(sim):
        cond = sim.people.alive
        if age_range is not None:
            lo, hi = age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age < hi)
        if sex_set is not None and len(sex_set) == 1:
            (s,) = sex_set
            # Starsim uses people.female (0) / people.male (1) BoolArrs
            # rather than a numeric sex array; map int -> BoolArr
            if s == 0:
                cond = cond & sim.people.female
            else:
                cond = cond & sim.people.male
        if extra is not None:
            cond = cond & _as_boolarr(extra(sim), sim.people)
        return cond.uids

    return elig


class BaseVaccination(ss.BaseVaccination):
    """HPV-specific prophylactic vaccination base.

    Wraps Starsim's ``ss.BaseVaccination`` to add v2-compatible
    ``age_range`` / ``sex`` / ``eligibility`` constructor args. These
    compose into a single Starsim eligibility callable. The originals are
    stored on the instance for introspection (e.g. AgeResults consumption).

    Also overrides ``_parse_product_str`` so that
    ``routine_vx(product='bivalent', ...)`` resolves through
    ``hpv.vx(name='bivalent')``, mirroring v2's string-product convention.
    """

    def __init__(self, *args, age_range=None, sex=None, eligibility=None,
                 **kwargs):
        composed = _compose_eligibility(age_range, sex, eligibility)
        super().__init__(*args, eligibility=composed, **kwargs)
        # Raw constructor args, preserved for introspection (e.g.
        # M04 AgeResults stratification by vaccination cohort).
        self.age_range = age_range
        self.sex_raw = sex
        self.sex = _cast_sex(sex)
        self.eligibility_raw = eligibility

    def _parse_product_str(self, product):
        """Resolve a string product name through hpv.vx default lookup."""
        return _vx(name=product)


class routine_vx(BaseVaccination, ss.RoutineDelivery):
    """Routine prophylactic HPV vaccination."""
    pass


class campaign_vx(BaseVaccination, ss.CampaignDelivery):
    """Campaign-style prophylactic HPV vaccination."""
    pass