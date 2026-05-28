"""HPV-specific Starsim interventions.

Contains the prophylactic vaccination intervention API:
``hpv.BaseVaccination`` (a subclass of ``ss.BaseVaccination`` that
accepts v2-compatible ``age_range`` / ``sex`` / ``eligibility``
constructor args and composes them into a single Starsim eligibility
callable) and the ``hpv.routine_vx`` / ``hpv.campaign_vx`` leaf classes
that combine it with Starsim's RoutineDelivery / CampaignDelivery.

M06 adds screening (routine_screening / campaign_screening), triage
(routine_triage / campaign_triage), treatment, dynamic_pars, and the
txvx family (BaseTxVx / routine_txvx / campaign_txvx / linked_txvx).
"""
from collections import defaultdict

import numpy as np
import starsim as ss

from hpvsim.products import vx as _vx

__all__ = [
    'BaseVaccination', 'routine_vx', 'campaign_vx',
    # M06
    'BaseTest', 'BaseScreening', 'BaseTriage',
    'routine_screening', 'campaign_screening',
    'routine_triage', 'campaign_triage',
    'BaseTreatment', 'treat_num', 'treat_delay',
    'BaseTxVx', 'routine_txvx', 'campaign_txvx', 'linked_txvx',
    'dynamic_pars',
]


def _coerce_sex(sex):
    """Coerce v2-style sex input into a set of allowed sex ints (0=F, 1=M).

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
            out |= _coerce_sex(s)
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
    out.raw[:] = False
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
    sex_set = _coerce_sex(sex)

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


def _compose_screening_eligibility(age_range, sex, extra, debut_age):
    """Compose v2-style screening eligibility into a Starsim callable.

    Extends ``_compose_eligibility`` with an optional ``debut_age`` lower-
    bound on ``sim.people.age``. When ``debut_age`` is None, semantics are
    identical to ``_compose_eligibility``.

    Returns ``elig(sim) -> ss.uids`` intersecting:
      - sim.people.alive
      - sim.people.female / male (per _coerce_sex(sex))
      - sim.people.age in [age_range[0], age_range[1]) if set
      - sim.people.age >= debut_age if set
      - extra(sim) if provided
    """
    sex_set = _coerce_sex(sex)

    def elig(sim):
        cond = sim.people.alive
        if sex_set is not None and len(sex_set) == 1:
            (s,) = sex_set
            cond = cond & (sim.people.female if s == 0 else sim.people.male)
        if age_range is not None:
            lo, hi = age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age < hi)
        if debut_age is not None:
            cond = cond & (sim.people.age >= debut_age)
        if extra is not None:
            cond = cond & _as_boolarr(extra(sim), sim.people)
        return cond.uids

    return elig


def _any_genotype_cancer(sim):
    """Return a BoolArr OR-ing module.cancerous across all HPV modules.

    Used by hpv.BaseTreatment.check_eligibility to gate on cancer status:
    treat_cancer=True interventions require this BoolArr be True; the
    inverse (~_any_genotype_cancer(sim)) gates non-cancer treatments.
    """
    # Late import to avoid the interventions <-> products circular import
    from hpvsim.products import _iter_hpv_modules
    out = sim.people.alive.asnew()
    out.raw[:] = False
    for module in _iter_hpv_modules(sim):
        out[module.cancerous.uids] = True
    return out


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
        self.sex = _coerce_sex(sex)
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


class BaseTest(ss.BaseTest):
    """HPV-specific test/screening base.

    Adds v2-compatible age_range / sex / eligibility / debut_age kwargs,
    composed into a single Starsim eligibility callable via
    _compose_screening_eligibility. Overrides _parse_product_str so
    routine_screening(product='via', ...) resolves through hpv.dx(name='via').
    """

    def __init__(self, *args, age_range=None, sex='f', eligibility=None,
                 debut_age=None, **kwargs):
        composed = _compose_screening_eligibility(age_range, sex, eligibility, debut_age)
        super().__init__(*args, eligibility=composed, **kwargs)
        self.age_range = age_range
        self.sex_raw = sex
        self.sex = _coerce_sex(sex)
        self.eligibility_raw = eligibility
        self.debut_age = debut_age

    def _parse_product_str(self, product):
        from hpvsim.products import dx as _dx
        return _dx(name=product)


class BaseScreening(BaseTest, ss.BaseScreening):
    """HPV-specific BaseScreening — composes HPV eligibility with Starsim's screening step."""
    pass


class BaseTriage(BaseTest, ss.BaseTriage):
    """HPV-specific BaseTriage.

    Overrides ss.BaseTriage.step to use integer-ti membership (the same
    pattern ss.BaseScreening uses correctly). Upstream Starsim 3.3.4 has
    a TODO on its triage step using `sim.t in timepoints` which silently
    fails under quarterly dt because sim.t is a freq object, not an int.

    Also mirrors ss.BaseScreening's per-step bookkeeping: records screened,
    screens, and ti_screened so that downstream eligibility callbacks (and
    test assertions) can inspect who was triaged.
    """

    def step(self):
        self.outcomes = {k: ss.uids() for k in self.product.hierarchy}
        accept_uids = ss.uids()
        if self.sim.ti in self.timepoints:
            accept_uids = self.deliver()
            if len(accept_uids):
                self.screened[accept_uids] = True
                self.screens[accept_uids] += 1
                self.ti_screened[accept_uids] = self.sim.ti
        return accept_uids


class routine_screening(BaseScreening, ss.RoutineDelivery):
    """Routine HPV screening."""
    pass


class campaign_screening(BaseScreening, ss.CampaignDelivery):
    """Campaign HPV screening."""
    pass


class routine_triage(BaseTriage, ss.RoutineDelivery):
    """Routine HPV triage."""
    pass


class campaign_triage(BaseTriage, ss.CampaignDelivery):
    """Campaign HPV triage."""
    pass


class BaseTreatment(ss.BaseTreatment):
    """HPV-specific treatment base.

    Adds:
    - v2-compatible age_range / sex / eligibility kwargs
    - HPV-specific eligibility: female + alive + cancer-status-matched
      (cancer treatments require any-genotype cancerous; non-cancer
      treatments require no cancerous on any genotype)
    - Per-intervention state: cin_treated / cin_treatments / ti_cin_treated
      for CIN treatments, cancer_treated / etc. for cancer treatments

    The `treat_cancer` flag is derived at __init__ time from whether the
    product is an `hpv.radiation` instance.
    """

    def __init__(self, *args, age_range=None, sex='f', eligibility=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.age_range = age_range
        # Late import to avoid circular
        from hpvsim.products import radiation as _radiation
        self.treat_cancer = isinstance(self.product, _radiation)
        # Cancer treatments are not sex-restricted by default (radiation is
        # post-triage; both sexes can have cervical/other HPV-related cancers
        # in multi-site models). CIN treatments default to female only.
        if self.treat_cancer and sex == 'f':
            sex = None
        self.sex_raw = sex
        self.sex = _coerce_sex(sex)
        self.eligibility_user = eligibility
        self.define_states(
            ss.BoolArr('cin_treated'),
            ss.FloatArr('cin_treatments', default=0),
            ss.FloatArr('ti_cin_treated'),
            ss.BoolArr('cancer_treated'),
            ss.FloatArr('cancer_treatments', default=0),
            ss.FloatArr('ti_cancer_treated'),
        )

    def _parse_product_str(self, product):
        from hpvsim.products import tx as _tx
        return _tx(name=product)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_cin_treated',    dtype=int, scale=True, label='Number first-CIN-treated'),
            ss.Result('new_cancer_treated', dtype=int, scale=True, label='Number first-cancer-treated'),
        )

    def check_eligibility(self):
        sim = self.sim
        cond = sim.people.alive
        if self.sex is not None:
            if 0 in self.sex:
                cond = cond & sim.people.female
            elif 1 in self.sex:
                cond = cond & sim.people.male
        if self.age_range is not None:
            lo, hi = self.age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age <= hi)
        any_cancer = _any_genotype_cancer(sim)
        cond = cond & (any_cancer if self.treat_cancer else ~any_cancer)
        if self.eligibility_user is not None:
            cond = cond & _as_boolarr(self.eligibility_user(sim), sim.people)
        return cond.uids


class treat_num(BaseTreatment, ss.treat_num):
    """Treat a fixed number of HPV+CIN+ agents each step (or all eligible if max_capacity=None)."""

    def step(self):
        treat_uids = super().step()
        if len(treat_uids):
            if self.treat_cancer:
                new = treat_uids[~self.cancer_treated[treat_uids]]
                self.cancer_treated[treat_uids] = True
                self.cancer_treatments[treat_uids] += 1
                self.ti_cancer_treated[treat_uids] = self.sim.ti
                self.results['new_cancer_treated'][self.sim.ti] += len(new)
            else:
                new = treat_uids[~self.cin_treated[treat_uids]]
                self.cin_treated[treat_uids] = True
                self.cin_treatments[treat_uids] += 1
                self.ti_cin_treated[treat_uids] = self.sim.ti
                self.results['new_cin_treated'][self.sim.ti] += len(new)
        return treat_uids


class treat_delay(BaseTreatment):
    """Treat HPV+CIN+ agents after a fixed delay.

    On each step:
      1. Newly-eligible accepters are enqueued at `due_ti = sim.ti +
         round(delay / dt)`.
      2. Agents whose due_ti is the current ti are treated.

    delay is in years. Integer-ti scheduler keys are the M05-lesson
    upgrade over v2's float subtraction (sim.t - delay/dt).

    v2 reference: hpvsim/_v2_legacy/interventions.py:1098-1134
    """

    def __init__(self, delay=None, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay or 0
        self.scheduler = defaultdict(list)

    def add_to_schedule(self):
        accept = self.get_accept_inds()
        if len(accept):
            due_ti = self.sim.ti + int(round(self.delay / self.sim.t.dt_year))
            self.scheduler[due_ti].extend(int(u) for u in accept)

    def get_candidates(self):
        return ss.uids(self.scheduler.pop(self.sim.ti, []))

    def step(self):
        self.add_to_schedule()
        treat_uids = super().step()
        # Mirror treat_num's per-intervention bookkeeping (BaseTreatment.step
        # is the upstream ss.BaseTreatment.step which only calls product.administer)
        if len(treat_uids):
            if self.treat_cancer:
                new = treat_uids[~self.cancer_treated[treat_uids]]
                self.cancer_treated[treat_uids] = True
                self.cancer_treatments[treat_uids] += 1
                self.ti_cancer_treated[treat_uids] = self.sim.ti
                self.results['new_cancer_treated'][self.sim.ti] += len(new)
            else:
                new = treat_uids[~self.cin_treated[treat_uids]]
                self.cin_treated[treat_uids] = True
                self.cin_treatments[treat_uids] += 1
                self.ti_cin_treated[treat_uids] = self.sim.ti
                self.results['new_cin_treated'][self.sim.ti] += len(new)
        return treat_uids


class BaseTxVx(BaseTreatment):
    """HPV therapeutic vaccination base.

    Extends BaseTreatment with txvx-specific per-intervention state:
    tx_vaccinated / txvx_doses / ti_tx_vaccinated.

    On each delivery the agent's txvx_imm is bumped per genotype (via
    hpv.txvx.administer). The intervention's own dose counters track
    program-level uptake.

    v2 reference: hpvsim/_v2_legacy/interventions.py:1137-1252
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_states(
            ss.BoolArr('tx_vaccinated'),
            ss.FloatArr('txvx_doses', default=0),
            ss.FloatArr('ti_tx_vaccinated'),
        )

    def _parse_product_str(self, product):
        from hpvsim.products import txvx as _txvx
        return _txvx(name=product)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_tx_vaccinated', dtype=int, scale=True, label='Number first-txvx-vaccinated'),
            ss.Result('new_txvx_doses',    dtype=int, scale=True, label='Number txvx doses administered'),
        )

    def check_eligibility(self):
        """TxVx eligibility — female + alive + cancer-free + age range.

        Unlike treat_num/treat_delay, BaseTxVx never targets cancer patients
        (radiation is for that path). treat_cancer is forced False here.
        """
        sim = self.sim
        cond = sim.people.alive & sim.people.female
        if self.age_range is not None:
            lo, hi = self.age_range
            cond = cond & (sim.people.age >= lo) & (sim.people.age <= hi)
        # Never treat cancer agents
        any_cancer = _any_genotype_cancer(sim)
        cond = cond & ~any_cancer
        if self.eligibility_user is not None:
            cond = cond & _as_boolarr(self.eligibility_user(sim), sim.people)
        return cond.uids

    def deliver(self):
        """One-step delivery — finds accepters, administers, bumps counters."""
        accept_uids = self.get_accept_inds()
        if len(accept_uids):
            self.product.administer(self.sim.people, accept_uids)
            new = accept_uids[~self.tx_vaccinated[accept_uids]]
            self.tx_vaccinated[accept_uids] = True
            self.txvx_doses[accept_uids] += 1
            self.ti_tx_vaccinated[accept_uids] = self.sim.ti
            self.results['new_tx_vaccinated'][self.sim.ti] += len(new)
            self.results['new_txvx_doses'][self.sim.ti] += len(accept_uids)
        return accept_uids

    def step(self):
        # Default: scheduled delivery via RoutineDelivery/CampaignDelivery's timepoints
        if self.sim.ti in self.timepoints:
            return self.deliver()
        return ss.uids()


class routine_txvx(BaseTxVx, ss.RoutineDelivery):
    """Routine therapeutic vaccination."""
    pass


class campaign_txvx(BaseTxVx, ss.CampaignDelivery):
    """Campaign therapeutic vaccination."""
    pass


class linked_txvx(BaseTxVx):
    """Therapeutic vaccination linked to another intervention's outcomes.

    Has no own timeline. Fires every step; eligibility= callback (required)
    determines who actually receives the dose. Typical usage:

        linked = hpv.linked_txvx(
            product='txvx1', prob=0.6,
            eligibility=lambda s: s.interventions['colpo'].outcomes['lsil'],
        )
    """

    def __init__(self, *args, eligibility=None, **kwargs):
        if eligibility is None:
            raise ValueError(
                "linked_txvx requires eligibility= "
                "(typically a screen.outcomes['positive'] callback)"
            )
        super().__init__(*args, eligibility=eligibility, **kwargs)
        self.timepoints = None  # No own schedule

    def step(self):
        return self.deliver()


def _set_dotted(sim, dotted_path, value):
    """Resolve a dotted-path string against (sim.diseases, sim.interventions, sim.pars) and set it.

    Top-level segment is looked up in:
      1. sim.diseases (by key)         e.g. 'hpv16.beta' -> sim.diseases['hpv16'].pars.beta
      2. sim.interventions (by name)   e.g. 'screen.prob' -> sim.interventions['screen'].prob
      3. sim.pars (by key)             e.g. 'rand_seed' -> sim.pars.rand_seed
    Diseases and interventions require at least one tail segment (the attribute
    to set on the resolved module). sim.pars accepts a single segment (set
    directly on sim.pars). Raises KeyError if the head doesn't resolve anywhere.
    """
    parts = dotted_path.split('.')
    head, tail = parts[0], parts[1:]

    if head in sim.diseases:
        if not tail:
            raise KeyError(
                f'Path {dotted_path!r}: missing attribute name after disease '
                f'key {head!r}.'
            )
        # Navigate into the disease module's .pars by default.
        target = sim.diseases[head].pars
        for seg in tail[:-1]:
            target = getattr(target, seg)
        setattr(target, tail[-1], value)
        return

    if head in sim.interventions:
        if not tail:
            raise KeyError(
                f'Path {dotted_path!r}: missing attribute name after '
                f'intervention name {head!r}.'
            )
        target = sim.interventions[head]
        for seg in tail[:-1]:
            target = getattr(target, seg)
        setattr(target, tail[-1], value)
        return

    # Fall back to sim.pars. Single segment: set directly. Multi-segment:
    # walk through head + tail[:-1] and set the last segment.
    if not hasattr(sim.pars, head):
        raise KeyError(
            f'Cannot resolve dotted path {dotted_path!r}: head segment '
            f'{head!r} is not a sim.diseases / sim.interventions / sim.pars key.'
        )
    if not tail:
        setattr(sim.pars, head, value)
        return
    target = getattr(sim.pars, head)
    for seg in tail[:-1]:
        target = getattr(target, seg)
    setattr(target, tail[-1], value)


class dynamic_pars(ss.Intervention):
    """Time-varying parameter editor.

    pars: dict mapping dotted-path strings to {'years': [...], 'vals': [...]}
    schedules. Each step, the resolved parameter is set to the interpolated
    (default) or stepwise (interpolate=False) value for the current year.

    Dotted-path resolution order: sim.diseases > sim.interventions > sim.pars.

    v2 reference: hpvsim/_v2_legacy/interventions.py:406-489 (uses timestep
    keys; v3 uses epoch-year keys for ergonomic schedule authoring).
    """

    def __init__(self, pars=None, interpolate=True, **kwargs):
        super().__init__(**kwargs)
        self.par_schedules = pars or {}
        self.interpolate = interpolate

    def step(self):
        year = self.sim.t.now('year')
        for dotted_path, schedule in self.par_schedules.items():
            years = np.asarray(schedule['years'], dtype=float)
            vals = np.asarray(schedule['vals'], dtype=float)
            if self.interpolate:
                val = float(np.interp(year, years, vals))
            else:
                idx = int(np.searchsorted(years, year, side='right')) - 1
                if idx < 0:
                    continue
                val = float(vals[idx])
            _set_dotted(self.sim, dotted_path, val)
