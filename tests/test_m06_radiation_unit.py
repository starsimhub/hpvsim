"""Unit tests for hpv.radiation — cancer treatment product."""
import numpy as np
import starsim as ss
import hpvsim as hpv
from hpvsim.products import radiation as hpv_radiation


def _four_genotype_sim():
    return hpv.Sim(
        n_agents=200, start=2020, stop=2021, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
    )


def _attach_and_init(sim, p_instance):
    """Attach product to stub treat_num and init; return live post-init copy."""
    sim.pars['interventions'] = [ss.treat_num(product=p_instance, prob=0.0)]
    sim.init()
    return sim.interventions[0].product


def test_radiation_extends_ti_dead_cancer_on_cancerous_agents():
    sim = _four_genotype_sim()
    r = _attach_and_init(sim, hpv_radiation())
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cancerous[uids] = True
    initial = 100.0
    sim.diseases['hpv16'].ti_dead_cancer[uids] = initial
    r.administer(uids)
    extension = sim.diseases['hpv16'].ti_dead_cancer[uids] - initial
    # Default duration: normal(mean=1.5 years, sd=0.167 years); at dt=0.25
    # the per-agent extension is ceil(years_drawn / dt_year) integer steps.
    # 99.99% of draws fall in (~1.0, ~2.0) yr → ceil/0.25 lies in [4, 8].
    # Guard against the dt-units bug (using freq-object dt would give ~2 steps,
    # i.e. extension ~2 instead of ~6 at the default config).
    assert np.all(extension >= 3), (
        f'radiation extension too small: {extension!r} — '
        'check dt_year vs dt freq-object arithmetic'
    )


def test_radiation_skips_non_cancer_agents():
    sim = _four_genotype_sim()
    r = _attach_and_init(sim, hpv_radiation())
    uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cancerous[uids] = False  # not cancer
    sim.diseases['hpv16'].ti_dead_cancer[uids] = np.nan
    r.administer(uids)
    assert np.all(np.isnan(sim.diseases['hpv16'].ti_dead_cancer[uids]))


def test_radiation_empty_uids_noop():
    sim = _four_genotype_sim()
    r = _attach_and_init(sim, hpv_radiation())
    out = r.administer(ss.uids())
    assert len(out) == 0


def test_radiation_default_duration_v2_match():
    """Default duration is normal(18 months, 2 months) converted to years."""
    r = hpv_radiation()
    assert r.pars.dur['par1'] == 18 / 12  # mean: 1.5 years
    assert r.pars.dur['par2'] == 2 / 12   # sd: ~0.167 years


def test_radiation_extension_is_positive_for_non_sorted_input():
    """Regression guard: radiation must extend ti_dead_cancer for every
    cancerous agent regardless of input uids ordering.

    Earlier implementation pre-drew durations indexed by uids-order, then
    wrote them via `cancer_uids` (sorted by intersect()). When the two
    orderings disagreed (non-sorted input), each agent received the wrong
    agent's duration. With non-sorted input, some agents could even pick
    up zero or near-zero durations from the wrong slot.
    """
    sim = _four_genotype_sim()
    r = _attach_and_init(sim, hpv_radiation())
    # Pick three cancer agents and pass them in REVERSE order
    cancer_uids = sim.people.alive.uids[:3]
    sim.diseases['hpv16'].cancerous[cancer_uids] = True
    initial = 100.0
    sim.diseases['hpv16'].ti_dead_cancer[cancer_uids] = initial
    reversed_input = ss.uids(cancer_uids[::-1])
    r.administer(reversed_input)
    # All three must have ti_dead_cancer strictly greater than initial
    # (radiation duration sample is always positive; ceil makes it >= 1 step).
    assert np.all(sim.diseases['hpv16'].ti_dead_cancer[cancer_uids] > initial)
