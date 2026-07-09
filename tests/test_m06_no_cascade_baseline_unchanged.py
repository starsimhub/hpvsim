"""Pinned-scalar guard: a no-cascade run must reproduce the M06-baseline
total infections and cancers. Catches accidental RNG-stream perturbations
from new ss.Dist instances introduced by M06 (cascade coverage_dist,
result_dist, efficacy_dist, _sterilizing_dist per txvx, _dur_dist per
radiation, etc.).

Mirrors M05's test_no_vx_baseline_unchanged pattern.
"""
import hpvsim as hpv

# Pinned at M06 baseline-cut on branch m06-test-and-treat-cascade.
# If you change any cascade-related RNG creation, this WILL fail and
# you must investigate (intentional change -> re-pin; accidental
# change -> fix the leaking dist).
# Updated 2026-07-08 (10191.0 -> 10855.0, 13.0 -> 18.0): starsim 3.5.0 upgrade
# reshuffled distribution RNG streams (see hpvsim/parameters.py ss.beta_dist
# fix). Model behaviour is unchanged — grow acceptance gates + full suite pass
# on 3.5.0; this is a single-seed CRN re-pin, not a model change.
PINNED_TOTAL_INFECTIONS = 10855.0
PINNED_TOTAL_CANCERS    = 18.0


def _build_sim():
    return hpv.Sim(
        n_agents=2000, start=2000, stop=2020, rand_seed=0,
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        location='nigeria',
        v2_compat_demographics=True,
    )


def test_no_cascade_baseline_unchanged():
    sim = _build_sim()
    sim.run()
    total_inf = float(sum(
        sim.diseases[g].results['cum_infections'][-1]
        for g in ('hpv16', 'hpv18', 'hi5', 'ohr')
    ))
    total_cancers = float(sum(
        sim.diseases[g].results['cum_cancers'][-1]
        for g in ('hpv16', 'hpv18', 'hi5', 'ohr')
    ))
    assert total_inf == PINNED_TOTAL_INFECTIONS, (
        f'cum_infections drift: {total_inf} != {PINNED_TOTAL_INFECTIONS}. '
        'Check that no new ss.Dist instance is sharing an RNG stream with '
        'HPV transmission decisions.'
    )
    assert total_cancers == PINNED_TOTAL_CANCERS, (
        f'cum_cancers drift: {total_cancers} != {PINNED_TOTAL_CANCERS}.'
    )
