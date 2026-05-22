"""Regression test for multi-genotype cancer dedup.

When an agent has CIN with multiple HPV genotypes simultaneously, exactly
ONE genotype must fire cancer for that agent — not one per genotype. v2's
``check_cancer`` enforces this via cross-genotype cancellation; v3's
``HPV.step_state`` must do the equivalent.
"""
import numpy as np
import pytest

import hpvsim as hpv


def _run_multigenotype_sim():
    sim = hpv.Sim(
        location='nigeria',
        start=1990, stop=2050,
        n_agents=2000,
        genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
        rand_seed=0,
    )
    sim.run()
    return sim


def test_no_double_counting_of_multigenotype_cancers():
    """No agent may have cancerous=True on more than one HPV module simultaneously.

    Without cross-genotype cancellation, an agent with CIN in two
    genotypes would transition to cancerous in both modules, resulting
    in multiple ``cancerous`` flags for the same agent. With the fix,
    each agent has at most one cancerous flag across all modules.
    """
    sim = _run_multigenotype_sim()

    genotypes = [m for m in sim.diseases.values() if isinstance(m, hpv.HPV)]
    # Use .raw (all ever-created agent slots) so all modules have the same shape.
    n_slots = genotypes[0].cancerous.raw.shape[0]
    cancerous_count_per_agent = np.zeros(n_slots, dtype=int)
    for mod in genotypes:
        cancerous_count_per_agent += np.asarray(mod.cancerous.raw).astype(int)

    max_concurrent = int(cancerous_count_per_agent.max())
    n_double = int((cancerous_count_per_agent >= 2).sum())
    assert max_concurrent <= 1, (
        f'Agents with cancer from multiple genotypes simultaneously: '
        f'{n_double} agents with >=2 cancerous flags, max={max_concurrent}. '
        f'Cancer must be exclusive across genotypes per agent.'
    )


def test_hpvtotal_new_cancers_equals_per_module_sum():
    """hpvtotal.new_cancers is consistent with per-module new_cancers sums.

    The hpvtotal aggregator accumulates each HPV module's new_cancers at
    each timestep. This test verifies:
      1. The total equals the sum of per-module totals (aggregation
         invariant — holds by construction).
      2. No single module records more new cancer events than the total
         (which would only be possible if the hpvtotal counter were not
         being incremented correctly — catches aggregation regressions).
      3. Each module's new_cancers count is non-negative everywhere
         (no negative correction steps).

    Together with test_no_double_counting_of_multigenotype_cancers, these
    confirm that each cancer event is counted at most once in hpvtotal.
    """
    sim = _run_multigenotype_sim()

    genotypes = [m for m in sim.diseases.values() if isinstance(m, hpv.HPV)]
    per_module_sums = {
        mod.genotype: int(mod.results.new_cancers[:].sum())
        for mod in genotypes
    }
    total_from_modules = sum(per_module_sums.values())
    total_hpvtotal = int(sim.results.hpvtotal.new_cancers[:].sum())

    # 1. Aggregation invariant.
    assert total_from_modules == total_hpvtotal, (
        f'Sum of per-module new_cancers ({total_from_modules}) != '
        f'hpvtotal.new_cancers ({total_hpvtotal}). '
        f'Per-module breakdown: {per_module_sums}.'
    )

    # 2. No individual module exceeds the total.
    for genotype, mod_sum in per_module_sums.items():
        assert mod_sum <= total_hpvtotal, (
            f'{genotype}.new_cancers sum ({mod_sum}) > hpvtotal '
            f'({total_hpvtotal}) — impossible.'
        )

    # 3. No negative counts anywhere.
    for mod in genotypes:
        neg_count = int((np.asarray(mod.results.new_cancers[:]) < 0).sum())
        assert neg_count == 0, (
            f'{mod.genotype}.new_cancers has {neg_count} negative timestep(s).'
        )