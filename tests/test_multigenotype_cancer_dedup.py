"""Multi-genotype cancer dedup.

When an agent has CIN with several HPV genotypes at once, exactly one genotype
fires cancer for that agent — not one per genotype. ``HPV.step_state`` enforces
this via cross-genotype cancellation.
"""
import numpy as np
import pytest

import hpvsim as hpv


@pytest.fixture(scope='module')
def multigenotype_sim():
    """One 4-genotype run shared (read-only) by both dedup tests below.

    Both tests are about how cancers are attributed, so the run has to actually
    produce some: 6000 agents at dt=0.5 over 2000-2050 realizes ~30-40 cancers
    across the four genotypes, and both tests assert that total is non-zero so
    a future shrink can't make them pass vacuously.
    """
    sim = hpv.Sim(
        location='nigeria',
        start=2000, stop=2050, dt=0.5,
        n_agents=6000,
        genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
        rand_seed=0,
    )
    sim.run()
    return sim


def test_no_double_counting_of_multigenotype_cancers(multigenotype_sim):
    """No agent may have cancerous=True on more than one HPV module simultaneously.

    Without cross-genotype cancellation, an agent with CIN in two
    genotypes would transition to cancerous in both modules, resulting
    in multiple ``cancerous`` flags for the same agent. With the fix,
    each agent has at most one cancerous flag across all modules.
    """
    sim = multigenotype_sim

    genotypes = [m for m in sim.diseases.values() if isinstance(m, hpv.HPV)]
    # Use .raw (all ever-created agent slots) so all modules have the same shape.
    n_slots = genotypes[0].cancerous.raw.shape[0]
    cancerous_count_per_agent = np.zeros(n_slots, dtype=int)
    for mod in genotypes:
        cancerous_count_per_agent += np.asarray(mod.cancerous.raw).astype(int)

    assert cancerous_count_per_agent.sum() > 0, (
        'no agent is cancerous at sim end — the dedup rule is untested here'
    )
    max_concurrent = int(cancerous_count_per_agent.max())
    n_double = int((cancerous_count_per_agent >= 2).sum())
    assert max_concurrent <= 1, (
        f'Agents with cancer from multiple genotypes simultaneously: '
        f'{n_double} agents with >=2 cancerous flags, max={max_concurrent}. '
        f'Cancer must be exclusive across genotypes per agent.'
    )


def test_hpvtotal_new_cancers_equals_per_module_sum(multigenotype_sim):
    """hpvtotal.new_cancers equals the sum of the per-module new_cancers.

    Also asserts no single module exceeds the total and no module records a
    negative count at any timestep.
    """
    sim = multigenotype_sim

    genotypes = [m for m in sim.diseases.values() if isinstance(m, hpv.HPV)]
    per_module_sums = {
        mod.genotype: float(mod.results.new_cancers[:].sum())
        for mod in genotypes
    }
    total_from_modules = sum(per_module_sums.values())
    total_hpvtotal = float(sim.results.all_hpv.new_cancers[:].sum())

    assert total_hpvtotal > 0, 'no cancers recorded — aggregation is untested here'

    # rtol=1e-9, not exact: pop_scale multiplication reorders the summation.
    assert np.isclose(total_from_modules, total_hpvtotal, rtol=1e-9), (
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


def test_hpvtotal_n_latent_is_a_union_not_a_sum():
    """all_hpv.n_latent must count each agent once even if latent for
    multiple genotypes simultaneously -- it is a _UNION_STATES entry (boolean
    OR across modules), not an element-wise sum of per-module n_latent.

    hpv_control_prob=1.0 forces latency on every clearance, so with 2
    genotypes co-infection is common and a naive sum would double-count.
    """
    sim = hpv.Sim(
        location='nigeria',
        start=1970, stop=2000, dt=0.5,
        n_agents=1000,
        genotypes=['hpv16', 'hpv18'],
        pars=dict(hpv_control_prob=1.0),
        rand_seed=0,
    )
    sim.run()

    genotypes = [m for m in sim.diseases.values() if isinstance(m, hpv.HPV)]
    naive_sum = sum(mod.results.n_latent[:] for mod in genotypes)
    union = sim.results.all_hpv.n_latent[:]

    assert union.max() > 0, 'no agent ever latent — the union logic is untested here'
    assert naive_sum.max() > union.max(), (
        'expected co-latency (naive sum > union) at hpv_control_prob=1.0 with '
        '2 genotypes -- if this no longer holds, either latency stopped '
        'firing or co-infection became too rare for this test to discriminate '
        'union-vs-sum; investigate before weakening the assertion below.'
    )
    assert (union <= naive_sum + 1e-6).all(), (
        'all_hpv.n_latent exceeds the naive per-module sum at some timestep -- '
        'a union count can never exceed a sum of its parts.'
    )