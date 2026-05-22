"""Unit tests for hpv.AgeResults analyzer."""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv

# Pattern note: hpv.Sim deep-copies its inputs (sc.mergedicts(_copy=True)),
# so the user-side AgeResults reference passed to analyzers=[...] becomes
# stale after sim construction. Tests construct the analyzer inline and
# retrieve the live instance post-run via sim.analyzers['ageresults'].


def test_age_results_importable_and_constructible():
    """hpv.AgeResults exists at top level and accepts a result_args dict."""
    ar = hpv.AgeResults(
        result_args=sc.objdict(
            cancers=sc.objdict(
                years=[2020],
                edges=np.array([0., 20., 40., 60., 100.]),
            ),
        ),
    )
    assert isinstance(ar, hpv.AgeResults)
    # result_args stored as objdict whether passed as dict or objdict
    assert 'cancers' in ar.result_args
    assert list(ar.result_args.cancers.years) == [2020]


def test_age_results_cancers_by_age_basic():
    """Snapshot of cancer counts by age bin at end-of-year matches alive cancerous agents.

    ``stop=2020`` (not 2021) so that the end-of-sim state coincides with the
    analyzer's 2020-end snapshot tick — otherwise an agent firing cancer in
    2021 would show up in the end-of-sim union but not in the snapshot.
    """
    edges = np.array([0., 20., 40., 60., 100.])
    sim = hpv.Sim(n_agents=2000, start=1990, stop=2020, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancers=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    ar = sim.analyzers['ageresults']

    df = ar.to_dataframe(key='cancers')
    assert list(df.index) == [2020.0]
    assert df.shape == (1, len(edges) - 1)
    # Counts must be non-negative integers (whole agents binned).
    assert (df.values >= 0).all()
    # Total cancers across age bins should match the union of per-genotype
    # cancerous BoolStates among alive agents at end of 2020.
    people = sim.people
    alive = people.alive.values
    cancerous_any = np.zeros_like(alive)
    for mod in sim.diseases.values():
        if isinstance(mod, hpv.HPV):
            cancerous_any |= mod.cancerous.values
    expected_total = int((cancerous_any & alive).sum())
    assert int(df.values.sum()) == expected_total


def test_age_results_multi_year_snapshot():
    """Two years requested -> two snapshots stored, both non-empty schemas."""
    edges = np.array([0., 30., 60., 100.])
    sim = hpv.Sim(n_agents=2000, start=1990, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancers=sc.objdict(years=[2010, 2020], edges=edges),
                      ),
                  )])
    sim.run()
    ar = sim.analyzers['ageresults']
    df = ar.to_dataframe(key='cancers')
    assert list(df.index) == [2010.0, 2020.0]
    assert df.shape == (2, len(edges) - 1)
    # 2010 and 2020 snapshots should differ — burden grows over the 30-year burn-in.
    # If they were identical, step() might be overwriting the same slot.
    assert not np.array_equal(df.loc[2010.0].values, df.loc[2020.0].values)


def test_age_results_age_label_schema():
    """Age labels are '<lo>-<hi>' for inner bins and '<last>+' for the final bin."""
    edges = np.array([0., 20., 40., 60., 100.])
    sim = hpv.Sim(n_agents=200, start=2019, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancers=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    ar = sim.analyzers['ageresults']
    df = ar.to_dataframe(key='cancers')
    assert list(df.columns) == ['0-20', '20-40', '40-60', '60+']


def test_age_results_hpv_prevalence_by_age():
    """hpv_prevalence is binned-infected / binned-alive per age bin, in [0,1]."""
    edges = np.array([0., 20., 40., 60., 100.])
    sim = hpv.Sim(n_agents=2000, start=1990, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          hpv_prevalence=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    ar = sim.analyzers['ageresults']
    df = ar.to_dataframe(key='hpv_prevalence')
    assert df.shape == (1, len(edges) - 1)
    vals = df.values[0]
    assert (vals >= 0).all() and (vals <= 1 + 1e-9).all()


def test_age_results_cancer_incidence_by_age():
    """cancer_incidence is per-100k new cancers among at-risk females per age bin."""
    edges = np.array([0., 30., 60., 100.])
    sim = hpv.Sim(n_agents=4000, start=1990, stop=2021, dt=1.0,
                  rand_seed=0, analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancer_incidence=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    ar = sim.analyzers['ageresults']
    df = ar.to_dataframe(key='cancer_incidence')
    assert df.shape == (1, len(edges) - 1)
    # Incidence rates are non-negative reals.
    assert (df.values >= 0).all()


def test_age_results_type_distribution_sums_to_one():
    """cancerous_genotype_dist normalizes to a probability distribution per year.

    Uses n_agents=10000 so a reasonable number of cancers (~8 at seed=0) have
    fired by 2020 even after small RNG shifts; with n=4000 the count hovers
    near zero and the precondition flakes under any RNG-affecting change.
    """
    edges = np.array([0., 100.])  # one age window — all ages
    sim = hpv.Sim(n_agents=10000, start=1990, stop=2021, dt=1.0,
                  rand_seed=0,
                  genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
                  analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancerous_genotype_dist=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    ar = sim.analyzers['ageresults']
    df = ar.to_dataframe(key='cancerous_genotype_dist')
    # Columns are genotype keys (hpv16, hpv18, hi5, ohr).
    assert list(df.columns) == ['hpv16', 'hpv18', 'hi5', 'ohr']
    # Precondition: the scenario must actually produce cancers so the
    # normalization path is exercised; if the sim defaults change to a
    # zero-cancer scenario the test would otherwise silently pass.
    row_sum = float(df.iloc[0].sum())
    assert row_sum > 0, (
        'Test precondition failed: no cancers in 2020. Adjust n_agents or '
        'stop year so that the normalization path is exercised.'
    )
    assert abs(row_sum - 1.0) < 1e-9


def test_age_results_type_distribution_per_genotype_sums_match_total():
    """Sum-over-genotypes of per-bin raw counts == sum-over-genotypes-elsewhere
    cancerous count for that bin. Confirms type-dist's binning matches the
    aggregate 'cancers' binning at the raw-count level.

    n_agents=10000 to match the sister test so both tests exercise the
    normalization paths even after RNG-affecting changes.
    """
    edges = np.array([0., 40., 100.])
    sim = hpv.Sim(n_agents=10000, start=1990, stop=2021, dt=1.0,
                  rand_seed=0,
                  genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
                  analyzers=[hpv.AgeResults(
                      result_args=sc.objdict(
                          cancers=sc.objdict(years=[2020], edges=edges),
                          cancerous_genotype_dist=sc.objdict(years=[2020], edges=edges),
                      ),
                  )])
    sim.run()
    ar = sim.analyzers['ageresults']
    # 'cancers' output is union-across-genotypes — undercounts when an agent
    # is cancerous in two genotypes (rare for cancer; cancer is attributed
    # to one genotype per agent in the natural-history model). Use a generous
    # tolerance: total dist count >= union count (each multi-genotype agent
    # contributes to dist but counts once in union).
    cancers_arr = ar.outputs['cancers'][2020.0]
    dist_arr = ar.outputs['cancerous_genotype_dist'][2020.0]
    dist_total_per_bin = dist_arr.sum(axis=1)
    # Each agent is cancerous in exactly one genotype in the standard model,
    # so the dist sum equals the union count exactly.
    assert np.allclose(dist_total_per_bin, cancers_arr)


@pytest.mark.slow
def test_age_results_v2_parity_cancers():
    """v3 AgeResults vs v2 age_results: per-bin cancer counts agree within +/- 30%.

    Note: v2 and v3 do NOT share an RNG stream (different framework). The
    parity gate is that the two implementations bin the same simulated
    population the same way; we match the simulation closely enough that
    the per-bin counts agree within calibration tolerance. Sim configs are
    matched on n_agents, location, start/stop, dt, and rand_seed.
    """
    edges = np.array([0., 30., 50., 70., 100.])
    seed = 0
    n_agents = 2000

    # ----- v3 run -----
    sim_v3 = hpv.Sim(n_agents=n_agents, start=1990, stop=2021, dt=1.0,
                     rand_seed=seed, analyzers=[hpv.AgeResults(
                         result_args=sc.objdict(
                             cancers=sc.objdict(years=[2020], edges=edges),
                         ),
                     )])
    sim_v3.run()
    ar_v3 = sim_v3.analyzers['ageresults']
    v3_counts = ar_v3.outputs['cancers'][2020.0]

    # ----- v2 run -----
    # Allowed: tests may import from _v2_legacy/ as regression anchors.
    from hpvsim._v2_legacy import sim as v2_sim_mod
    from hpvsim._v2_legacy import analysis as v2_analysis

    v2_ar = v2_analysis.age_results(
        result_args=sc.objdict(
            cancers=sc.objdict(years=[2020], edges=edges),
        ),
    )
    sim_v2 = v2_sim_mod.Sim(n_agents=n_agents, start=1990, end=2020,
                            dt=1.0, rand_seed=seed,
                            analyzers=[v2_ar])
    sim_v2.run()
    # v2 Sim also deep-copies analyzers; retrieve the live instance.
    live_v2_ar = sim_v2['analyzers'][0]
    v2_counts = live_v2_ar.results['cancers'][2020]

    # Compare per-bin counts. Allow per-bin abs(rel error) <= 0.30 with a
    # floor of 5 agents (small bins are noisy at n_agents=2000).
    tol = 0.30
    floor = 5.0
    for i in range(len(edges) - 1):
        a, b = float(v3_counts[i]), float(v2_counts[i])
        denom = max(abs(b), floor)
        rel = abs(a - b) / denom
        assert rel <= tol, (
            f'AgeResults v2 parity failure in bin '
            f'{edges[i]}-{edges[i+1]}: v3={a}, v2={b}, rel={rel:.3f}'
        )
