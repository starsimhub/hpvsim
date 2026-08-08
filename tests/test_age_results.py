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
    # n_agents must be large enough that the two snapshot years hold a
    # meaningfully DIFFERENT cancer-by-age profile: at n=2000 there is ~1
    # prevalent cancer, so 2010 and 2020 can coincidentally match (a false
    # "identical" trip) — which is exactly what a distribution reshuffle (e.g.
    # the starsim 3.5.0 upgrade) exposed. At n=15000 each year has ~10-16
    # cancers with a clear age-profile shift, so the guard is robust.
    sim = hpv.Sim(n_agents=15000, start=1990, stop=2021, dt=1.0,
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
    # 2010 and 2020 snapshots should differ — the cancer age-profile shifts over
    # the 30-year burn-in. If they were identical, step() might be overwriting
    # the same slot.
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


@pytest.fixture(scope='module')
def type_dist_sim():
    """One 4-genotype run recording both `cancers` and `cancerous_genotype_dist`.

    n_agents=10000 so a reasonable number of cancers (~8 at seed=0) have fired
    by 2020 even after small RNG shifts; with n=4000 the count hovers near zero
    and the precondition flakes under any RNG-affecting change. Both type-dist
    tests read the same two outputs off this one run.
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
    return sim


def test_age_results_type_distribution_sums_to_one(type_dist_sim):
    """cancerous_genotype_dist normalizes to a probability distribution per year."""
    ar = type_dist_sim.analyzers['ageresults']
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


def test_age_results_type_distribution_per_genotype_sums_match_total(type_dist_sim):
    """Sum-over-genotypes of per-bin raw counts == sum-over-genotypes-elsewhere
    cancerous count for that bin. Confirms type-dist's binning matches the
    aggregate 'cancers' binning at the raw-count level.
    """
    ar = type_dist_sim.analyzers['ageresults']
    # 'cancers' output is union-across-genotypes — undercounts when an agent
    # is cancerous in two genotypes (rare for cancer; cancer is attributed
    # to one genotype per agent in the natural-history model).
    cancers_arr = ar.outputs['cancers'][2020.0]
    dist_arr = ar.outputs['cancerous_genotype_dist'][2020.0]
    dist_total_per_bin = dist_arr.sum(axis=1)
    # Each agent is cancerous in exactly one genotype in the standard model,
    # so the dist sum equals the union count exactly.
    assert np.allclose(dist_total_per_bin, cancers_arr)
