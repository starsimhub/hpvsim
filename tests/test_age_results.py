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
    """Snapshot of cancer counts by age bin at end-of-year matches alive cancerous agents."""
    edges = np.array([0., 20., 40., 60., 100.])
    sim = hpv.Sim(n_agents=2000, start=1990, stop=2021, dt=1.0,
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
    """Age labels follow v2 convention: '0-20', '20-40', ..., '<last>+'."""
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
