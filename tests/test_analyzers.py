"""Tests for hpv.AgeResults."""
import numpy as np
import pytest

import hpvsim as hpv


def test_age_results_class_exists():
    """AgeResults is exported and inherits ss.Analyzer."""
    import starsim as ss
    assert hasattr(hpv, 'AgeResults')
    assert issubclass(hpv.AgeResults, ss.Analyzer)


def test_age_results_produces_cancer_incidence_by_age():
    """A sim with AgeResults populates cancer_incidence_by_age."""
    sim = hpv.Sim(
        n_agents=5000, location='nigeria',
        start=1990, stop=2030, dt=0.25, rand_seed=0,
        analyzers=[hpv.AgeResults(results=('cancer',), year=[2020, 2025])],
    )
    sim.run()
    az = sim.analyzers.age_results
    arr = np.asarray(az.results.cancer_incidence_by_age)
    # shape: (n_years, n_bins) — year=[2020, 2025] → 2 rows; default 5-yr bins 0–100 → 20
    assert arr.shape == (2, 20)
    assert (arr >= 0).all()


def test_age_results_rejects_unsupported_keys():
    """M02 supports only 'cancer'; passing other keys raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        hpv.AgeResults(results=('cancer', 'cins'), year=[2020])
    with pytest.raises(NotImplementedError):
        hpv.AgeResults(results=('hpv',), year=[2020])