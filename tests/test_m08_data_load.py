"""M08 T10: Rwanda HIV/ART data loader tests.

Verifies that the Rwanda HIV inputs (ART coverage by age/sex/year + an
initial HIV prevalence seed) are bundled into the package and exposed via
``hpv.data.load_hiv`` and ``hpv.HIV.from_location``.
"""

import numpy as np
import pandas as pd
import pytest

import hpvsim as hpv


def test_load_hiv_rwanda_returns_inputs():
    data = hpv.data.load_hiv('rwanda')
    assert data is not None

    # init_prev: a sane probability in [0, 1].
    init_prev = data['init_prev']
    assert isinstance(init_prev, float)
    assert 0.0 <= init_prev <= 1.0
    assert init_prev > 0.0  # a non-trivial seed

    # art_coverage: tidy long DataFrame keyed by age/sex/year.
    art = data['art_coverage']
    assert isinstance(art, pd.DataFrame)
    assert set(['age', 'sex', 'year', 'coverage']).issubset(art.columns)
    assert {'f', 'm'} == set(art['sex'].unique())
    # Coverage is a fraction. The upstream modeled series has a handful of
    # tiny rounding overshoots above 1.0 (max ~1.001); the loader preserves the
    # raw data faithfully rather than clipping (clipping is a T10b/T12 concern).
    assert art['coverage'].between(0.0, 1.02).all()
    # Spans the documented year range and single-year ages.
    assert art['year'].min() == 2004
    assert art['year'].max() == 2030


def test_load_hiv_unknown_location_raises():
    with pytest.raises(ValueError):
        hpv.data.load_hiv('atlantis')


def test_hiv_from_location_rwanda():
    h = hpv.HIV.from_location('rwanda')
    assert h.init_prev_data is not None
    assert np.isclose(float(h.init_prev_data), hpv.data.load_hiv('rwanda')['init_prev'])
