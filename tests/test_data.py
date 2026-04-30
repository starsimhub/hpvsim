"""Tests for hpvsim.data.country.load_country — country-data adapter."""

import pandas as pd

import hpvsim


def test_load_country_returns_expected_keys():
    """load_country returns a dict with exactly the expected top-level keys."""
    out = hpvsim.data.load_country('nigeria')
    expected = {'age_data', 'birth_rate', 'death_rate', 'network_pars'}
    assert set(out.keys()) == expected, f'unexpected keys: {set(out.keys())}'


def test_age_data_shape():
    """age_data is a DataFrame with the columns ss.People accepts."""
    out = hpvsim.data.load_country('nigeria')
    df = out['age_data']
    assert isinstance(df, pd.DataFrame)
    assert {'age', 'value'}.issubset(df.columns), f'columns: {df.columns.tolist()}'
    assert len(df) > 0
    assert (df['value'] >= 0).all()


def test_birth_rate_shape():
    """birth_rate is a DataFrame ss.Births(birth_rate=...) accepts."""
    out = hpvsim.data.load_country('nigeria')
    df = out['birth_rate']
    assert isinstance(df, pd.DataFrame)
    assert {'Year', 'CBR'}.issubset(df.columns), f'columns: {df.columns.tolist()}'
    assert len(df) > 0
    assert (df['CBR'] >= 0).all()


def test_death_rate_shape():
    """death_rate is a DataFrame ss.Deaths(death_rate=...) accepts."""
    out = hpvsim.data.load_country('nigeria')
    df = out['death_rate']
    assert isinstance(df, pd.DataFrame)
    assert {'Year', 'AgeGrp', 'Sex', 'Rate'}.issubset(df.columns), f'columns: {df.columns.tolist()}'
    assert len(df) > 0
    assert (df['Rate'] >= 0).all()


def test_network_pars_per_layer():
    """network_pars contains entries for the two v2 layers m/c."""
    out = hpvsim.data.load_country('nigeria')
    np_pars = out['network_pars']
    assert set(np_pars.keys()) == {'m', 'c'}, f'layers: {list(np_pars.keys())}'
    expected = {'partners', 'mixing', 'layer_probs', 'cross_layer', 'duration', 'acts'}
    for layer, layer_pars in np_pars.items():
        assert expected.issubset(layer_pars.keys()), \
            f'layer {layer} missing keys: {expected - set(layer_pars.keys())}'


def test_unknown_location_raises():
    """Unknown location raises ValueError listing supported locations."""
    import pytest
    with pytest.raises(ValueError, match='nigeria'):
        hpvsim.data.load_country('atlantis')