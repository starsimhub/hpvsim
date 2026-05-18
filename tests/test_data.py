"""Tests for hpvsim.data.country.load_country — country-data adapter."""

import pandas as pd

import hpvsim


def test_load_country_returns_expected_keys():
    out = hpvsim.data.load_country('nigeria')
    expected = {
        'age_data', 'birth_rate', 'death_rate', 'network_pars',
        'pop_total', 'pop_by_age',  # added in M02 for AgeMigration
    }
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
    """death_rate is a DataFrame ss.Deaths(death_rate=...) accepts directly.

    Columns match ss.Deaths' default UN-style metadata
    (Time, AgeGrpStart, Sex, mx); rates are per-1000 (matching default
    rate_units=1e-3) and Sex labels are 'Female'/'Male' (matching default
    sex_keys).
    """
    out = hpvsim.data.load_country('nigeria')
    df = out['death_rate']
    assert isinstance(df, pd.DataFrame)
    assert {'Time', 'AgeGrpStart', 'Sex', 'mx'}.issubset(df.columns), f'columns: {df.columns.tolist()}'
    assert len(df) > 0
    assert (df['mx'] >= 0).all()
    assert set(df['Sex'].unique()) <= {'Female', 'Male'}


def test_network_pars_shape():
    """network_pars carries layer_pars for v2 layers (m, c) plus shared debut."""
    out = hpvsim.data.load_country('nigeria')
    np_pars = out['network_pars']
    assert set(np_pars.keys()) == {'layer_pars', 'debut'}, \
        f'top-level keys: {list(np_pars.keys())}'
    layer_pars = np_pars['layer_pars']
    assert set(layer_pars.keys()) == {'m', 'c'}, f'layers: {list(layer_pars.keys())}'
    expected = {'partners', 'mixing', 'layer_probs', 'cross_layer', 'duration', 'acts'}
    for layer, lp in layer_pars.items():
        assert expected.issubset(lp.keys()), \
            f'layer {layer} missing keys: {expected - set(lp.keys())}'
    assert set(np_pars['debut'].keys()) == {'f', 'm'}


def test_unknown_location_raises():
    """Unknown location raises ValueError listing supported locations."""
    import pytest
    with pytest.raises(ValueError, match='nigeria'):
        hpvsim.data.load_country('atlantis')