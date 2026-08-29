"""Tests for hpvsim.data.country.load_country — country-data adapter."""

import pandas as pd

import hpvsim


def test_load_country_returns_expected_keys():
    out = hpvsim.data.load_country('nigeria')
    expected = {
        'age_data', 'birth_rate', 'death_rate', 'network_pars',
        'pop_total', 'pop_by_age',  # added for AgeMigration
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
    """network_pars is the flat NetworkPars dict — every knob at the top level."""
    out = hpvsim.data.load_country('nigeria')
    np_pars = out['network_pars']
    expected_top = {
        'm_cross_layer', 'f_cross_layer',
        'debut_f', 'debut_m',
        'm_partners_marital', 'm_partners_casual',
        'f_partners_marital', 'f_partners_casual',
        'acts_marital', 'acts_casual',
        'dur_pship_marital', 'dur_pship_casual',
        'age_act_pars_marital', 'age_act_pars_casual',
        'mixing_marital', 'mixing_casual',
        'layer_probs_marital', 'layer_probs_casual',
    }
    assert set(np_pars.keys()) == expected_top, \
        f'top-level keys: {sorted(np_pars.keys())}'


def test_unknown_location_raises():
    """Unknown location raises ValueError with a suggestion-based message."""
    import pytest
    with pytest.raises(ValueError, match='not recognized'):
        hpvsim.data.load_country('atlantis')