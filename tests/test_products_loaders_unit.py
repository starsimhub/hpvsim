"""Unit tests for product CSV loader helpers."""
import pytest
from hpvsim.products import (
    _load_dx_products,
    _load_tx_products,
    _load_txvx_products,
)


def test_dx_loader_returns_expected_products():
    dxs = _load_dx_products()
    expected = {'via', 'lbc', 'pap', 'colposcopy', 'hpv', 'hpv1618',
                'hpv_type', 'txvx_assigner', 'tx_assigner'}
    assert set(dxs.keys()) == expected


def test_dx_loader_has_state_and_result_columns():
    dxs = _load_dx_products()
    via = dxs['via']
    cols = set(via.columns)
    assert {'state', 'genotype', 'result', 'probability'} <= cols


def test_tx_loader_returns_expected_products():
    txs = _load_tx_products()
    assert {'ablation', 'excision', 'txvx1', 'txvx2'} <= set(txs.keys())


def test_tx_loader_has_efficacy_column():
    txs = _load_tx_products()
    df = txs['ablation']
    assert 'efficacy' in df.columns
    assert 'state' in df.columns
    assert 'genotype' in df.columns


def test_txvx_loader_returns_expected_products():
    txvxs = _load_txvx_products()
    assert {'txvx1', 'txvx2'} == set(txvxs.keys())


def test_txvx_loader_returns_genotype_rel_imm_dict():
    txvxs = _load_txvx_products()
    d = txvxs['txvx1']
    assert isinstance(d, dict)
    assert all(isinstance(v, float) for v in d.values())


def test_loaders_are_cached():
    a = _load_dx_products()
    b = _load_dx_products()
    assert a is b  # lru_cache returns same dict identity


def test_dx_loader_includes_latent_rows():
    """The dx CSV references the latent state; loader should preserve those rows."""
    dxs = _load_dx_products()
    hpv_df = dxs['hpv']
    assert 'latent' in hpv_df['state'].unique()
