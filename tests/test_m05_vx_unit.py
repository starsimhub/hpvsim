"""Unit tests for M05 vaccination components."""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv
from hpvsim.products import _load_vx_products


def test_load_vx_products_returns_dict_of_genotype_to_rel_imm():
    """_load_vx_products() returns {product_name: {genotype: rel_imm}} from CSV."""
    products = _load_vx_products()
    # Three default products from v2's CSV
    assert set(products.keys()) >= {'bivalent', 'quadrivalent', 'nonavalent'}
    # Bivalent has full protection against hpv16 and hpv18
    assert products['bivalent']['hpv16'] == pytest.approx(1.0)
    assert products['bivalent']['hpv18'] == pytest.approx(1.0)
    # Nonavalent has full protection against hi5
    assert products['nonavalent']['hi5'] == pytest.approx(1.0)
    # Bivalent has partial cross-protection against hi5
    assert 0 < products['bivalent']['hi5'] < 1.0


def test_load_vx_products_cached():
    """Repeat calls return the same dict object (module-level cache)."""
    first = _load_vx_products()
    second = _load_vx_products()
    assert first is second


def test_vx_constructor_with_name_loads_csv():
    """hpv.vx(name='bivalent') resolves rel_imm from the CSV."""
    from hpvsim.products import vx
    product = vx(name='bivalent')
    assert product._rel_imm['hpv16'] == pytest.approx(1.0)
    assert product._rel_imm['hpv18'] == pytest.approx(1.0)


def test_vx_constructor_with_rel_imm_uses_override():
    """hpv.vx(rel_imm={...}) uses the explicit dict and ignores the CSV."""
    from hpvsim.products import vx
    custom = {'hpv16': 0.7, 'hpv18': 0.6}
    product = vx(rel_imm=custom)
    assert product._rel_imm == custom


def test_vx_constructor_both_name_and_rel_imm_raises():
    """Providing both name and rel_imm is ambiguous; raise."""
    from hpvsim.products import vx
    with pytest.raises(ValueError, match='exactly one'):
        vx(name='bivalent', rel_imm={'hpv16': 0.5})


def test_vx_constructor_neither_raises():
    """Providing neither name nor rel_imm has no efficacy; raise."""
    from hpvsim.products import vx
    with pytest.raises(ValueError, match='exactly one'):
        vx()


def test_vx_unknown_name_raises_with_valid_names_listed():
    """Unknown product name surfaces the list of valid names."""
    from hpvsim.products import vx
    with pytest.raises(ValueError, match='bivalent.*quadrivalent.*nonavalent'):
        vx(name='not_a_real_vaccine')