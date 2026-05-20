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
    assert product.rel_imm['hpv16'] == pytest.approx(1.0)
    assert product.rel_imm['hpv18'] == pytest.approx(1.0)


def test_vx_constructor_with_rel_imm_uses_override():
    """hpv.vx(rel_imm={...}) uses the explicit dict and ignores the CSV."""
    from hpvsim.products import vx
    custom = {'hpv16': 0.7, 'hpv18': 0.6}
    product = vx(rel_imm=custom)
    assert product.rel_imm == custom


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


def _make_small_sim(genotypes=('hpv16', 'hpv18', 'hi5', 'ohr')):
    """Construct a small initialized 4-genotype sim suitable for administer tests."""
    sim = hpv.Sim(
        location='nigeria',
        start=2010, stop=2012,
        n_agents=200,
        genotypes=list(genotypes),
        rand_seed=0,
    )
    sim.init()
    return sim


def test_vx_administer_bumps_nab_imm_for_active_genotypes():
    """administer() writes per-genotype nab_imm; max-of-existing semantics."""
    from hpvsim.products import vx
    sim = _make_small_sim()
    product = vx(rel_imm={'hpv16': 1.0, 'hpv18': 0.5})
    product.init_pre(sim)  # bind to sim before use
    # Pick 20 agents and vaccinate them
    uids = sim.people.alive.uids[:20]
    pre_hpv16 = sim.diseases['hpv16'].nab_imm[uids].copy()
    pre_hpv18 = sim.diseases['hpv18'].nab_imm[uids].copy()
    product.administer(sim.people, uids)
    post_hpv16 = sim.diseases['hpv16'].nab_imm[uids]
    post_hpv18 = sim.diseases['hpv18'].nab_imm[uids]
    # hpv16 has rel_imm=1.0 -> every agent is sterilizing -> post = 1.0
    assert np.all(post_hpv16 == 1.0)
    # hpv18 has rel_imm=0.5 -> each agent's post is either 1.0 (sterilizing)
    # or 0.5 (leaky); never less than 0.5
    assert np.all(post_hpv18 >= 0.5)
    assert np.all(post_hpv18 <= 1.0)
    # No regressions in initial state
    assert np.all(post_hpv16 >= pre_hpv16)
    assert np.all(post_hpv18 >= pre_hpv18)


def test_vx_administer_skips_inactive_genotypes_silently():
    """A 9-valent product in a 4-genotype sim must not error."""
    from hpvsim.products import vx
    sim = _make_small_sim()  # has hpv16/hpv18/hi5/ohr only
    # Bivalent CSV has entries for hpv45, hi4, hr, lr which are NOT in this sim
    product = vx(name='bivalent')
    product.init_pre(sim)
    uids = sim.people.alive.uids[:10]
    # Must not raise
    product.administer(sim.people, uids)


def test_vx_administer_does_not_downgrade_natural_immunity():
    """If nab_imm is already higher than the vaccine peak, it is preserved."""
    from hpvsim.products import vx
    sim = _make_small_sim()
    product = vx(rel_imm={'hpv16': 0.5})
    product.init_pre(sim)
    uids = sim.people.alive.uids[:10]
    # Force-bump nab_imm to 0.95 (simulating natural clearance immunity)
    sim.diseases['hpv16'].nab_imm[uids] = 0.95
    product.administer(sim.people, uids)
    # leaky floor is 0.5 (rel_imm) — must not downgrade the 0.95
    assert np.all(sim.diseases['hpv16'].nab_imm[uids] >= 0.95)


def test_vx_administer_empty_uids_is_noop():
    """Calling administer with empty uids does nothing."""
    from hpvsim.products import vx
    sim = _make_small_sim()
    product = vx(name='bivalent')
    product.init_pre(sim)
    # Should not raise
    product.administer(sim.people, sim.people.alive.uids[:0])