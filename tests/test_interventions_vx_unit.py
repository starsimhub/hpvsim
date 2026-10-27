"""Unit tests for vaccination intervention components."""
import numpy as np
import pytest
import sciris as sc

import hpvsim as hpv
from hpvsim.products import _load_vx_products


def test_load_vx_products_returns_dict_of_genotype_to_rel_imm():
    """_load_vx_products() returns {product_name: {genotype: rel_imm}} from CSV."""
    products = _load_vx_products()
    # Three default products from the CSV
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


def _make_small_sim_with_product(product, genotypes=('hpv16', 'hpv18', 'hi5', 'ohr')):
    """Construct a small 4-genotype sim with `product` wired through ss.vaccinate.

    coverage=0 means it never vaccinates on its own — we call
    product.administer() directly in the test. Wiring it through the delivery
    is purely so that sim.init() initializes the product's distributions.

    Returns (sim, initialized_product): the sim copies the modules it is
    given, so the product retrieved from the sim is a different object from
    the one passed in.
    """
    import starsim as ss
    intv = ss.vaccinate(product, coverage=0.0, name='routine_vx')
    sim = hpv.Sim(
        location='nigeria',
        start=2010, stop=2012,
        n_agents=200,
        genotypes=list(genotypes),
        rand_seed=0,
        interventions=[intv],
    )
    sim.init()
    # Starsim deep-copies modules at init, so the passed-in product is stale.
    initialized_product = sim.interventions['routine_vx'].product
    return sim, initialized_product


def test_vx_administer_bumps_vax_imm_for_active_genotypes():
    """administer() writes per-genotype vax_imm; max-of-existing semantics."""
    from hpvsim.products import vx
    sim, product = _make_small_sim_with_product(vx(rel_imm={'hpv16': 1.0, 'hpv18': 0.5}))
    uids = sim.people.alive.uids[:20]
    pre_hpv16 = sim.diseases['hpv16'].vax_imm[uids].copy()
    pre_hpv18 = sim.diseases['hpv18'].vax_imm[uids].copy()
    product.administer(uids)
    post_hpv16 = sim.diseases['hpv16'].vax_imm[uids]
    post_hpv18 = sim.diseases['hpv18'].vax_imm[uids]
    # hpv16 has rel_imm=1.0: sterilizing agents -> vax_imm=1.0, leaky -> 1.0*0.95=0.95
    assert np.all(post_hpv16 >= 0.95)
    assert np.all(post_hpv16 <= 1.0)
    # hpv18 has rel_imm=0.5: sterilizing -> vax_imm=0.5, leaky -> 0.5*0.95=0.475
    assert np.all(post_hpv18 >= 0.475)
    assert np.all(post_hpv18 <= 0.5)
    # max-of-existing: vax_imm never decreases.
    assert np.all(post_hpv16 >= pre_hpv16)
    assert np.all(post_hpv18 >= pre_hpv18)
    assert np.all(sim.diseases['hpv16'].nab_imm[uids] == 0.0), \
        'administer() must not touch nab_imm (clearance-only array)'
    assert np.all(sim.diseases['hpv18'].nab_imm[uids] == 0.0), \
        'administer() must not touch nab_imm (clearance-only array)'


def test_vx_administer_skips_inactive_genotypes_silently():
    """A 9-valent product in a 4-genotype sim must not error."""
    from hpvsim.products import vx
    # has hpv16/hpv18/hi5/ohr only; bivalent CSV has entries for hpv45, hi4, hr, lr
    sim, product = _make_small_sim_with_product(vx(name='bivalent'))
    uids = sim.people.alive.uids[:10]
    # Must not raise
    product.administer(uids)


def test_vx_administer_does_not_downgrade_natural_immunity():
    """Vaccine writes vax_imm; leaves clearance-conferred nab_imm untouched.

    Natural and vaccine immunity live in separate arrays, so the vaccine cannot
    downgrade natural immunity. vax_imm also respects max-of-existing when the
    agent already carries prior vaccine immunity.
    """
    from hpvsim.products import vx
    sim, product = _make_small_sim_with_product(vx(rel_imm={'hpv16': 0.5}))
    uids = sim.people.alive.uids[:10]
    # Prior vax_imm=0.8 sits above the new vaccine's leaky floor of 0.5.
    sim.diseases['hpv16'].nab_imm[uids] = 0.95
    sim.diseases['hpv16'].vax_imm[uids] = 0.8
    product.administer(uids)
    assert np.all(sim.diseases['hpv16'].nab_imm[uids] == 0.95), \
        'administer() must not touch nab_imm (clearance-only array)'
    assert np.all(sim.diseases['hpv16'].vax_imm[uids] >= 0.8), \
        'administer() must not downgrade existing vax_imm'


def test_vx_administer_empty_uids_is_noop():
    """Calling administer with empty uids does nothing."""
    from hpvsim.products import vx
    sim, product = _make_small_sim_with_product(vx(name='bivalent'))
    # Should not raise
    product.administer(sim.people.alive.uids[:0])
