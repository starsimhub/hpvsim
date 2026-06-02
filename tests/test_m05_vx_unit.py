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


def _make_small_sim_with_product(product, genotypes=('hpv16', 'hpv18', 'hi5', 'ohr')):
    """Construct a small 4-genotype sim with `product` wired through ss.routine_vx.

    The intervention's prob=0.0 means it never vaccinates on its own —
    we call product.administer() directly in the test. Wiring it through
    the intervention is purely so that sim.init() runs the full dist-init
    chain that initializes the product's bare-attribute distributions.

    Returns (sim, initialized_product) — Starsim deep-copies modules during
    init_module_attrs, so the product reference retrieved from the sim after
    init is distinct from (and replaces) the one passed in.
    """
    import starsim as ss
    intv = ss.routine_vx(product=product, prob=0.0, start_year=2010)
    sim = hpv.Sim(
        location='nigeria',
        start=2010, stop=2012,
        n_agents=200,
        genotypes=list(genotypes),
        rand_seed=0,
        interventions=[intv],
    )
    sim.init()
    # Retrieve the initialized copy from the sim — Starsim deep-copies modules
    # during init_module_attrs, so the original `product` variable is stale.
    initialized_product = sim.interventions['routine_vx'].product
    return sim, initialized_product


def test_vx_administer_bumps_vax_imm_for_active_genotypes():
    """administer() writes per-genotype vax_imm; max-of-existing semantics."""
    from hpvsim.products import vx
    sim, product = _make_small_sim_with_product(vx(rel_imm={'hpv16': 1.0, 'hpv18': 0.5}))
    # Pick 20 agents and vaccinate them
    uids = sim.people.alive.uids[:20]
    pre_hpv16 = sim.diseases['hpv16'].vax_imm[uids].copy()
    pre_hpv18 = sim.diseases['hpv18'].vax_imm[uids].copy()
    product.administer(sim.people, uids)
    post_hpv16 = sim.diseases['hpv16'].vax_imm[uids]
    post_hpv18 = sim.diseases['hpv18'].vax_imm[uids]
    # hpv16 has rel_imm=1.0: sterilizing agents -> vax_imm=1.0, leaky -> 1.0*0.95=0.95
    assert np.all(post_hpv16 >= 0.95)
    assert np.all(post_hpv16 <= 1.0)
    # hpv18 has rel_imm=0.5: sterilizing -> vax_imm=0.5, leaky -> 0.5*0.95=0.475
    assert np.all(post_hpv18 >= 0.475)
    assert np.all(post_hpv18 <= 0.5)
    # No regressions in initial state
    assert np.all(post_hpv16 >= pre_hpv16)
    assert np.all(post_hpv18 >= pre_hpv18)
    # nab_imm must be untouched — vaccine writes only to vax_imm
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
    product.administer(sim.people, uids)


def test_vx_administer_does_not_downgrade_natural_immunity():
    """Vaccine writes vax_imm; leaves clearance-conferred nab_imm untouched.

    Natural immunity is in nab_imm. Vaccine immunity is in vax_imm.  Since
    they are separate arrays, the vaccine can never downgrade natural immunity.
    This test also verifies that vax_imm respects the max-of-existing rule when
    the agent already has prior vaccine immunity.
    """
    from hpvsim.products import vx
    sim, product = _make_small_sim_with_product(vx(rel_imm={'hpv16': 0.5}))
    uids = sim.people.alive.uids[:10]
    # Simulate natural clearance immunity (nab_imm=0.95) and prior vaccine
    # immunity (vax_imm=0.8, higher than the new vaccine's leaky floor of 0.5).
    sim.diseases['hpv16'].nab_imm[uids] = 0.95
    sim.diseases['hpv16'].vax_imm[uids] = 0.8
    product.administer(sim.people, uids)
    # nab_imm must be completely untouched — vaccine writes only to vax_imm
    assert np.all(sim.diseases['hpv16'].nab_imm[uids] == 0.95), \
        'administer() must not touch nab_imm (clearance-only array)'
    # vax_imm must respect max-of-existing: prior 0.8 > leaky floor 0.5, so
    # vax_imm must still be >= 0.8 (sterilizing draw may push it to 1.0, but
    # it must never fall below the pre-administer value).
    assert np.all(sim.diseases['hpv16'].vax_imm[uids] >= 0.8), \
        'administer() must not downgrade existing vax_imm'


def test_vx_administer_empty_uids_is_noop():
    """Calling administer with empty uids does nothing."""
    from hpvsim.products import vx
    sim, product = _make_small_sim_with_product(vx(name='bivalent'))
    # Should not raise
    product.administer(sim.people, sim.people.alive.uids[:0])


def test_cast_sex_none_returns_none():
    from hpvsim.interventions import _cast_sex
    assert _cast_sex(None) is None


def test_cast_sex_f_returns_zero_set():
    from hpvsim.interventions import _cast_sex
    assert _cast_sex('f') == {0}


def test_cast_sex_m_returns_one_set():
    from hpvsim.interventions import _cast_sex
    assert _cast_sex('m') == {1}


def test_cast_sex_int_zero_or_one():
    from hpvsim.interventions import _cast_sex
    assert _cast_sex(0) == {0}
    assert _cast_sex(1) == {1}


def test_cast_sex_list_both():
    from hpvsim.interventions import _cast_sex
    assert _cast_sex(['f', 'm']) == {0, 1}
    assert _cast_sex([0, 1]) == {0, 1}


def test_cast_sex_invalid_raises():
    from hpvsim.interventions import _cast_sex
    with pytest.raises(ValueError, match='sex'):
        _cast_sex('female')
    with pytest.raises(ValueError, match='sex'):
        _cast_sex(2)
    with pytest.raises(ValueError, match='sex'):
        _cast_sex(['x', 'y'])
    with pytest.raises(ValueError, match='empty'):
        _cast_sex([])


def _make_plain_small_sim(genotypes=('hpv16', 'hpv18', 'hi5', 'ohr')):
    """A small initialized 4-genotype sim without any product/intervention.

    Used for eligibility-composition tests that exercise the callable
    directly against sim.people without needing a product context.
    """
    sim = hpv.Sim(
        location='nigeria',
        start=2010, stop=2012,
        n_agents=200,
        genotypes=list(genotypes),
        rand_seed=0,
    )
    sim.init()
    return sim


def test_compose_vaccine_eligibility_no_filters_returns_all_alive():
    """No age, no sex, no extra -> all alive agents are eligible."""
    from hpvsim.interventions import _compose_vaccine_eligibility
    sim = _make_plain_small_sim()
    elig = _compose_vaccine_eligibility(age_range=None, sex=None, extra=None)
    uids = elig(sim)
    # All eligible uids must be alive
    assert np.all(sim.people.alive[uids])
    # And there are some
    assert len(uids) > 0


def test_compose_vaccine_eligibility_age_range_filters():
    """age_range=[lo, hi] yields only agents with lo <= age < hi."""
    from hpvsim.interventions import _compose_vaccine_eligibility
    sim = _make_plain_small_sim()
    elig = _compose_vaccine_eligibility(age_range=[9, 14], sex=None, extra=None)
    uids = elig(sim)
    ages = sim.people.age[uids]
    assert np.all(ages >= 9)
    assert np.all(ages < 14)


def test_compose_vaccine_eligibility_sex_female_filters():
    """sex='f' yields only agents with people.female==True (sex==0)."""
    from hpvsim.interventions import _compose_vaccine_eligibility
    sim = _make_plain_small_sim()
    elig = _compose_vaccine_eligibility(age_range=None, sex='f', extra=None)
    uids = elig(sim)
    # Starsim encodes sex via people.female (BoolState); all returned uids must be female
    assert np.all(sim.people.female[uids])


def test_compose_vaccine_eligibility_sex_male_filters():
    """sex='m' yields only agents with people.male==True (sex==1)."""
    from hpvsim.interventions import _compose_vaccine_eligibility
    sim = _make_plain_small_sim()
    elig = _compose_vaccine_eligibility(age_range=None, sex='m', extra=None)
    uids = elig(sim)
    # Starsim encodes sex via people.male (BoolArr); all returned uids must be male
    assert np.all(sim.people.male[uids])


def test_compose_vaccine_eligibility_sex_both_applies_no_filter():
    """sex=['f', 'm'] applies no sex filter."""
    from hpvsim.interventions import _compose_vaccine_eligibility
    sim = _make_plain_small_sim()
    elig_both = _compose_vaccine_eligibility(age_range=None, sex=['f', 'm'], extra=None)
    elig_none = _compose_vaccine_eligibility(age_range=None, sex=None, extra=None)
    assert set(elig_both(sim)) == set(elig_none(sim))


def test_compose_vaccine_eligibility_extra_callback_intersects():
    """An `extra` callable intersects further with age/sex conditions."""
    from hpvsim.interventions import _compose_vaccine_eligibility
    sim = _make_plain_small_sim()
    # Eligible: agents alive AND age >= 20 (extra is the only filter)
    extra = lambda s: (s.people.age >= 20).uids
    elig = _compose_vaccine_eligibility(age_range=None, sex=None, extra=extra)
    uids = elig(sim)
    assert np.all(sim.people.age[uids] >= 20)
    assert np.all(sim.people.alive[uids])


def test_compose_vaccine_eligibility_combines_age_sex_extra():
    """All three filters compose via intersection."""
    from hpvsim.interventions import _compose_vaccine_eligibility
    sim = _make_plain_small_sim()
    extra = lambda s: (s.people.age >= 12).uids
    elig = _compose_vaccine_eligibility(age_range=[9, 14], sex='f', extra=extra)
    uids = elig(sim)
    ages = sim.people.age[uids]
    assert np.all(ages >= 12)
    assert np.all(ages < 14)
    # Starsim uses people.female for sex==0 (female)
    assert np.all(sim.people.female[uids])


def test_base_vaccination_accepts_v2_args():
    """hpv.BaseVaccination accepts age_range, sex, eligibility as kwargs."""
    from hpvsim.interventions import BaseVaccination, routine_vx
    from hpvsim.products import vx
    intv = routine_vx(
        product=vx(name='bivalent'),
        prob=0.9,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
    )
    assert isinstance(intv, BaseVaccination)
    assert intv.age_range == [9, 14]
    assert intv.sex == {0}


def test_parse_product_str_resolves_to_default_vx():
    """routine_vx(product='bivalent', ...) resolves through hpv.vx(name='bivalent')."""
    from hpvsim.interventions import routine_vx
    from hpvsim.products import vx
    intv = routine_vx(product='bivalent', prob=0.5, start_year=2020)
    assert isinstance(intv.product, vx)
    assert intv.product.rel_imm['hpv16'] == pytest.approx(1.0)


def test_routine_vx_isinstance_chain():
    """Class identity preserved across the diamond."""
    import starsim as ss
    from hpvsim.interventions import routine_vx, campaign_vx, BaseVaccination
    intv_r = routine_vx(product='bivalent', prob=0.5, start_year=2020)
    intv_c = campaign_vx(product='bivalent', prob=0.5, years=[2020])
    assert isinstance(intv_r, BaseVaccination)
    assert isinstance(intv_r, ss.BaseVaccination)
    assert isinstance(intv_r, ss.RoutineDelivery)
    assert isinstance(intv_c, BaseVaccination)
    assert isinstance(intv_c, ss.CampaignDelivery)


def test_campaign_vx_passes_years_through():
    """campaign_vx accepts years= and stores it for init_pre."""
    from hpvsim.interventions import campaign_vx
    intv = campaign_vx(product='bivalent', prob=[0.7, 0.5], years=[2020, 2021])
    assert list(intv.years) == [2020, 2021]

