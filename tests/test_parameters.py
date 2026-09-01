"""Tests for hpv.Sim's parameter-routing surface: the user-facing
equivalence between bare kwargs, pars=, and calib_pars."""
import pytest
import starsim as ss

import hpvsim as hpv


def make_sim(**kw):
    if 'rand_seed' not in kw and 'rand_seed' not in (kw.get('pars') or {}):
        kw['rand_seed'] = 0
    return hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, **kw)


def test_expanddict():
    """expanddict converts flat dotted keys to nested dicts, merging shared prefixes."""
    assert hpv.expanddict({'beta': 0.15}) == {'beta': 0.15}
    assert hpv.expanddict({'hpv16.cin_fn.k': 0.77}) == {'hpv16': {'cin_fn': {'k': 0.77}}}
    assert hpv.expanddict({'hpv16.cin_fn.k': 0.77, 'hpv16.beta': 0.3}) == {
        'hpv16': {'cin_fn': {'k': 0.77}, 'beta': 0.3}
    }


def test_par_routing():
    """route_pars: a bare registry key broadcasts to every matching category;
    a per-instance nested override wins over a same-key broadcast regardless
    of input order; cross_immunity/network scalar pars route to their module."""
    sim = make_sim(genotypes=[16, 18])
    sim.init()

    # Bare registry key broadcasts to both genotypes.
    hpv.route_pars(sim, pars={'ms_agent_ratio': 3})
    assert sim.diseases.hpv16.pars.ms_agent_ratio == 3
    assert sim.diseases.hpv18.pars.ms_agent_ratio == 3

    # Per-genotype override wins even when it appears BEFORE the broadcast
    # key in the input dict (order must not matter).
    hpv.route_pars(sim, pars={'hpv18': {'ms_agent_ratio': 5}, 'ms_agent_ratio': 2})
    assert sim.diseases.hpv16.pars.ms_agent_ratio == 2
    assert sim.diseases.hpv18.pars.ms_agent_ratio == 5

    # Nested genotype-scoped override preserves sibling keys (Task 1's fix).
    hpv.route_pars(sim, pars={'hpv16': {'cin_fn': {'k': 0.55}}})
    assert sim.diseases.hpv16.pars.cin_fn['k'] == 0.55
    assert sim.diseases.hpv16.pars.cin_fn['form'] == 'logf2'

    # Cross-immunity and network scalar pars route to their module.
    hpv.route_pars(sim, pars={'cross_immunity': {'own_imm_hr': 0.5}})
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv.CrossImmunity)][0]
    assert conn.pars.own_imm_hr == 0.5

    hpv.route_pars(sim, pars={'network': {'m_cross_layer': 0.5}})
    assert sim.networks.sexualnetwork.pars.m_cross_layer == 0.5


def test_par_routing_calib_pars_flat_dotted():
    """calib_pars (flat dotted) produces the same result as the nested pars= form."""
    sim = make_sim(genotypes=[16])
    sim.init()
    hpv.route_pars(sim, calib_pars={'hpv16.cin_fn.k': 0.77})
    assert sim.diseases.hpv16.pars.cin_fn['k'] == 0.77


def test_par_routing_raises_on_unknown_key():
    sim = make_sim(genotypes=[16])
    sim.init()
    with pytest.raises(ValueError):
        hpv.route_pars(sim, pars={'not_a_real_par': 1})


def test_beta_scalar_expansion():
    """A scalar beta expands via rel_beta/transf2m/transm2f identically whether
    set via genotype_pars= (bare HPV kwargs) or a nested pars= override; an
    explicit beta dict bypasses the expansion."""
    from hpvsim.parameters import get_genotype_pars
    g = get_genotype_pars('hpv16')

    a = make_sim(genotypes=[16], genotype_pars={'hpv16': dict(beta=0.2, rel_beta=1.5)})
    b = make_sim(genotypes=[16], pars=dict(hpv16=dict(beta=0.2, rel_beta=1.5)))
    a.init(); b.init()
    betamap_a = a.diseases.hpv16.validate_beta()
    betamap_b = b.diseases.hpv16.validate_beta()
    assert betamap_a == betamap_b
    assert betamap_a['sexualnetwork'][0] == pytest.approx(0.2 * 1.5 * g.transf2m)
    assert betamap_a['sexualnetwork'][1] == pytest.approx(0.2 * 1.5 * g.transm2f)

    explicit = hpv.HPV(genotype='hpv16', beta={'sexualnetwork': [0.11, 0.22]})
    assert explicit.pars.beta == {'sexualnetwork': [0.11, 0.22]}


def test_construction():
    """nw_pars=/imm_pars= apply to the auto-constructed network/connector;
    each raises when the caller also supplies their own instance."""
    sim = make_sim(genotypes=[16], nw_pars=dict(m_cross_layer=0.3), imm_pars=dict(own_imm_hr=0.5))
    sim.init()
    assert sim.networks.sexualnetwork.pars.m_cross_layer == 0.3
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv.CrossImmunity)][0]
    assert conn.pars.own_imm_hr == 0.5

    with pytest.raises(ValueError, match='nw_pars'):
        make_sim(genotypes=[16], networks=[hpv.SexualNetwork()], nw_pars=dict(m_cross_layer=0.3))
    with pytest.raises(ValueError, match='imm_pars'):
        make_sim(genotypes=[16], connectors=[hpv.CrossImmunity()], imm_pars=dict(own_imm_hr=0.5))


def test_sim_construction():
    """Bare kwargs and pars= are equivalent for sim-level, HPV-scalar, and
    nested genotype-scoped overrides; an unrecognized bare key raises."""
    a = make_sim(rand_seed=7)
    b = make_sim(pars=dict(rand_seed=7))
    assert a.pars.rand_seed == b.pars.rand_seed == 7

    sim = make_sim(genotypes=[16], beta=0.15)
    sim.init()
    assert sim.diseases.hpv16.pars.beta == 0.15

    a = make_sim(genotypes=[16, 18], hpv16=dict(cin_fn=dict(k=0.55)))
    b = make_sim(genotypes=[16, 18], pars=dict(hpv16=dict(cin_fn=dict(k=0.55))))
    a.init(); b.init()
    assert a.diseases.hpv16.pars.cin_fn['k'] == b.diseases.hpv16.pars.cin_fn['k'] == 0.55
    assert a.diseases.hpv18.pars.cin_fn['k'] != 0.55

    with pytest.raises(ValueError):
        make_sim(genotypes=[16], not_a_real_par=1)


def test_hiv_par_routing():
    """pars=dict(hiv=dict(...)) routes into an already-constructed HIV
    disease; hiv_pars= (construction-time) and pars=dict(hiv=...)
    (post-construction) compose."""
    sim = hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0,
                  genotypes=[16], model_hiv='transmission',
                  hiv_pars=dict(rel_sus_lo=0.4), pars=dict(hiv=dict(rel_sus_hi=0.5)))
    sim.init()
    assert sim.diseases.hiv.pars.rel_sus_lo == 0.4
    assert sim.diseases.hiv.pars.rel_sus_hi == 0.5


def test_bare_beta_does_not_broadcast_to_hiv():
    """A bare beta= override must not leak into the auto-constructed HIV
    disease. beta/init_prev are inherited stisim BaseSTIPars keys shared
    with HPV, but 'hiv' is intentionally excluded from route_pars' bare-key
    broadcast registry (par_registry) -- only the scoped hiv=/hiv_pars=
    forms may touch HIV's pars."""
    from hpvsim.hiv import HIV
    default_hiv_beta = HIV().pars.beta
    sim = make_sim(genotypes=[16], model_hiv='transmission', pars=dict(beta=0.2))
    sim.init()
    assert sim.diseases.hiv.pars.beta == default_hiv_beta
    assert sim.diseases.hpv16.pars.beta == 0.2


def test_pars_accepts_flat_dotted_keys():
    """pars= (not just calib_pars=) accepts a flat dotted key -- a bare
    dotted key via pars= must route the same as the equivalent nested form."""
    sim = make_sim(genotypes=[16], pars={'hpv16.cin_fn.k': 0.5})
    sim.init()
    assert sim.diseases.hpv16.pars.cin_fn['k'] == 0.5


def test_pars_location_raises_value_error():
    """pars=dict(location=...) cannot be forwarded to super().__init__
    (vanilla ss.SimPars lacks 'location'), so it must fail with route_pars'
    clear ValueError, not sciris' KeyNotFoundError from deep inside
    ss.Sim.__init__. Bare location= (not via pars=) is unaffected -- see
    test_sim_construction / test_hiv_par_routing for that path."""
    with pytest.raises(ValueError, match='location'):
        make_sim(genotypes=[16], pars=dict(location='nigeria'))


def test_copy_inputs_kwarg_accepted():
    """copy_inputs=/data= are genuine ss.Sim.__init__ kwargs (defaults
    True/None) that must still construct without error now that
    unrecognized bare kwargs route through mod_pars/route_pars instead of
    flowing straight through to super().__init__."""
    make_sim(genotypes=[16], copy_inputs=False)


def test_validate_beta_zeroes_non_sexual_networks():
    """A scalar beta must not leak a nonzero symmetric value onto a
    non-sexual network sharing the sim -- ss.Infection.validate_beta's
    scalar branch expands to every network by default; HPV.validate_beta
    must zero out anything that isn't the sexual network."""
    sim = make_sim(genotypes=[16], networks=[hpv.SexualNetwork(), ss.MaternalNet()],
                   beta=0.2)
    sim.init()
    betamap = sim.diseases.hpv16.validate_beta()
    assert betamap['maternal'] == [0.0, 0.0]
    assert betamap['sexualnetwork'] != [0.0, 0.0]


def test_sim_key_source_of_truth_consistent():
    """route_pars' bare-key 'sim' category and hpv.Sim.__init__'s own
    sim/mod split must agree on what counts as a sim-level key -- calling
    route_pars directly (as calibration does) and constructing via
    pars= must both accept the same sim-level keys."""
    sim = make_sim(genotypes=[16])
    sim.init()
    hpv.route_pars(sim, pars={'pop_scale': 2})
    assert sim.pars.pop_scale == 2

    sim2 = make_sim(genotypes=[16], pars=dict(pop_scale=2))
    sim2.init()
    assert sim2.pars.pop_scale == 2
