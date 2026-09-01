"""Tests for hpv.Sim's parameter-routing surface: the user-facing
equivalence between bare kwargs, pars=, and calib_pars."""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss

import hpvsim as hpv


def make_sim(**kw):
    return hpv.Sim(n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0, **kw)


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

    hpv.route_pars(sim, pars={'network': {'m_cross_layer': 8}})
    assert sim.networks.sexualnetwork.pars.m_cross_layer == 8


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
