"""Tests for HPV genotype-parameter overrides.

1. ``rel_beta`` overrides must feed into the directional transmission ``beta`` dict.
2. Unit-less duration distributions must raise: a duration without ``ss.years``
   is read in timesteps and silently collapses the epidemic.
"""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv
from hpvsim.parameters import get_genotype_pars


def _beta(mod):
    return mod.validate_beta()['sexualnetwork']


def test_default_beta_matches_formula():
    """Default beta = base * rel_beta * transf2m/transm2f."""
    g = get_genotype_pars('hpv16')
    sim = hpv.Sim(genotypes=[16], n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    sim.init()
    b = _beta(sim.diseases.hpv16)
    assert b[0] == pytest.approx(g.beta * g.rel_beta * g.transf2m)
    assert b[1] == pytest.approx(g.beta * g.rel_beta * g.transm2f)


def test_rel_beta_override_scales_beta():
    """A rel_beta override must scale the transmission beta dict (bug 1)."""
    g = get_genotype_pars('hpv16')
    base_sim = hpv.Sim(genotypes=[16], n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0)
    hi_sim = hpv.Sim(genotypes=[16], n_agents=200, start=2019, stop=2020, dt=1.0, rand_seed=0,
                     genotype_pars={'hpv16': {'rel_beta': 2.0}})
    base_sim.init(); hi_sim.init()
    base, hi = base_sim.diseases.hpv16, hi_sim.diseases.hpv16
    for i in (0, 1):
        assert _beta(hi)[i] == pytest.approx(2.0 * _beta(base)[i])
    assert _beta(hi)[0] == pytest.approx(g.beta * 2.0 * g.transf2m)
    assert hi.pars.rel_beta == 2.0


def test_rel_beta_override_via_sim_genotype_pars():
    """rel_beta override flows through hpv.Sim(genotype_pars=...).

    Use rel_beta<1: keeps directional beta below the per-act probability
    ceiling of 1.0 (default m->f beta = 0.25 * transm2f = 0.5 at rel_beta=1.0).
    """
    sim = hpv.Sim(genotypes=[16], genotype_pars={'hpv16': {'rel_beta': 0.5}},
                  n_agents=200, start=2000, stop=2001, dt=1.0)
    sim.init()  # sim.diseases is populated at init
    g = get_genotype_pars('hpv16')
    b = _beta(sim.diseases.hpv16)
    assert b[0] == pytest.approx(g.beta * 0.5 * g.transf2m)
    assert b[1] == pytest.approx(g.beta * 0.5 * g.transm2f)


def test_explicit_beta_override_is_preserved():
    """An explicit beta dict must NOT be recomputed away."""
    mod = hpv.HPV(genotype='hpv16', beta={'sexualnetwork': [0.11, 0.22]})
    assert _beta(mod) == [0.11, 0.22]


def test_higher_rel_beta_raises_prevalence():
    """Behavioral check: higher rel_beta -> more transmission.

    Both values keep the directional beta below the per-act probability ceiling
    of 1.0 (default m->f beta = 0.25 * transm2f = 0.5 at rel_beta=1.0).
    """
    def prev(rb):
        sim = hpv.Sim(genotypes=[16], genotype_pars={'hpv16': {'rel_beta': rb}},
                      n_agents=2000, start=1990, stop=2010, dt=0.5, rand_seed=1)
        sim.run()
        return float(sim.results.all_hpv.cum_infections[-1])
    assert prev(1.0) > prev(0.3)


def test_unitless_duration_raises():
    """A duration distribution without ss.years must raise (bug 2)."""
    with pytest.raises(ValueError, match='without time units'):
        hpv.HPV(genotype='hpv16', dur_precin=ss.lognorm_ex(mean=3, std=9))


def test_unitless_duration_raises_via_sim():
    with pytest.raises(ValueError, match='without time units'):
        hpv.Sim(genotypes=[16],
                genotype_pars={'hpv16': {'dur_cin': ss.lognorm_ex(mean=5, std=20)}},
                n_agents=200, start=2000, stop=2001)


def test_years_wrapped_duration_ok():
    """A properly unit-wrapped duration override must be accepted."""
    mod = hpv.HPV(genotype='hpv16',
                  dur_precin=ss.lognorm_ex(mean=ss.years(3), std=ss.years(9)))
    assert isinstance(mod.pars.dur_precin, ss.Dist)


def test_partial_years_duration_ok():
    """lognorm_ex with only mean wrapped (test_hpv.py pattern) is accepted."""
    mod = hpv.HPV(genotype='hpv16', dur_precin=ss.lognorm_ex(mean=ss.years(4)))
    assert isinstance(mod.pars.dur_precin, ss.Dist)
