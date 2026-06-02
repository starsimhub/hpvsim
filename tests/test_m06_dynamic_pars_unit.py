"""Unit tests for hpv.dynamic_pars — time-varying parameter editor."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv


def _four_genotype_sim_with(intv):
    return hpv.Sim(
        n_agents=300, start=2020, stop=2030, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
        interventions=[intv],
    )


def test_dynamic_pars_linear_interpolation():
    """A linear ramp on hpv16.beta should hit the midpoint at the midpoint year."""
    intv = hpv.dynamic_pars(pars={
        'hpv16.beta': {'years': [2020, 2030], 'vals': [1.0, 0.0]},
    })
    sim = _four_genotype_sim_with(intv)
    sim.init()
    # Step until year 2025 (midpoint)
    while sim.t.now('year') < 2025:
        sim.run_one_step()
    sim.run_one_step()
    # hpv16.beta should be near 0.5
    assert abs(sim.diseases['hpv16'].pars.beta - 0.5) < 0.1


def test_dynamic_pars_stepwise_mode():
    intv = hpv.dynamic_pars(
        pars={'hpv16.beta': {'years': [2020, 2025], 'vals': [1.0, 0.2]}},
        interpolate=False,
    )
    sim = _four_genotype_sim_with(intv)
    sim.init()
    while sim.t.now('year') < 2023:
        sim.run_one_step()
    sim.run_one_step()
    assert sim.diseases['hpv16'].pars.beta == 1.0
    while sim.t.now('year') < 2026:
        sim.run_one_step()
    sim.run_one_step()
    assert sim.diseases['hpv16'].pars.beta == 0.2


def test_dynamic_pars_unresolvable_path_raises_at_step():
    intv = hpv.dynamic_pars(pars={
        'nonexistent.foo': {'years': [2020, 2030], 'vals': [1.0, 0.5]},
    })
    sim = _four_genotype_sim_with(intv)
    sim.init()
    with pytest.raises(KeyError, match='nonexistent'):
        sim.run_one_step()


def test_dynamic_pars_single_segment_sim_pars_path():
    """Single-segment paths address sim.pars directly (e.g. 'rand_seed')."""
    sim = hpv.Sim(
        n_agents=100, start=2020, stop=2021, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16',)],
    )
    sim.init()
    # Setting via dynamic_pars._set_dotted should mutate sim.pars in place.
    hpv.dynamic_pars._set_dotted(sim, 'rand_seed', 42)
    assert sim.pars.rand_seed == 42


def test_dynamic_pars_extrapolation_clamps_to_endpoints():
    """np.interp clamps below first year (returns vals[0]) and above last
    year (returns vals[-1]). This is the documented + intended behavior."""
    intv = hpv.dynamic_pars(pars={
        'hpv16.beta': {'years': [2025, 2026], 'vals': [0.5, 0.7]},
    })
    sim = _four_genotype_sim_with(intv)
    sim.init()
    # Step the sim to t < first schedule year — beta should clamp to vals[0]
    sim.run_one_step()
    assert sim.diseases['hpv16'].pars.beta == 0.5
    # Step past last schedule year — beta should clamp to vals[-1]
    while sim.t.now('year') < 2028:
        sim.run_one_step()
    sim.run_one_step()
    assert sim.diseases['hpv16'].pars.beta == 0.7
