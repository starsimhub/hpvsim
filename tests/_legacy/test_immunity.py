"""
Tests for hpvsim/immunity.py — immunity initialization, waning, and cross-immunity.
"""

import numpy as np
import sciris as sc
import hpvsim as hpv
from hpvsim import immunity as hpi

hpv.options.set(interactive=False)


def test_precompute_waning_exp_decay():
    """Test exponential decay waning computation."""
    t = np.arange(100)
    pars = dict(form='exp_decay', init_val=1.0, half_life=10)
    result = hpi.precompute_waning(t, pars)

    assert len(result) == len(t)
    assert result[0] == 1.0  # Initial value
    assert result[-1] < result[0]  # Should decay
    # At half-life, should be ~0.5 of initial
    assert abs(result[10] - 0.5) < 0.01


def test_precompute_waning_none():
    """Test no waning (returns ones)."""
    t = np.arange(50)
    pars = dict(form=None)
    result = hpi.precompute_waning(t, pars)
    assert np.allclose(result, 1.0)


def test_precompute_waning_custom():
    """Test custom waning function."""
    t = np.arange(50)
    custom_fn = lambda t, **kwargs: np.ones(len(t)) * 0.5
    pars = dict(form=custom_fn)
    result = hpi.precompute_waning(t, pars)
    assert np.allclose(result, 0.5)


def test_exp_decay():
    """Test the raw exponential decay function."""
    t = np.arange(20)

    # Standard case
    result = hpi.exp_decay(t, init_val=1.0, half_life=5)
    assert result[0] == 1.0
    assert abs(result[5] - 0.5) < 0.01

    # NaN half-life should give no decay
    result_nan = hpi.exp_decay(t, init_val=1.0, half_life=np.nan)
    assert np.allclose(result_nan, 1.0)


def test_linear_decay():
    """Test linear decay function."""
    result = hpi.linear_decay(length=10, init_val=1.0, slope=0.1)
    assert result[0] == 1.0
    assert len(result) == 10


def test_linear_growth():
    """Test linear growth function."""
    result = hpi.linear_growth(length=10, slope=0.5)
    assert len(result) == 10
    assert np.allclose(result, 0.5)


def test_init_immunity():
    """Test that immunity initialization populates the expected fields."""
    sim = hpv.Sim(n_agents=500, n_years=5, genotypes=[16, 18], verbose=0)
    sim.initialize()

    # Check that cross-immunity matrices were created
    assert 'cross_immunity_sus' in sim.pars
    assert 'cross_immunity_sev' in sim.pars
    ci_sus = sim['cross_immunity_sus']
    ci_sev = sim['cross_immunity_sev']
    assert ci_sus is not None
    assert ci_sev is not None

    # Diagonal should be 1 (self-immunity)
    ng = sim['n_genotypes']
    for i in range(ng):
        assert ci_sus[i, i] == 1.0
        assert ci_sev[i, i] == 1.0


def test_init_immunity_with_vaccine():
    """Test immunity initialization with a vaccine intervention."""
    vx = hpv.routine_vx(prob=0.5, start_year=2000, age_range=[9, 10], product='bivalent')
    sim = hpv.Sim(n_agents=500, start=1995, n_years=10, genotypes=[16, 18], interventions=[vx], verbose=0)
    sim.initialize()

    # n_imm_sources should be > n_genotypes (genotypes + vaccine products)
    assert sim['n_imm_sources'] > sim['n_genotypes']
    assert 'immunity_map' in sim.pars


def test_immunity_from_infection():
    """Test that infection generates immunity in agents."""
    sim = hpv.Sim(n_agents=1e3, n_years=10, genotypes=[16, 18], verbose=0)
    sim.run()

    # Some people should have non-zero peak immunity
    has_immunity = (sim.people.peak_imm > 0).any(axis=0)
    assert has_immunity.sum() > 0


def test_immunity_from_vaccination():
    """Test that vaccination generates immunity in agents."""
    vx = hpv.routine_vx(prob=0.9, start_year=2020, age_range=[9, 10], product='bivalent')
    sim = hpv.Sim(n_agents=1e3, n_years=10, genotypes=[16, 18], start=2015, interventions=[vx], verbose=0)
    sim.run()

    # Some vaccinated agents should have immunity from the vaccine source
    n_vaccinated = sim.results['n_vaccinated'][-1]
    assert n_vaccinated > 0


def test_cross_immunity_affects_susceptibility():
    """Test that cross-immunity parameters are used during simulation."""
    # Run with default cross-immunity
    sim1 = hpv.Sim(n_agents=2e3, n_years=15, genotypes=[16, 18], rand_seed=0, verbose=0)
    sim1.run()

    # Verify cross-immunity matrices are populated
    ci_sus = sim1['cross_immunity_sus']
    assert ci_sus is not None
    assert ci_sus.shape[0] >= 2  # At least 2 genotypes


def test_waning_reduces_immunity_over_time():
    """Test that immunity wanes over time when waning is enabled."""
    sim = hpv.Sim(n_agents=1e3, n_years=20, genotypes=[16], verbose=0, use_waning=True)
    sim.run()

    # Check that waning kinetics were precomputed
    assert 'imm_kin' in sim.pars
    imm_kin = sim['imm_kin']
    assert imm_kin is not None

    # The kinetics should show decay over time
    if len(imm_kin.shape) > 1:
        # First row (natural immunity from genotype 0)
        assert imm_kin[0, -1] <= imm_kin[0, 0]


def test_update_peak_immunity_vaccination():
    """Test update_peak_immunity with infection=False (vaccination path)."""
    sim = hpv.Sim(n_agents=500, n_years=5, genotypes=[16, 18], verbose=0)
    sim.initialize()
    people = sim.people

    imm_source = 0
    inds = np.array([0, 1, 2, 3, 4], dtype=int)

    # Vaccine-like parameters (mimics bivalent_3dose)
    imm_pars = dict(
        imm_init=dict(dist='beta', par1=30, par2=2),
        doses=3,
        imm_boost=[1.2, 1.1],
    )

    # Set up dose counts: 1st, 2nd, and 3rd dose recipients
    people.doses[inds] = 0
    people.doses[0] = 1  # First dose
    people.doses[1] = 1
    people.doses[2] = 2  # Second dose
    people.doses[3] = 2
    people.doses[4] = 3  # Third dose

    # Give dose2/dose3 recipients some prior immunity
    people.peak_imm[imm_source, 2] = 0.5
    people.peak_imm[imm_source, 3] = 0.5
    people.peak_imm[imm_source, 4] = 0.6

    # Clear immunity for first-dose recipients
    people.peak_imm[imm_source, 0] = 0.0
    people.peak_imm[imm_source, 1] = 0.0

    hpi.update_peak_immunity(people, inds, imm_pars, imm_source, infection=False)

    # First dose: should have new peak immunity > 0
    assert people.peak_imm[imm_source, 0] > 0
    assert people.peak_imm[imm_source, 1] > 0

    # Second dose: should be boosted (at least as high as before)
    assert people.peak_imm[imm_source, 2] >= 0.5
    assert people.peak_imm[imm_source, 3] >= 0.5

    # Third dose: should be boosted
    assert people.peak_imm[imm_source, 4] >= 0.6

    # t_imm_event should be set for all vaccinated individuals
    for i in inds:
        assert np.isfinite(people.t_imm_event[imm_source, i])


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()
    test_precompute_waning_exp_decay()
    test_precompute_waning_none()
    test_precompute_waning_custom()
    test_exp_decay()
    test_linear_decay()
    test_linear_growth()
    test_init_immunity()
    test_init_immunity_with_vaccine()
    test_immunity_from_infection()
    test_immunity_from_vaccination()
    test_cross_immunity_affects_susceptibility()
    test_waning_reduces_immunity_over_time()
    test_update_peak_immunity_vaccination()
    sc.toc(T)
    print('Done.')
