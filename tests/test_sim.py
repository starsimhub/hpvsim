"""Integration tests for hpvsim.sim.Sim."""

import warnings

import pytest
import starsim as ss

import hpvsim as hpv
from hpvsim.sim import Sim
from hpvsim.hpv import HPV
from hpvsim.network import SexualNetwork


def _tiny(**kw):
    return dict(n_agents=200, start=2000, stop=2002, dt=0.25, location='nigeria', **kw)


def test_sim_constructs_with_defaults():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.25)
    assert sim is not None


def test_sim_init_runs():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.25)
    sim.init()
    assert len(sim.people) == 500


def test_sim_has_one_multilayer_sexual_network():
    """Default config produces a single hpv.SexualNetwork holding both
    partnership layers (m, c).
    """
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2002, dt=0.25)
    sim.init()
    sx = [n for n in sim.networks() if isinstance(n, SexualNetwork)]
    assert len(sx) == 1
    assert set(sx[0].layers) == {'m', 'c'}


def test_sim_has_one_hpv_disease():
    sim = Sim(location='nigeria', genotypes=['hpv16'], n_agents=500,
              start=2000, stop=2002, dt=0.25)
    sim.init()
    hpv_diseases = [d for d in sim.diseases() if isinstance(d, HPV)]
    assert len(hpv_diseases) == 1
    assert hpv_diseases[0].genotype == 'hpv16'


def test_sim_runs_short_window():
    sim = Sim(location='nigeria', n_agents=500, start=2000, stop=2003, dt=0.25)
    sim.run()
    assert sim.results.hpv16.n_infected[-1] >= 0


def test_sim_pop_scale_computed_from_total_pop():
    """If total_pop is set, pop_scale = total_pop / n_agents."""
    import hpvsim as hpv
    sim = hpv.Sim(n_agents=10_000, total_pop=2_000_000,
                  start=1990, stop=1991, dt=1.0, rand_seed=0)
    sim.init()
    assert sim.pars.pop_scale == 200.0


def test_sim_pop_scale_default_one_when_no_location():
    """Bare hpv.Sim() (no location) keeps pop_scale=1 (stisim parity)."""
    import hpvsim as hpv
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = hpv.Sim(n_agents=1000, start=1990, stop=1991, dt=1.0, rand_seed=0)
        sim.init()
    assert sim.pars.pop_scale == 1.0
    assert sim.location is None


def test_sim_auto_populates_total_pop_from_location():
    """When location is supplied without total_pop, total_pop is the sum of
    the UN WPP per-age counts at the start year (stisim pattern)."""
    import hpvsim as hpv
    sim = hpv.Sim(location='kazakhstan', n_agents=1000, start=1960,
                  stop=1961, dt=1.0, rand_seed=0)
    sim.init()
    # Kazakhstan 1960 UN WPP total ≈ 9.28M; pop_scale ≈ 9275.
    assert sim.pars.total_pop > 9_000_000
    assert sim.pars.total_pop < 10_000_000
    assert sim.pars.pop_scale > 9_000


def test_sim_explicit_total_pop_wins_over_auto_populate():
    """Caller-supplied total_pop wins over the location auto-populate."""
    import hpvsim as hpv
    sim = hpv.Sim(location='kazakhstan', n_agents=1000, total_pop=500_000,
                  start=1960, stop=1961, dt=1.0, rand_seed=0)
    sim.init()
    assert sim.pars.total_pop == 500_000
    assert sim.pars.pop_scale == 500.0


def test_bare_sim_uniform_ages_no_vitals_default_network():
    """No-location Sim: uniform ages (starsim default), no births/deaths/
    migration modules, but the default SexualNetwork is still wired up."""
    import hpvsim as hpv
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = hpv.Sim(n_agents=500, start=2000, stop=2002, dt=1.0, rand_seed=0)
        sim.init()
    # No vitals modules.
    assert len(list(sim.demographics.values())) == 0
    # But sexual network is present.
    sx = [n for n in sim.networks() if isinstance(n, SexualNetwork)]
    assert len(sx) == 1
    # Location tracked as None on the sim.
    assert sim.location is None


def test_bare_sim_emits_config_warning():
    """Bare Sim emits a warning describing the auto-configuration."""
    import hpvsim as hpv
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hpv.Sim(n_agents=100, start=2000, stop=2001, dt=1.0, rand_seed=0)
    msgs = [str(w.message) for w in caught]
    assert any('no location supplied' in m and 'uniform ages' in m for m in msgs), \
        f'expected bare-Sim config warning; got {msgs}'


def test_hpv_demo_returns_nigeria_sim():
    """hpv.demo() with no args returns a fully configured Nigeria sim."""
    import hpvsim as hpv
    sim = hpv.demo(run=False, plot=False)
    sim.init()
    assert sim.location == 'nigeria'
    # Auto-populated total_pop and pop_scale for Nigeria.
    assert sim.pars.total_pop > 10_000_000
    assert sim.pars.pop_scale > 100
    # Demographics wired up.
    dem_names = [type(d).__name__ for d in sim.demographics.values()]
    assert any('irth' in n for n in dem_names), dem_names  # Births or AnnualBirths
    assert 'Deaths' in dem_names, dem_names
    # Sexual network wired up.
    assert any(isinstance(n, SexualNetwork) for n in sim.networks())


def test_hpv_demo_unknown_example_raises():
    """hpv.demo('atlantis') raises with a clear error."""
    import hpvsim as hpv
    try:
        hpv.demo('atlantis', run=False, plot=False)
    except ValueError as e:
        assert 'atlantis' in str(e)
        return
    raise AssertionError('expected ValueError for unknown example')


def test_datafolder_override_uses_user_csv(tmp_path):
    """A user CSV in datafolder overrides the bundled UN WPP data."""
    import hpvsim as hpv
    import pandas as pd
    # Minimal fake age_data with a distinct signature: single 30-year-old cohort.
    df = pd.DataFrame({
        'Time': [2000] * 3,
        'AgeGrpStart': [29, 30, 31],
        'PopTotal': [0.0, 500.0, 0.0],  # PopTotal is in thousands
    })
    (tmp_path / 'age_data.csv').write_text(df.to_csv(index=False))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # missing birth/death/pop_total CSVs will warn
        # location is still required (used for bundled fallbacks on other
        # indicators + for network defaults); user CSV overrides the age
        # distribution.
        sim = hpv.Sim(location='nigeria', datafolder=str(tmp_path),
                      n_agents=200, start=2000, stop=2001, dt=1.0, rand_seed=0)
        sim.init()
    # total_pop is the sum of PopTotal * 1e3 = 500,000.
    assert sim.pars.total_pop == 500_000


def test_datafolder_missing_indicator_warns(tmp_path):
    """When a datafolder is given but an indicator CSV is missing, warn."""
    import hpvsim as hpv
    # Empty datafolder: every indicator triggers a warning (and falls back
    # to bundled UN WPP for the caller-specified location).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hpv.Sim(location='nigeria', datafolder=str(tmp_path),
                n_agents=100, start=2000, stop=2001, dt=1.0, rand_seed=0)
    msgs = [str(w.message) for w in caught]
    # At least one indicator file should be reported missing.
    assert any('not found in datafolder' in m for m in msgs), (
        f'expected a missing-indicator warning; got {msgs}'
    )


def test_sim_pop_scale_default_one_when_total_pop_none():
    """Legacy name preserved: bare hpv.Sim() keeps pop_scale=1."""
    test_sim_pop_scale_default_one_when_no_location()


def test_end_is_legacy_alias_for_stop():
    """v2 scripts pass ``end=``; it should set ``stop`` and warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim_end = Sim(n_agents=200, start=2000, end=2005, dt=1.0)
    sim_stop = Sim(n_agents=200, start=2000, stop=2005, dt=1.0)
    assert sim_end.pars.stop == sim_stop.pars.stop
    assert any('deprecated alias' in str(w.message) for w in caught), \
        'passing end= should emit a deprecation warning'


def test_stop_without_end_does_not_warn():
    """Using the canonical ``stop=`` must not emit the legacy-alias warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Sim(n_agents=200, start=2000, stop=2005, dt=1.0)
    assert not any('deprecated alias' in str(w.message) for w in caught)


def test_genotypes_plus_other_disease_merges():
    """A non-HPV disease passed via diseases= merges with genotype-built HPV."""
    other = ss.SIS()  # any non-HPV ss.Disease as a stand-in
    sim = hpv.Sim(**_tiny(genotypes=[16, 18], diseases=[other]))
    sim.init()
    hpv_mods = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)]
    assert len(hpv_mods) == 2                      # genotypes still built
    assert any(isinstance(d, ss.SIS) for d in sim.diseases.values())  # other merged in


def test_hpv_instance_override_still_works():
    """diseases=[HPV,...] override path is unchanged (no genotypes=)."""
    sim = hpv.Sim(**_tiny(diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')]))
    sim.init()
    hpv_mods = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)]
    assert len(hpv_mods) == 2


def test_hpv_instances_plus_genotypes_raises():
    """Specifying the HPV set two ways still raises."""
    with pytest.raises(ValueError, match='genotypes='):
        hpv.Sim(**_tiny(genotypes=[16], diseases=[hpv.HPV(genotype='hpv16')]))