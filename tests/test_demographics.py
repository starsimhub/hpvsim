"""Tests for hpv.AgeMigration."""
import numpy as np
import pandas as pd
import pytest

import hpvsim as hpv


def test_age_migration_class_exists():
    """AgeMigration is exported from hpvsim and inherits ss.Demographics."""
    import starsim as ss
    assert hasattr(hpv, 'AgeMigration')
    assert issubclass(hpv.AgeMigration, ss.Demographics)


def test_load_country_exposes_pop_trend_and_pop_age_trend():
    """load_country('nigeria') returns the migration data tables."""
    data = hpv.data.load_country('nigeria')
    assert 'pop_trend' in data
    assert 'pop_age_trend' in data
    pt = data['pop_trend']
    assert {'year', 'pop_size'}.issubset(pt.columns)
    pat = data['pop_age_trend']
    assert {'year', 'age', 'male', 'female'}.issubset(pat.columns)


def test_age_migration_runs_without_error():
    """A sim with AgeMigration runs to completion."""
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=2000, dt=1.0, rand_seed=0,
        demographics=[hpv.AgeMigration()],
    )
    sim.run()
    assert int(sim.results['n_alive'][-1]) > 0


def test_age_migration_pulls_pyramid_toward_target():
    """With AgeMigration on, end-of-sim age pyramid tracks pop_age_trend.

    Compare a sim with AgeMigration to one without; with on, the per-age-bin
    population matches the year-X target distribution more tightly than off.
    """
    pars = dict(n_agents=2000, location='nigeria',
                start=1990, stop=2020, dt=1.0, rand_seed=0)

    sim_with = hpv.Sim(**pars, demographics=[hpv.AgeMigration()])
    sim_with.run()

    sim_off = hpv.Sim(**pars, demographics=[])
    sim_off.run()

    target_pat = hpv.data.load_country('nigeria')['pop_age_trend']
    target_2019 = target_pat[target_pat['year'] == 2019]
    target_dist = (target_2019['male'] + target_2019['female']).values
    target_dist = target_dist / target_dist.sum()

    def normalized_age_pyramid(sim):
        ages = np.asarray(sim.people.age.values[sim.people.alive.values]).astype(int)
        bins = np.arange(0, max(int(target_2019['age'].max()) + 2, 101))
        counts, _ = np.histogram(ages, bins=bins)
        counts = counts[:len(target_dist)]
        return counts / counts.sum() if counts.sum() > 0 else counts

    p_with = normalized_age_pyramid(sim_with)
    p_off = normalized_age_pyramid(sim_off)
    tvd_with = 0.5 * np.abs(p_with - target_dist[:len(p_with)]).sum()
    tvd_off  = 0.5 * np.abs(p_off  - target_dist[:len(p_off)]).sum()
    assert tvd_with < tvd_off, f'AgeMigration off→TVD {tvd_off:.3f}, on→TVD {tvd_with:.3f}'