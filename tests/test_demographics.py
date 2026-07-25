"""Tests for hpv.AgeMigration."""
import numpy as np

import hpvsim as hpv


def test_age_migration_class_exists():
    import starsim as ss
    assert hasattr(hpv, 'AgeMigration')
    assert issubclass(hpv.AgeMigration, ss.Demographics)


def test_load_country_exposes_pop_total_and_pop_by_age():
    data = hpv.data.load_country('nigeria')
    assert 'pop_total' in data
    assert 'pop_by_age' in data
    pt = data['pop_total']
    assert {'year', 'pop_size'}.issubset(pt.columns)
    pat = data['pop_by_age']
    assert {'year', 'age', 'male', 'female'}.issubset(pat.columns)


def test_age_migration_runs_without_error():
    sim = hpv.Sim(
        n_agents=500, location='nigeria',
        start=1990, stop=2000, dt=1.0, rand_seed=0,
        demographics=[hpv.AgeMigration()],
    )
    sim.run()
    assert int(sim.results['n_alive'][-1]) > 0


def test_age_migration_pulls_pyramid_toward_target():
    """With AgeMigration on, end-of-sim age pyramid tracks pop_by_age.

    Compare a sim with AgeMigration to one without; with on, the per-age-bin
    population matches the year-X target distribution more tightly than off.
    """
    pars = dict(n_agents=2000, location='nigeria',
                start=1990, stop=2020, dt=1.0, rand_seed=0)

    sim_with = hpv.Sim(**pars, demographics=[hpv.AgeMigration()])
    sim_with.run()

    sim_off = hpv.Sim(**pars, demographics=[])
    sim_off.run()

    target_pat = hpv.data.load_country('nigeria')['pop_by_age']
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


def _module(sim, name):
    return [m for m in sim.modules if getattr(m, 'name', '') == name][0]


def test_demographic_data_trimmed_to_sim_window():
    """Demographic modules retain only data within the sim window.

    Guards the footprint optimization: the raw country tables span ~1950-2101,
    but AgeMigration and ss.Deaths only ever query years in [start, stop]. The
    raw pop_by_age table is released after building the per-year lookup, that
    lookup is trimmed to the window, and death_rate is trimmed before it
    reaches ss.Deaths. Regressing any of these reinflates every save file.
    """
    start, stop = 1990, 2010
    sim = hpv.Sim(n_agents=300, location='nigeria', genotypes=[16],
                  start=start, stop=stop, dt=0.25, rand_seed=0, verbose=0)
    sim.run()

    am = _module(sim, 'agemigration')
    assert am._pop_by_age is None, 'raw pop_by_age should be released after init_pre'
    yrs = sorted(am._pop_by_year)
    assert yrs[0] >= start - 1 and yrs[-1] <= stop + 1, \
        f'_pop_by_year spans {yrs[0]}-{yrs[-1]}, expected within {start-1}-{stop+1}'
    assert len(yrs) <= (stop - start) + 3, f'_pop_by_year kept {len(yrs)} years (untrimmed?)'

    dr = _module(sim, 'deaths').pars['death_rate']
    assert dr['Time'].min() >= start - 1 and dr['Time'].max() <= stop + 1, \
        f"death_rate Time spans {dr['Time'].min()}-{dr['Time'].max()}, expected within window"