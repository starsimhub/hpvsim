"""Incidence-driven HIV importer tests.

The ``hpv.hiv_incidence_import`` intervention drives STIsim's HIV by imposing
the Rwanda HIV incidence curve: each step it selects HIV-negative susceptibles
per the (year, sex, age) incidence rate and calls ``hiv.set_prognoses`` on them
(which flips them to infected AND wires the full CD4 trajectory). With HIV
``beta_m2f=0`` and ``init_prev_data=0``, the epidemic is built entirely by the
importer, following an incidence-based approach.
"""

import numpy as np
import pytest

import hpvsim as hpv


def _build_sim(seed=0, n_agents=3000, start=1985, stop=2010, dt=0.25):
    return hpv.Sim(
        location='rwanda',
        rand_seed=seed,
        genotypes=[16, 18, 'hi5', 'ohr'],
        n_agents=n_agents,
        start=start,
        stop=stop,
        dt=dt,
        diseases=[hpv.HIV.from_location('rwanda', beta_m2f=0.0, init_prev_data=0.0)],
        interventions=[hpv.hiv_incidence_import.from_location('rwanda')],
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_importer_builds_hiv_epidemic():
    sim = _build_sim()
    sim.run()
    hivmod = sim.diseases.hiv
    prev = np.asarray(hivmod.results['prevalence'])
    tv = np.asarray(sim.results.timevec, dtype=float)
    if len(tv) != len(prev):
        tv = np.linspace(float(sim.t.start), float(sim.t.stop), len(prev))
    years = np.floor(tv).astype(int)

    def prev_at(year):
        m = years == year
        return float(np.nanmean(prev[m])) if m.any() else float('nan')

    # ~0 at the very start (no seeding; importer hasn't acted yet, 1985 inc ~ 0).
    assert prev_at(1985) < 0.005
    # Rises to a nonzero, plausible level (a few percent) by 2000.
    p2000 = prev_at(2000)
    assert p2000 > 0.01, f'HIV prevalence at 2000 too low: {p2000}'
    assert p2000 < 0.20, f'HIV prevalence at 2000 implausibly high: {p2000}'


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_imported_agents_get_cd4_trajectory():
    sim = _build_sim()
    sim.run()
    hivmod = sim.diseases.hiv
    infected = hivmod.infected.uids
    assert len(infected) > 0, 'importer infected nobody'
    cd4 = np.asarray(hivmod.cd4[infected])
    # set_prognoses wired CD4: HIV+ agents have finite CD4 values.
    finite = np.isfinite(cd4)
    assert finite.any(), 'no HIV+ agent has a finite CD4 (set_prognoses not wired)'
    # CD4 has declined below the healthy start (~800) for at least some agents,
    # confirming the falling/AIDS trajectory is progressing over time.
    assert np.nanmin(cd4[finite]) < 700.0
