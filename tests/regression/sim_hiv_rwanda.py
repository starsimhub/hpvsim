"""Rwanda HIV–HPV co-infection sim builder + a β trial harness (M08 T12 prep).

``build_rwanda_sim`` is the reusable Rwanda co-infection sim (4-genotype HPV +
transmission-based ``hpv.HIV`` + the coverage-based ``hpv.hiv_art`` shortcut),
parameterized by the HIV ``beta_m2f`` that T12 calibrates. Run this module
directly for a single trial run that prints the modeled HIV-prevalence
trajectory against the Rwanda data target — used to bracket β before a full
calibration.
"""

import numpy as np
import pandas as pd

import hpvsim as hpv


def build_rwanda_sim(beta_m2f=0.004, seed=0, n_agents=5000,
                     start=1985, stop=2020, dt=0.25):
    """Rwanda HIV–HPV co-infection sim, β as the tunable HIV transmission rate."""
    return hpv.Sim(
        location='rwanda',
        rand_seed=seed,
        genotypes=[16, 18, 'hi5', 'ohr'],
        n_agents=n_agents,
        start=start,
        stop=stop,
        dt=dt,
        diseases=[hpv.HIV.from_location('rwanda', beta_m2f=beta_m2f)],
        interventions=[hpv.hiv_art.from_location('rwanda')],
    )


def build_rwanda_sim_incidence(seed=0, n_agents=5000, start=1985, stop=2020, dt=0.25):
    """Rwanda HIV–HPV co-infection sim, incidence-driven (v2-faithful).

    HIV transmission is OFF (``beta_m2f=0``) and there is no seeding
    (``init_prev_data=0``); the epidemic is built entirely by imposing the
    Rwanda HIV incidence curve via ``hpv.hiv_incidence_import``, so the
    prevalence trajectory tracks the target by construction. The coverage-based
    ``hpv.hiv_art`` shortcut still supplies ART/CD4 reconstitution.
    """
    return hpv.Sim(
        location='rwanda',
        rand_seed=seed,
        genotypes=[16, 18, 'hi5', 'ohr'],
        n_agents=n_agents,
        start=start,
        stop=stop,
        dt=dt,
        diseases=[hpv.HIV.from_location('rwanda', beta_m2f=0.0, init_prev_data=0.0)],
        interventions=[
            hpv.hiv_incidence_import.from_location('rwanda'),
            hpv.hiv_art.from_location('rwanda'),
        ],
    )


def _modeled_hiv_prevalence(sim):
    """Return {year: HIV prevalence among the living adult (15-49) population}."""
    hivmod = sim.diseases.hiv
    prev = np.asarray(hivmod.results['prevalence'])
    # Align to the HIV module's OWN time axis (it may differ in length from
    # sim.results.timevec); fall back to a linear map if no timevec is exposed.
    tv = getattr(hivmod.results, 'timevec', None)
    if tv is None or len(tv) != len(prev):
        tv = np.linspace(float(sim.t.start), float(sim.t.stop), len(prev))
    years = np.floor(np.asarray(tv, dtype=float)).astype(int)
    out = {}
    for y in np.unique(years):
        out[int(y)] = float(np.nanmean(prev[years == y]))
    return out


def _target_hiv_prevalence():
    """Rwanda data-target aggregate HIV prevalence by year (from the package)."""
    data = hpv.data.load_hiv('rwanda')  # init_prev only; read the series file directly
    from pathlib import Path
    import hpvsim
    f = Path(hpvsim.__file__).parent / 'data' / 'hiv' / 'rwanda' / 'hiv_prevalence.csv'
    df = pd.read_csv(f)
    return {int(r.year): float(r.total) for r in df.itertuples()}


def _print_trial(label, sim, target):
    modeled = _modeled_hiv_prevalence(sim)
    print(f'=== {label} ===')
    print(f'{"year":>6} {"modeled":>10} {"target":>10}')
    for y in range(1990, 2021, 5):
        m = modeled.get(y, float('nan'))
        t = target.get(y, float('nan'))
        print(f'{y:>6} {m:>10.4f} {t:>10.4f}')
    if modeled:
        mpk_y = max(modeled, key=modeled.get)
        print(f'modeled peak: {modeled[mpk_y]:.4f} at {mpk_y}')
    tpk_y = max(target, key=target.get)
    print(f'target  peak: {target[tpk_y]:.4f} at {tpk_y}')
    print()


if __name__ == '__main__':
    target = _target_hiv_prevalence()

    # Incidence-driven (v2-faithful) trial: the headline build.
    sim_inc = build_rwanda_sim_incidence(n_agents=5000, start=1985, stop=2020)
    sim_inc.run()
    _print_trial('Rwanda HIV trial: incidence-driven importer (beta=0)',
                 sim_inc, target)

    # Transmission-based trial (beta=0.004) for comparison.
    BETA = 0.004
    sim_beta = build_rwanda_sim(beta_m2f=BETA, n_agents=5000, start=1985, stop=2020)
    sim_beta.run()
    _print_trial(f'Rwanda HIV trial: transmission-based (beta_m2f={BETA})',
                 sim_beta, target)
