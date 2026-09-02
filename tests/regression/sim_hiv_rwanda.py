"""Rwanda HIV–HPV co-infection sim builder + a β trial harness (M08 T12 prep).

``build_rwanda_sim`` is the reusable Rwanda co-infection sim (4-genotype HPV +
transmission-based ``hpv.HIV_transmit`` + plain ``sti.ART``), parameterized by
the HIV ``beta_m2f`` that T12 calibrates. Run this module directly for a
single trial run that prints the modeled HIV-prevalence trajectory against
the Rwanda data target — used to bracket β before a full calibration.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import stisim as sti

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hpvsim as hpv  # noqa: E402
from tests.regression.rwanda_calib import RWANDA_HIV_DATA, RWANDA_HIV_DATA_DIR  # noqa: E402


def build_rwanda_sim(beta_m2f=0.004, seed=0, n_agents=5000,
                     start=1985, stop=2020, dt=0.25):
    """Rwanda HIV–HPV co-infection sim, β as the tunable HIV transmission rate."""
    art = sti.ART(coverage=hpv.data.reshape_art_coverage(RWANDA_HIV_DATA['art_coverage']))
    return hpv.Sim(
        location='rwanda',
        rand_seed=seed,
        genotypes=[16, 18, 'hi5', 'ohr'],
        n_agents=n_agents,
        start=start,
        stop=stop,
        dt=dt,
        diseases=[hpv.HIV_transmit(beta_m2f=beta_m2f, init_prev_data=RWANDA_HIV_DATA['init_prev'])],
        interventions=[art],
    )


def build_rwanda_sim_incidence(seed=0, n_agents=5000, start=1985, stop=2020, dt=0.25):
    """Rwanda HIV–HPV co-infection sim, incidence-driven (v2-faithful).

    There is no seeding (``init_prev_data=0``); the epidemic is built entirely
    by imposing the Rwanda HIV incidence curve via ``hpv.HIV_incidence``, so
    the prevalence trajectory tracks the target by construction. Plain
    ``sti.ART`` supplies ART/CD4 reconstitution.
    """
    art = sti.ART(coverage=hpv.data.reshape_art_coverage(RWANDA_HIV_DATA['art_coverage']))
    return hpv.Sim(
        location='rwanda',
        rand_seed=seed,
        genotypes=[16, 18, 'hi5', 'ohr'],
        n_agents=n_agents,
        start=start,
        stop=stop,
        dt=dt,
        diseases=[hpv.HIV_incidence(incidence=RWANDA_HIV_DATA['incidence'], init_prev_data=0.0)],
        interventions=[art],
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
    """Rwanda data-target aggregate HIV prevalence by year."""
    df = pd.read_csv(RWANDA_HIV_DATA_DIR / 'hiv_prevalence.csv')
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
