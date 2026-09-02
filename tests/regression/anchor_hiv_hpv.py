"""Co-infection anchor: 4-genotype HPV + transmission HIV + ART on Nigeria.

Phase-1 anchor uses a hand-tuned HIV beta to produce a non-trivial HIV+
subpopulation; it is NOT calibrated to a country (that is Phase 2 / Rwanda).

API notes (STIsim 1.5.0):
  - ``sti.HIV.init_prev_data`` accepts a scalar float (parsed by
    ``make_init_prev`` -> ``ss.bernoulli`` over everyone). We seed ~2% so the
    epidemic has something to grow from. A DataFrame form (by risk group/sex)
    is also supported but requires a ``structuredsexual`` network, which hpvsim
    does not use, so the scalar form is the right one here.
  - ``sti.ART`` is an INTERVENTION; ``coverage`` accepts a scalar proportion of
    diagnosed-infected to keep on treatment. ART only treats agents that have
    been diagnosed, so without an HIVTest it is effectively a no-op on CD4 — we
    keep it wired in (per the milestone API) but rely on untreated CD4 decline
    to drive HIV+ HPV/cancer outcomes.

Run as a script to print HIV prevalence + HIV+ cancer count:
    python tests/regression/anchor_hiv_hpv.py
"""
import stisim as sti
import hpvsim as hpv

PARS = dict(n_agents=5000, start=1990, stop=2030, dt=0.25, location='nigeria')


def build_sim(seed=0, hiv_beta_m2f=0.009, init_prev=0.02, art_coverage=0.3):
    """Build (but do not run) the Phase-1 co-infection anchor sim.

    Args:
        seed:         random seed.
        hiv_beta_m2f: male->female HIV per-act transmission (hand-tuned, not
                      calibrated). Drives the size of the HIV+ subpopulation.
        init_prev:    scalar initial HIV prevalence seeded at sim start.
        art_coverage: scalar ART coverage (proportion of diagnosed-infected).
    """
    return hpv.Sim(
        rand_seed=seed,
        genotypes=[16, 18, 'hi5', 'ohr'],
        diseases=[hpv.HIV_transmit(beta_m2f=hiv_beta_m2f, init_prev_data=init_prev)],
        interventions=[sti.ART(coverage=art_coverage)],
        verbose=0,
        **PARS,
    )


if __name__ == '__main__':
    import numpy as np
    sim = build_sim()
    sim.run()
    res = sim.results.hiv
    print('HIV prevalence (15-49), final:', float(res['prevalence_15_49'][-1]))
    print('HIV cum infections, final:', float(res['cum_infections'][-1]))
    print('HPV prev (HIV+), last-10y mean:',
          float(np.nanmean(res['hpv_prevalence_with_hiv'][-40:])))
    print('HPV prev (HIV-), last-10y mean:',
          float(np.nanmean(res['hpv_prevalence_no_hiv'][-40:])))
    all_hpv = sim.results.all_hpv
    print('HIV+ cancers (total):', int(all_hpv['cancers_with_hiv'].sum()))
    print('HIV- cancers (total):', int(all_hpv['cancers_no_hiv'].sum()))
