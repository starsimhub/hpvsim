"""M05 anchor scenario: one-off campaign bivalent vaccination of girls 9-14.

Mirrors the headline shape of hpvsim_1dose-style catch-up campaigns.
Used by tests/test_m05_vx_campaign_parity.py and the v2 baseline generator.
"""
import sciris as sc

PARS = sc.objdict(
    location='nigeria',
    start=1990, stop=2060,
    rand_seed=0,
    n_agents=20_000,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    # v2_compat_demographics gives every cohort (births + migration + initial
    # pop) integer ages, matching v2's annual birth pulse / dt_demog=1.
    # Required for per-cohort vaccination parity.
    v2_compat_demographics=True,
    intervention=sc.objdict(
        kind='campaign_vx',
        product='bivalent',
        prob=[0.7, 0.5],
        age_range=[9, 14],
        sex='f',
        years=[2020, 2021],
        interpolate=False,
        name='campaign_bivalent_catchup',
    ),
)


def build_v3_intervention():
    import hpvsim as hpv
    cfg = PARS.intervention
    return hpv.campaign_vx(
        product=cfg.product,
        prob=list(cfg.prob),
        age_range=cfg.age_range,
        sex=cfg.sex,
        years=list(cfg.years),
        interpolate=cfg.interpolate,
        name=cfg.name,
    )


def build_v3_sim():
    """Construct the v3 hpv.Sim used by the campaign parity test.

    v2_compat_demographics enables AnnualBirths (annual-pulse births) +
    AgeMigration jitter-disabled + initial-population age floor, so every
    agent — initial, born, or immigrated — lands at an exact integer age
    (matching v2's add_births / dt_demog=1 convention).

    Year-end convention: see ``anchor_vx_routine.build_v3_sim`` docstring.
    v2's sim builds ``yearvec = inclusiverange(start, end + 1 - dt, dt)``,
    so v2's end=2060 covers 1990.0 through 2060.75 (4 quarters of year
    2060). v3's stop is half-open, so we translate to stop + 1 - dt.

    NOTE: the campaign anchor does NOT use the +1-step trick that the
    routine anchor uses for its boundary-slice catch. The campaign
    intervention only fires in 2020-2021; by 2060 it has no boundary
    eligibility window to extend into. Adding a +1 step would only apply
    an extra quarter of mortality to the campaign-vaxed cohort, dropping
    n_vaccinated_2060 / n_doses_2060 below v2 (validated empirically
    2026-05-27).
    """
    import hpvsim as hpv
    base_stop = PARS.stop
    if isinstance(base_stop, (int, float)):
        dt = 0.25
        effective_stop = base_stop + (1 - dt)
    else:
        effective_stop = base_stop
    return hpv.Sim(
        location=PARS.location,
        start=PARS.start, stop=effective_stop,
        rand_seed=PARS.rand_seed,
        n_agents=PARS.n_agents,
        genotypes=list(PARS.genotypes),
        interventions=[build_v3_intervention()],
        v2_compat_demographics=PARS.v2_compat_demographics,
    )