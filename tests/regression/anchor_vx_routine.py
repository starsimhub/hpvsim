"""M05 anchor scenario: routine bivalent vaccination of girls aged 9-10.

Mirrors the headline shape of hpvsim_pxv_younger-style routine programs.
Used by tests/test_m05_vx_routine_parity.py and the v2 baseline generator.
"""
import sciris as sc

# Anchor PARS — vanilla M03 Nigeria 4-genotype + one routine_vx intervention.
PARS = sc.objdict(
    location='nigeria',
    start=1990, stop=2060,
    rand_seed=0,
    n_agents=20_000,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    # The intervention spec is recorded as a serializable dict because the
    # v2 baseline script (Task 9) constructs the v2 equivalent from these
    # fields.
    intervention=sc.objdict(
        kind='routine_vx',
        product='bivalent',
        prob=0.9,
        age_range=[9, 10],
        sex='f',
        start_year=2020,
        name='routine_bivalent_girls',
        v2_age_compat=True,  # demonstrate v2/v3 step-ordering alignment
    ),
)


def build_v3_intervention():
    """Construct the v3 hpv.routine_vx from PARS.intervention."""
    import hpvsim as hpv
    cfg = PARS.intervention
    return hpv.routine_vx(
        product=cfg.product,
        prob=cfg.prob,
        age_range=cfg.age_range,
        sex=cfg.sex,
        start_year=cfg.start_year,
        name=cfg.name,
        v2_age_compat=cfg.v2_age_compat,
    )


def build_v3_sim():
    """Construct the v3 hpv.Sim used by the parity test."""
    import hpvsim as hpv
    # v2_compat_demographics enables both AnnualBirths (annual-pulse births)
    # and AgeMigration jitter-disabled, so every year's cohort — whether born
    # or immigrated — lands at exact integer ages (matching v2's add_births /
    # dt_demog=1 convention).
    return hpv.Sim(
        location=PARS.location,
        start=PARS.start, stop=PARS.stop,
        rand_seed=PARS.rand_seed,
        n_agents=PARS.n_agents,
        genotypes=list(PARS.genotypes),
        interventions=[build_v3_intervention()],
        v2_compat_demographics=PARS.intervention.v2_age_compat,
    )