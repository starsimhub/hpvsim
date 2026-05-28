"""M06 anchor scenario: routine therapeutic vaccination (txvx1) of women aged 25-26.

Used by tests/test_m06_txvx_routine_parity.py and the v2 baseline generator.
"""
import sciris as sc

# Anchor PARS — vanilla M03 Nigeria 4-genotype sim + one routine_txvx intervention.
PARS = sc.objdict(
    location='nigeria',
    start=1990, stop=2060,
    rand_seed=0,
    n_agents=20_000,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    # v2_compat_demographics gives every cohort (births + migration + initial
    # pop) integer ages, matching v2's annual birth pulse / dt_demog=1.
    # Required for per-cohort vaccination parity (M05 lesson).
    v2_compat_demographics=True,
    # The intervention spec is recorded as a serializable dict because the
    # v2 baseline script (Task 19) constructs the v2 equivalent from these
    # fields.
    intervention=sc.objdict(
        kind='routine_txvx',
        product='txvx1',
        prob=0.6,
        age_range=[25, 26],
        sex='f',
        start_year=2030,
        end_year=2060,
        name='txvx',
    ),
)


def _build_intervention():
    """Construct the v3 hpv.routine_txvx from PARS.intervention."""
    import hpvsim as hpv
    cfg = PARS.intervention
    return hpv.routine_txvx(
        name=cfg.name,
        product=cfg.product,
        prob=cfg.prob,
        age_range=cfg.age_range,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
    )


def build_v3_sim(stop=None, n_agents=None):
    """Construct the v3 hpv.Sim used by the parity test.

    The parity gate always uses ``PARS.stop`` (full 1990-2060 horizon) — do
    NOT pass ``stop`` from the parity test. The kwarg exists for diagnostic
    plot scripts that want a shorter horizon while troubleshooting.

    ``n_agents`` overrides ``PARS.n_agents`` for smoke/diagnostic runs that
    need a smaller population. Intervention end_year (2060) must remain <=
    stop, so callers should not reduce stop below 2060.

    v2_compat_demographics enables AnnualBirths (annual-pulse births) +
    AgeMigration jitter-disabled + initial-population age floor, so every
    agent — initial, born, or immigrated — lands at an exact integer age
    (matching v2's add_births / dt_demog=1 convention).

    Year-end convention: mirrors M05's anchor — v3 stop is extended by +1
    so that v3's year-end-inclusive coverage matches v2's convention
    (see anchor_vx_routine.py docstring for full explanation).
    """
    import hpvsim as hpv
    base_stop = stop if stop is not None else PARS.stop
    # Match v2's year-end-inclusive convention. Only translate plain numbers
    # — leave anything pre-wrapped (e.g. ss.years) untouched.
    if isinstance(base_stop, (int, float)):
        effective_stop = base_stop + 1
    else:
        effective_stop = base_stop
    return hpv.Sim(
        location=PARS.location,
        start=PARS.start, stop=effective_stop,
        rand_seed=PARS.rand_seed,
        n_agents=n_agents if n_agents is not None else PARS.n_agents,
        genotypes=list(PARS.genotypes),
        interventions=[_build_intervention()],
        v2_compat_demographics=PARS.v2_compat_demographics,
    )
