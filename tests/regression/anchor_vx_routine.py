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
    # v2_compat_demographics gives every cohort (births + migration + initial
    # pop) integer ages, matching v2's annual birth pulse / dt_demog=1.
    # Required for per-cohort vaccination parity.
    v2_compat_demographics=True,
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
    )


def build_v3_sim(stop=None):
    """Construct the v3 hpv.Sim used by the parity test.

    The parity gate always uses ``PARS.stop`` (full 1990-2060 horizon) — do
    NOT pass ``stop`` from the parity test. The kwarg exists for diagnostic
    plot scripts that want a shorter horizon while troubleshooting.

    v2_compat_demographics enables AnnualBirths (annual-pulse births) +
    AgeMigration jitter-disabled + initial-population age floor, so every
    agent — initial, born, or immigrated — lands at an exact integer age
    (matching v2's add_births / dt_demog=1 convention).

    Year-end convention: v2's sim builds ``yearvec = inclusiverange(start,
    end + 1 - dt, dt)`` (sim.py:253), so v2 with end=2060 actually covers
    1990.0 through 2060.75 (4 quarters of year 2060). v3's Starsim takes
    stop literally, so stop=2060 covers only 1990.0 through 2060.0 (1
    quarter of year 2060). To match v2's year-end-inclusive coverage AND
    its pre-vs-post-increment-age step ordering at the final step, we run
    v3 for one additional quarterly step (effective_stop = PARS.stop + 1
    rather than + 1 - dt). The extra step lets v3's routine_vx fire on
    the agents whose post-finish_step age has just crossed into the 9-10
    window — the boundary slice v2 catches via its update_states_pre age
    advance. See ``docs/superpowers/specs/2026-05-26-m05-parity-investigation.md``.
    Trajectory test clips v3 to v2's year range to absorb the extra bucket.
    """
    import hpvsim as hpv
    base_stop = stop if stop is not None else PARS.stop
    # Match v2's year-end-inclusive convention (see docstring above). Only
    # translate plain numbers — leave anything pre-wrapped (e.g. ss.years)
    # untouched so callers can override if needed.
    if isinstance(base_stop, (int, float)):
        effective_stop = base_stop + 1
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