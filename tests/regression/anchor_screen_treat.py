"""M06 anchor scenario: HPV screen -> colposcopy triage -> excision treatment cascade.

Mirrors the headline HSP (HPV Screen and Treat) shape of the
hpvsim_methods_manuscript scenario on a Nigeria-base M03 sim.
Used by tests/test_m06_screen_treat_parity.py and the v2 baseline generator.
"""
import sciris as sc

# Anchor PARS — vanilla M03 Nigeria 4-genotype sim + 3-step cascade.
PARS = sc.objdict(
    location='nigeria',
    start=1990, stop=2060,
    rand_seed=0,
    n_agents=20_000,
    genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    # v2_compat_demographics gives every cohort (births + migration + initial
    # pop) integer ages, matching v2's annual birth pulse / dt_demog=1.
    # Required for per-cohort cascade parity (M05 lesson).
    v2_compat_demographics=True,
    # Each cascade step is recorded as a serializable dict so the v2 baseline
    # script (Task 19) can construct the equivalent v2 interventions.
    screen=sc.objdict(
        kind='routine_screening',
        product='hpv',
        prob=0.7,
        age_range=[30, 50],
        sex='f',
        start_year=2020,
        end_year=2060,
        name='primary',
    ),
    triage=sc.objdict(
        kind='routine_triage',
        product='colposcopy',
        prob=0.9,
        # eligibility: screen-positives from the primary screen
        eligibility_ref='primary.outcomes.positive',
        sex='f',
        start_year=2020,
        end_year=2060,
        name='colpo',
    ),
    treat=sc.objdict(
        kind='treat_num',
        product='excision',
        prob=0.8,
        # eligibility: HSIL-positive from colposcopy triage
        eligibility_ref='colpo.outcomes.hsil',
        name='excision_rx',
    ),
)


def _build_interventions():
    """Construct the v3 cascade intervention list from PARS."""
    import hpvsim as hpv

    screen_cfg = PARS.screen
    primary = hpv.routine_screening(
        name=screen_cfg.name,
        product=screen_cfg.product,
        prob=screen_cfg.prob,
        age_range=screen_cfg.age_range,
        sex=screen_cfg.sex,
        start_year=screen_cfg.start_year,
        end_year=screen_cfg.end_year,
    )

    triage_cfg = PARS.triage
    colpo = hpv.routine_triage(
        name=triage_cfg.name,
        product=triage_cfg.product,
        prob=triage_cfg.prob,
        eligibility=lambda s: s.interventions['primary'].outcomes['positive'],
        start_year=triage_cfg.start_year,
        end_year=triage_cfg.end_year,
    )

    treat_cfg = PARS.treat
    excision = hpv.treat_num(
        name=treat_cfg.name,
        product=treat_cfg.product,
        prob=treat_cfg.prob,
        eligibility=lambda s: s.interventions['colpo'].outcomes['hsil'],
    )

    return [primary, colpo, excision]


def build_v3_sim(stop=None, n_agents=None):
    """Construct the v3 hpv.Sim used by the parity test.

    The parity gate always uses ``PARS.stop`` (full 1990-2060 horizon) — do
    NOT pass ``stop`` from the parity test. The kwarg exists for diagnostic
    plot scripts that want a shorter horizon while troubleshooting.

    ``n_agents`` overrides ``PARS.n_agents`` for smoke/diagnostic runs that
    need a smaller population. Intervention end_year values (2060) must
    remain <= stop, so callers should not reduce stop below 2060 unless they
    also rebuild the interventions with matching end_year values.

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
        interventions=_build_interventions(),
        v2_compat_demographics=PARS.v2_compat_demographics,
    )
