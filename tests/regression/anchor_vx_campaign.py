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
    intervention=sc.objdict(
        kind='campaign_vx',
        product='bivalent',
        prob=[0.7, 0.5],
        age_range=[9, 14],
        sex='f',
        years=[2020, 2021],
        interpolate=False,
        name='campaign_bivalent_catchup',
        v2_age_compat=True,  # demonstrate v2/v3 step-ordering alignment
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
        v2_age_compat=cfg.v2_age_compat,
    )


def build_v3_sim():
    import hpvsim as hpv
    # v2_age_compat also enables AnnualBirths so every year's birth cohort is
    # released as a single pulse (matching v2's add_births / dt_demog=1 logic).
    return hpv.Sim(
        location=PARS.location,
        start=PARS.start, stop=PARS.stop,
        rand_seed=PARS.rand_seed,
        n_agents=PARS.n_agents,
        genotypes=list(PARS.genotypes),
        interventions=[build_v3_intervention()],
        v2_compat_births=PARS.intervention.v2_age_compat,
    )