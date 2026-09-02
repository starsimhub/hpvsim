"""A single-genotype vaccine must not bleed protection to other genotypes.

administer() writes to vax_imm, and CrossImmunity applies vax_imm directly per
target genotype rather than through the cross-immunity matrix, so a vaccine with
no entry for a genotype confers nothing against it.
"""
import numpy as np
import pytest

import hpvsim as hpv


def test_single_genotype_vaccine_does_not_bleed_to_others():
    """A vaccine with rel_imm only for hpv16 gives zero protection against the
    other genotypes via the CrossImmunity connector."""
    intv = hpv.routine_vx(
        product=hpv.vx(rel_imm={'hpv16': 1.0}),
        prob=1.0,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='hpv16_only_smoke',
    )
    sim = hpv.Sim(
        location='nigeria', start=2018, stop=2022,
        n_agents=500,
        genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
        rand_seed=0,
        interventions=[intv],
    )
    sim.run()
    intv = sim.interventions[0]
    vacc_uids = intv.vaccinated.uids
    if len(vacc_uids) == 0:
        pytest.skip('No agents vaccinated in this small-sim window')

    assert (sim.diseases['hpv16'].vax_imm[vacc_uids] > 0).all(), \
        'Vaccinated agents must have vax_imm[hpv16]>0 (direct vaccine effect)'

    # Genotypes with no vaccine entry must have vax_imm exactly zero.
    for g in ['hpv18', 'hi5', 'ohr']:
        vax = sim.diseases[g].vax_imm[vacc_uids]
        assert float(vax.sum()) == 0.0, (
            f'{g} vax_imm leaked to {int((vax > 0).sum())} vaccinated agents; '
            f'a vaccine with no entry for {g} must not bump its vax_imm.'
        )

    # With vax_imm[g]==0, rel_sus[g] must track nab_imm only: no lower than
    # age-matched unvaccinated females.
    unvax_mask = sim.people.alive & ~intv.vaccinated
    ages = sim.people.age
    for g in ['hpv18', 'hi5', 'ohr']:
        vacc_vax = sim.diseases[g].vax_imm[vacc_uids]
        assert float(vacc_vax.sum()) == 0.0
        vacc_age_min = float(ages[vacc_uids].min())
        vacc_age_max = float(ages[vacc_uids].max())
        age_matched = (unvax_mask
                       & sim.people.female
                       & (ages >= vacc_age_min)
                       & (ages <= vacc_age_max)).uids
        if len(age_matched) < 5:
            continue
        vacc_rel_sus = float(sim.diseases[g].rel_sus[vacc_uids].mean())
        peer_rel_sus = float(sim.diseases[g].rel_sus[age_matched].mean())
        # 15% tolerance: small-N and the age window only approximately matches.
        rel_diff = abs(vacc_rel_sus - peer_rel_sus) / max(peer_rel_sus, 1e-12)
        assert rel_diff < 0.15, (
            f'{g} rel_sus differs between vaccinated and age-matched unvaccinated '
            f'females: vacc={vacc_rel_sus:.4f}, peers={peer_rel_sus:.4f} '
            f'(rel_diff={rel_diff:.4f}). '
            f'With vax_imm[{g}]==0 confirmed, rel_sus should track nab_imm only.'
        )


def test_hpv16_vaccine_does_reduce_hpv16_susceptibility():
    """Positive control: the hpv16-only vaccine must reduce hpv16 rel_sus
    for vaccinated agents compared to unvaccinated females.
    """
    intv = hpv.routine_vx(
        product=hpv.vx(rel_imm={'hpv16': 1.0}),
        prob=1.0,
        age_range=[9, 14],
        sex='f',
        start_year=2020,
        name='hpv16_only_positive_control',
    )
    sim = hpv.Sim(
        location='nigeria', start=2018, stop=2022,
        n_agents=500,
        genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
        rand_seed=0,
        interventions=[intv],
    )
    sim.run()
    intv = sim.interventions[0]
    vacc_uids = intv.vaccinated.uids
    if len(vacc_uids) == 0:
        pytest.skip('No agents vaccinated in this small-sim window')

    unvax_mask = sim.people.alive & ~intv.vaccinated
    unvax_f = (unvax_mask & sim.people.female).uids
    if len(unvax_f) == 0:
        pytest.skip('No unvaccinated females for comparison')

    vacc_rel_sus = sim.diseases['hpv16'].rel_sus[vacc_uids].mean()
    unvax_rel_sus = sim.diseases['hpv16'].rel_sus[unvax_f].mean()
    assert vacc_rel_sus < unvax_rel_sus, (
        f'hpv16-only vaccine must reduce hpv16 rel_sus for vaccinated agents: '
        f'vacc={vacc_rel_sus:.4f}, unvacc={unvax_rel_sus:.4f}'
    )