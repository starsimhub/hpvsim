"""Regression test: single-genotype vaccine must not bleed to other genotypes.

Verifies the fix for the v3 correctness bug where vaccine-conferred immunity
written to nab_imm was matrix-multiplied by CrossImmunity, giving vaccinated
agents extra protection against non-target genotypes on top of the CSV's
per-genotype rel_imm values.

After the fix: administer() writes to vax_imm (not nab_imm); CrossImmunity
applies vax_imm directly per target genotype without matrix amplification.
"""
import numpy as np
import pytest

import hpvsim as hpv


def test_single_genotype_vaccine_does_not_bleed_to_others():
    """A vaccine with rel_imm only for hpv16 must produce zero protection
    against other genotypes via the CrossImmunity connector.

    Regression test for the bug where v3 vaccine-conferred immunity
    propagated through the cross-immunity matrix, double-counting
    cross-protection on top of the CSV's per-genotype values.
    """
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

    # hpv16: vax_imm must be bumped (direct vaccine effect for the target genotype)
    assert (sim.diseases['hpv16'].vax_imm[vacc_uids] > 0).all(), \
        'Vaccinated agents must have vax_imm[hpv16]>0 (direct vaccine effect)'

    # Other genotypes: vax_imm must be EXACTLY zero (no vaccine entry for them).
    # This is the core regression guard — before the fix, nab_imm[hpv16]=1.0
    # bled through the cross-immunity matrix to reduce rel_sus for hpv18/hi5/ohr.
    for g in ['hpv18', 'hi5', 'ohr']:
        vax = sim.diseases[g].vax_imm[vacc_uids]
        assert float(vax.sum()) == 0.0, (
            f'{g} vax_imm leaked to {int((vax > 0).sum())} vaccinated agents; '
            f'a vaccine with no entry for {g} must not bump its vax_imm.'
        )

    # Verify rel_sus for non-hpv16 genotypes is driven only by nab_imm
    # (clearance) for vaccinated agents — not by any vaccine signal.
    # Since vax_imm[g] == 0 for all non-hpv16 genotypes (confirmed above),
    # rel_sus[g] = (1 - sus_imm_from_nab[g]) * (1 - 0) = 1 - sus_imm_from_nab[g].
    # We verify this by checking that rel_sus[g] is exactly consistent with
    # vax_imm[g] == 0: rel_sus[g] must not be artificially suppressed BELOW
    # what the nab_imm-only path would produce. The tightest test: since
    # vax_imm[g]==0 for all vaccinated agents, rel_sus[g] can range from 0 to 1
    # depending on clearance immunity — any value is valid so long as the mean
    # is not systematically lower than age-matched unvaccinated females.
    #
    # Age-matching: vaccinated agents are aged 9-14 in 2020 so at sim end
    # (2022) they are 11-18. We compare to unvaccinated females of the same
    # age range, which is the only fair cohort comparison.
    unvax_mask = sim.people.alive & ~intv.vaccinated
    ages = sim.people.age
    for g in ['hpv18', 'hi5', 'ohr']:
        vacc_vax = sim.diseases[g].vax_imm[vacc_uids]
        # Already confirmed to be zero above, but make it explicit for the
        # rel_sus reasoning.
        assert float(vacc_vax.sum()) == 0.0
        # Age-matched unvaccinated females (same approximate age window)
        vacc_age_min = float(ages[vacc_uids].min())
        vacc_age_max = float(ages[vacc_uids].max())
        age_matched = (unvax_mask
                       & sim.people.female
                       & (ages >= vacc_age_min)
                       & (ages <= vacc_age_max)).uids
        if len(age_matched) < 5:
            # Not enough age-matched peers; skip the rel_sus comparison
            # (the vax_imm==0 guard above is the authoritative check).
            continue
        vacc_rel_sus = float(sim.diseases[g].rel_sus[vacc_uids].mean())
        peer_rel_sus = float(sim.diseases[g].rel_sus[age_matched].mean())
        # Allow 15% relative noise (small-N, cohort age overlap is approximate).
        # The key guard is vax_imm==0 above; this is a belt-and-suspenders check.
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