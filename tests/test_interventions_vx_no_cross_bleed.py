"""A single-genotype vaccine must not bleed protection to other genotypes.

administer() writes to vax_imm, and CrossImmunity applies vax_imm directly per
target genotype rather than through the cross-immunity matrix, so a vaccine with
no entry for a genotype confers nothing against it.
"""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv


def _run_hpv16_vaccine():
    """Vaccinate girls aged 9-14 against hpv16 only; return the sim, those vaccinated (alive), and whether each agent was"""
    reached = []
    def track(sim):
        reached.extend(sim.interventions.vx.reached().tolist())
    girls = ss.ppl.female & (ss.ppl.age >= 9) & (ss.ppl.age < 14)
    vx = ss.vaccinate(hpv.vx(rel_imm={'hpv16': 1.0}), where=girls, coverage=1.0, start=2020, name='vx')
    sim = hpv.Sim(location='nigeria', start=2018, stop=2022, n_agents=500, genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
                  rand_seed=0, interventions=[vx], analyzers=track, verbose=0)
    sim.run()
    vaccinated = sim.people.alive.asnew()
    vaccinated.raw[:] = False
    vaccinated[ss.uids(reached)] = True
    vacc_uids = (vaccinated & sim.people.alive).uids
    if len(vacc_uids) == 0:
        pytest.skip('No agents vaccinated in this small-sim window')
    return sim, vacc_uids, vaccinated


def test_single_genotype_vaccine_does_not_bleed_to_others():
    """A vaccine with rel_imm only for hpv16 gives zero protection against the
    other genotypes via the CrossImmunity connector."""
    sim, vacc_uids, vaccinated = _run_hpv16_vaccine()

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
    unvax_mask = sim.people.alive & ~vaccinated
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
    sim, vacc_uids, vaccinated = _run_hpv16_vaccine()

    unvax_mask = sim.people.alive & ~vaccinated
    unvax_f = (unvax_mask & sim.people.female).uids
    if len(unvax_f) == 0:
        pytest.skip('No unvaccinated females for comparison')

    vacc_rel_sus = sim.diseases['hpv16'].rel_sus[vacc_uids].mean()
    unvax_rel_sus = sim.diseases['hpv16'].rel_sus[unvax_f].mean()
    assert vacc_rel_sus < unvax_rel_sus, (
        f'hpv16-only vaccine must reduce hpv16 rel_sus for vaccinated agents: '
        f'vacc={vacc_rel_sus:.4f}, unvacc={unvax_rel_sus:.4f}'
    )