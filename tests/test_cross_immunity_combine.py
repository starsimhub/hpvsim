"""Verify CrossImmunity combines its protection paths correctly.

rel_sus[target] = (1 - sus_imm_nab[target]) * (1 - vax_imm[target])
sev_imm[target] = sum_k cross_imm_sev[target, k] * cell_imm[k] + txvx_sev_imm[target]

The therapeutic acts on severity, not acquisition, so it appears in the second
formula only.
"""
import numpy as np
import starsim as ss
import hpvsim as hpv


def make_sim():
    return hpv.Sim(
        n_agents=500, start=2020, stop=2022, location='nigeria',
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
    )


def test_vax_imm_reduces_rel_sus_independently():
    sim = make_sim()
    sim.init()
    mod16 = sim.diseases['hpv16']
    uids = sim.people.alive.uids[:5]
    mod16.vax_imm[uids] = 0.5
    sim.connectors['crossimmunity'].step()
    np.testing.assert_allclose(mod16.rel_sus[uids], 0.5, atol=1e-6)


def test_vax_imm_does_not_bleed_across_genotypes():
    """vax_imm is per-target, NOT matrix-multiplied."""
    sim = make_sim()
    sim.init()
    mod16, mod18 = sim.diseases['hpv16'], sim.diseases['hpv18']
    uids = sim.people.alive.uids[:5]
    mod16.vax_imm[uids] = 1.0
    sim.connectors['crossimmunity'].step()
    np.testing.assert_allclose(mod18.rel_sus[uids], 1.0, atol=1e-6)
    np.testing.assert_allclose(mod16.rel_sus[uids], 0.0, atol=1e-6)


def test_txvx_sev_imm_adds_to_sev_imm_without_touching_rel_sus():
    """Therapeutic immunity raises sev_imm on its target only, and never rel_sus."""
    sim = make_sim()
    sim.init()
    mod16, mod18 = sim.diseases['hpv16'], sim.diseases['hpv18']
    uids = sim.people.alive.uids[:5]
    mod16.txvx_sev_imm[uids] = 0.4
    sim.connectors['crossimmunity'].step()
    np.testing.assert_allclose(mod16.sev_imm[uids], 0.4, atol=1e-6)
    np.testing.assert_allclose(mod18.sev_imm[uids], 0.0, atol=1e-6)
    np.testing.assert_allclose(mod16.rel_sus[uids], 1.0, atol=1e-6)


def test_txvx_sev_imm_sums_with_clearance_immunity_and_clips():
    """Therapeutic and clearance-conferred severity immunity add, capped at 1."""
    sim = make_sim()
    sim.init()
    mod16 = sim.diseases['hpv16']
    uids = sim.people.alive.uids[:5]
    mod16.cell_imm[uids] = 0.5          # own-immunity diagonal is 1.0 for hpv16
    mod16.txvx_sev_imm[uids] = 0.2
    sim.connectors['crossimmunity'].step()
    np.testing.assert_allclose(mod16.sev_imm[uids], 0.7, atol=1e-6)

    mod16.txvx_sev_imm[uids] = 0.9
    sim.connectors['crossimmunity'].step()
    np.testing.assert_allclose(mod16.sev_imm[uids], 1.0, atol=1e-6)
