"""Verify CrossImmunity combines nab_imm, vax_imm, and txvx_imm as
independent protection paths.

rel_sus[target] = (1 - sus_imm_nab[target]) * (1 - vax_imm[target]) * (1 - txvx_imm[target])
"""
import numpy as np
import starsim as ss
import hpvsim as hpv


def _two_genotype_sim():
    return hpv.Sim(
        n_agents=500, start=2020, stop=2022, location='nigeria',
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
    )


def test_txvx_imm_reduces_rel_sus_independently():
    sim = _two_genotype_sim()
    sim.init()
    mod16 = sim.diseases['hpv16']
    uids = sim.people.alive.uids[:5]
    mod16.vax_imm[uids] = 0.5
    mod16.txvx_imm[uids] = 0.5
    sim.connectors['crossimmunity'].step()
    # Independent combine: rel_sus = (1 - 0) * (1 - 0.5) * (1 - 0.5) = 0.25
    np.testing.assert_allclose(mod16.rel_sus[uids], 0.25, atol=1e-6)


def test_txvx_imm_alone_reduces_rel_sus():
    sim = _two_genotype_sim()
    sim.init()
    mod16 = sim.diseases['hpv16']
    uids = sim.people.alive.uids[:5]
    mod16.txvx_imm[uids] = 0.7
    sim.connectors['crossimmunity'].step()
    # rel_sus = (1 - 0) * (1 - 0) * (1 - 0.7) = 0.3
    np.testing.assert_allclose(mod16.rel_sus[uids], 0.3, atol=1e-6)


def test_txvx_imm_does_not_bleed_across_genotypes():
    """txvx_imm is per-target, NOT matrix-multiplied."""
    sim = _two_genotype_sim()
    sim.init()
    mod16, mod18 = sim.diseases['hpv16'], sim.diseases['hpv18']
    uids = sim.people.alive.uids[:5]
    mod16.txvx_imm[uids] = 1.0
    sim.connectors['crossimmunity'].step()
    # hpv18 rel_sus is the all-1.0 baseline (no vax, no nab, no txvx on hpv18)
    np.testing.assert_allclose(mod18.rel_sus[uids], 1.0, atol=1e-6)
    # hpv16 rel_sus is zero
    np.testing.assert_allclose(mod16.rel_sus[uids], 0.0, atol=1e-6)
