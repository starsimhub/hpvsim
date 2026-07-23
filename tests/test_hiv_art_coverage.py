"""T10b: coverage-based ART shortcut.

Rwanda has no HIV testing cascade — ART is assigned directly to hit an
age/sex/year coverage curve. These tests check that hpv.hiv_art drives
sti.ART from the Rwanda ART-coverage data so that (a) a plausible nonzero
fraction of HIV+ agents end up on_art in the ART era, and (b) STIsim's CD4
reconstitution actually fires for treated agents.
"""
import numpy as np
import hpvsim as hpv


def _coinfection_sim(interventions=None, stop=2018, seed=1):
    """Small Rwanda-like HIV+HPV sim seeded into the ART era."""
    # HIV/ART curves are loaded from Rwanda's bundled inputs; the sim's
    # demographics use 'nigeria' (the only location with bundled country
    # demographics so far) — the ART shortcut is independent of demographics.
    h = hpv.HIV.from_location('rwanda', beta_m2f=0.02, rel_beta_f2m=0.5)
    return hpv.Sim(
        n_agents=2000, start=1990, stop=stop, dt=0.5, rand_seed=seed,
        location='nigeria', genotypes=[16, 18], diseases=[h],
        interventions=interventions or [],
    )


def test_art_shortcut_treats_nonzero_fraction():
    """With hpv.hiv_art, a plausible nonzero fraction of HIV+ are on_art by 2017."""
    art = hpv.hiv_art.from_location('rwanda')
    sim = _coinfection_sim(interventions=[art])
    sim.run()
    hiv = sim.diseases.hiv
    hiv_pos = hiv.infected.uids
    assert len(hiv_pos) > 0, 'no HIV+ agents to treat'
    frac_on_art = hiv.on_art[hiv_pos].mean()
    # Rwanda ART coverage among HIV+ adults is well above 0 and below 1 by 2017.
    assert 0.1 < frac_on_art < 1.0, f'on-ART fraction {frac_on_art:.3f} out of plausible range'


def test_art_shortcut_reconstitutes_cd4():
    """CD4 among on-ART HIV+ agents exceeds CD4 among untreated HIV+ agents."""
    art = hpv.hiv_art.from_location('rwanda')
    sim = _coinfection_sim(interventions=[art])
    sim.run()
    hiv = sim.diseases.hiv
    hiv_pos = hiv.infected.uids
    on_art = hiv_pos[hiv.on_art[hiv_pos]]
    off_art = hiv_pos[~hiv.on_art[hiv_pos]]
    assert len(on_art) > 0 and len(off_art) > 0, 'need both treated and untreated HIV+'
    cd4_on = np.nanmean(hiv.cd4[on_art])
    cd4_off = np.nanmean(hiv.cd4[off_art])
    assert cd4_on > cd4_off, f'CD4 on-ART {cd4_on:.0f} not > untreated {cd4_off:.0f}'


def test_no_art_intervention_treats_nobody():
    """Sanity baseline: without the shortcut, sti.ART has nobody diagnosed so
    nobody is treated (the gap this task closes)."""
    sim = _coinfection_sim(interventions=[])
    sim.run()
    hiv = sim.diseases.hiv
    assert not hiv.on_art.any(), 'no ART intervention, yet someone is on_art'
