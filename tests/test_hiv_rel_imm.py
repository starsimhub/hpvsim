"""HPV clearance-conferred immunity is scaled by the HIV connector's
per-agent ``hiv_rel_imm`` factor (CD4-stratified), as a gated no-op when no HIV
connector is present.

The test is deterministic: two sims share a ``rand_seed`` and manipulate the SAME
female uid identically except for HIV status. Because Starsim's per-agent draws
(imm_init / cell_imm_init / seroconversion) are CRN-keyed by uid, those draws match
across the two sims for that uid, so the only difference in the conferred immunity
is the HIV rel_imm factor. With ``sero_prob=1.0`` seroconversion is guaranteed, so
nab_imm is non-zero and the ratio is clean.

Note on gating (from hpv.py step_state): nab_imm is gated on seroconversion
(``seroconvert * nab_all``) while cell_imm is NOT (all first-clearance females get
``cell_all``). With sero_prob=1.0 seroconvert==1, so both nab and cell are non-zero
and both must be scaled by the same hiv_rel_imm factor.
"""
import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector


def _build(seed, make_positive):
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25, location='nigeria',
                  rand_seed=seed, genotypes=[16],
                  genotype_pars={'hpv16': {'sero_prob': 1.0}},
                  diseases=[hpv.HIV(beta_m2f=0.0)])
    sim.init()
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    hivmod = sim.diseases.hiv
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]

    females = sim.people.auids[sim.people.female[sim.people.auids]]
    uid = females[0]

    if make_positive:
        hivmod.infected[uid] = True
        hivmod.cd4[uid] = 100.0  # lt200 stratum

    conn.step()  # populate hiv_rel_imm BEFORE clearance reads it

    # Put uid in first-clearance precin state, clearing this step.
    hpvmod.infected[uid] = True
    hpvmod.precin[uid] = True
    hpvmod.cin[uid] = False
    hpvmod.cancerous[uid] = False
    hpvmod.nab_imm[uid] = 0.0
    hpvmod.cell_imm[uid] = 0.0
    hpvmod.ti_clearance[uid] = sim.ti
    hpvmod.step_state()
    return float(hpvmod.nab_imm[uid]), float(hpvmod.cell_imm[uid])


def test_clearance_immunity_scaled_by_hiv_rel_imm():
    nab_neg, cell_neg = _build(seed=1, make_positive=False)
    nab_pos, cell_pos = _build(seed=1, make_positive=True)
    assert nab_neg > 0   # seroconversion fired (sero_prob=1.0)
    assert cell_neg > 0  # cell_imm always set for first-clearance females
    factor = hpv_hiv_connector().pars.rel_imm_lo  # == 0.36 (lo-stratum immunity multiplier)
    assert np.isclose(nab_pos, nab_neg * factor, rtol=1e-6)
    assert np.isclose(cell_pos, cell_neg * factor, rtol=1e-6)
