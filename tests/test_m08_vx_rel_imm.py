"""M08 Task 6: vaccine / txvx products scale conferred immunity by hiv_rel_imm.

When a prophylactic (hpv.vx) or therapeutic (hpv.txvx) vaccine confers immunity
to an HIV+ agent, the conferred vax_imm / txvx_imm must be reduced by the
hpv_hiv_connector's per-agent hiv_rel_imm factor. Gated no-op without an HIV
connector (covered by the M05/M06 vaccine guards, which run without HIV).
"""
import numpy as np
import starsim as ss
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector, _HIV_EFFECTS
from hpvsim.products import vx as hpv_vx, txvx as hpv_txvx


def _coinfection_sim(product=None):
    """A small HIV+HPV sim with one HIV+ (lt200) and one HIV- female.

    If ``product`` is given it is attached via a stub treat_num intervention
    (prob=0.0) so the full sim.init() initializes the product's internal Dists,
    exactly like the M06 unit tests. Returns the LIVE post-init product copy.
    """
    interventions = [ss.treat_num(product=product, prob=0.0)] if product is not None else None
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25, location='nigeria',
                  genotypes=[16, 18], diseases=[hpv.HIV(beta_m2f=0.0)],
                  interventions=interventions)
    sim.init()
    hivmod = sim.diseases.hiv
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    females = sim.people.auids[sim.people.female[sim.people.auids]]
    assert len(females) >= 2  # fixture must yield at least two female agents
    uid_pos, uid_neg = females[0], females[1]
    hivmod.infected[uid_pos] = True
    hivmod.cd4[uid_pos] = 100.0  # lt200
    conn.step()
    live_product = sim.interventions[0].product if product is not None else None
    return sim, conn, uid_pos, uid_neg, live_product


def test_connector_rel_imm_factors():
    """Sanity: the connector sets the expected per-agent rel_imm factors."""
    sim, conn, uid_pos, uid_neg, _ = _coinfection_sim()
    assert np.isclose(conn.hiv_rel_imm[uid_pos], _HIV_EFFECTS['rel_imm']['lt200'])  # 0.36
    assert np.isclose(conn.hiv_rel_imm[uid_neg], 1.0)


def test_vaccine_imm_scaled_for_hiv_positive():
    """A prophylactic vaccine confers strictly less vax_imm to an HIV+ lt200
    agent than to an otherwise-identical HIV- agent."""
    # Deterministic take (sterilizing_p=1.0) so both agents get rel_imm[g] pre-scaling;
    # the HIV+ one must end up strictly smaller after hiv_rel_imm scaling.
    sim, conn, uid_pos, uid_neg, product = _coinfection_sim(
        product=hpv_vx(name='nonavalent', sterilizing_p=1.0))
    product.administer(sim.people, np.array([uid_pos, uid_neg]))
    hpvmod = sim.diseases.hpv16
    assert float(hpvmod.vax_imm[uid_pos]) < float(hpvmod.vax_imm[uid_neg])


def test_txvx_imm_scaled_for_hiv_positive():
    """A therapeutic vaccine confers strictly less txvx_imm to an HIV+ lt200
    agent than to an otherwise-identical HIV- agent."""
    sim, conn, uid_pos, uid_neg, product = _coinfection_sim(
        product=hpv_txvx(name='txvx1', sterilizing_p=1.0))
    product.administer(sim.people, np.array([uid_pos, uid_neg]))
    hpvmod = sim.diseases.hpv16
    assert float(hpvmod.txvx_imm[uid_pos]) < float(hpvmod.txvx_imm[uid_neg])
