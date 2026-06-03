import numpy as np
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector, _HIV_EFFECTS


def test_cd4_stratum_boundaries():
    c = hpv_hiv_connector()
    cd4 = np.array([50.0, 199.0, 200.0, 400.0, 700.0])
    strata = c._cd4_stratum(cd4)
    # 0 = lt200, 1 = gt200 (>=200, including >=500)
    assert list(strata) == [0, 0, 1, 1, 1]


def test_rel_sus_scaled_for_hiv_positive():
    """After step(), an HIV+ agent's HPV rel_sus is multiplied by the stratum factor."""
    h = hpv.HIV(beta_m2f=0.0)  # no transmission; we set HIV state manually
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h],
                  connectors=[hpv_hiv_connector()])
    sim.init()
    hivmod = sim.diseases.hiv
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]

    # Force one agent HIV+ with low CD4, rest HIV-negative.
    uid = sim.people.auids[0]
    hivmod.infected[uid] = True
    hivmod.cd4[uid] = 100.0  # lt200
    hpvmod.rel_sus[sim.people.auids] = 1.0

    conn.step()
    assert np.isclose(hpvmod.rel_sus[uid], _HIV_EFFECTS['rel_sus']['lt200'])
    other = sim.people.auids[1]
    assert np.isclose(hpvmod.rel_sus[other], 1.0)  # HIV- unchanged

    # The stored connector states (consumed by Tasks 4-6) are populated too.
    assert np.isclose(conn.hiv_rel_sus[uid], _HIV_EFFECTS['rel_sus']['lt200'])
    assert np.isclose(conn.hiv_rel_sev[uid], _HIV_EFFECTS['rel_sev']['lt200'])
    assert np.isclose(conn.hiv_rel_imm[uid], _HIV_EFFECTS['rel_imm']['lt200'])
    # Absolute-value anchor (documents the expected lt200 acquisition factor).
    assert np.isclose(hpvmod.rel_sus[uid], 2.2)
    # HIV- agent's stored factors stay neutral.
    assert np.isclose(conn.hiv_rel_sev[other], 1.0)
    assert np.isclose(conn.hiv_rel_imm[other], 1.0)


def test_rel_sus_gt200_stratum():
    """An HIV+ agent with CD4>=200 gets the gt200 acquisition factor."""
    h = hpv.HIV(beta_m2f=0.0)
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h],
                  connectors=[hpv_hiv_connector()])
    sim.init()
    hivmod = sim.diseases.hiv
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    uid = sim.people.auids[0]
    hivmod.infected[uid] = True
    hivmod.cd4[uid] = 350.0  # gt200
    hpvmod.rel_sus[sim.people.auids] = 1.0
    conn.step()
    assert np.isclose(hpvmod.rel_sus[uid], _HIV_EFFECTS['rel_sus']['gt200'])
    assert np.isclose(conn.hiv_rel_sev[uid], _HIV_EFFECTS['rel_sev']['gt200'])
