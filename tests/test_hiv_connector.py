import numpy as np
import pytest
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector, HIVStratifiedResults, _HIV_EFFECTS


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
                  location='nigeria', genotypes=[16], diseases=[h])
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
                  location='nigeria', genotypes=[16], diseases=[h])
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


def test_cd4_above_500_gets_no_effect():
    """v2-faithful: HIV+ agents with CD4 >= 500 fall outside v2's gt200=[200,500)
    band and receive NO HIV->HPV effect (factor 1.0). HIV+ agents START at
    CD4 ~594 and ART reconstitutes above 500, so this band is the majority of
    HIV+ person-time; applying gt200 there over-amplifies HIV+ cancer ~10x."""
    h = hpv.HIV(beta_m2f=0.0)
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h])
    sim.init()
    hivmod = sim.diseases.hiv
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    conn = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)][0]
    uid = sim.people.auids[0]
    hivmod.infected[uid] = True
    hivmod.cd4[uid] = 594.0  # newly-infected starting CD4, >= 500
    hpvmod.rel_sus[sim.people.auids] = 1.0
    conn.step()
    assert np.isclose(hpvmod.rel_sus[uid], 1.0)        # no acquisition boost
    assert np.isclose(conn.hiv_rel_sev[uid], 1.0)      # no severity boost
    assert np.isclose(conn.hiv_rel_imm[uid], 1.0)      # no immunity reduction


# --- Configurable effects (location calibration override) -------------------

_RWANDA_EFFECTS = {
    'rel_sus': {'lt200': 4.75, 'gt200': 2.75},
    'rel_sev': {'lt200': 2.5, 'gt200': 3.5},
    'rel_imm': {'lt200': 0.36, 'gt200': 0.76},
}


def test_effects_override_applied():
    """A connector built with custom effects= uses those multipliers, not defaults."""
    h = hpv.HIV(beta_m2f=0.0)
    conn = hpv_hiv_connector(effects=_RWANDA_EFFECTS)
    sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h],
                  connectors=[conn])
    sim.init()
    hivmod = sim.diseases.hiv
    hpvmod = [d for d in sim.diseases.values() if isinstance(d, hpv.HPV)][0]
    # Auto-wiring must NOT add a second connector when one is user-supplied.
    conns = [c for c in sim.connectors.values() if isinstance(c, hpv_hiv_connector)]
    assert len(conns) == 1
    conn = conns[0]
    uid = sim.people.auids[0]
    hivmod.infected[uid] = True
    hivmod.cd4[uid] = 100.0  # lt200
    hpvmod.rel_sus[sim.people.auids] = 1.0
    conn.step()
    assert np.isclose(hpvmod.rel_sus[uid], 4.75)          # Rwanda lt200 rel_sus
    assert np.isclose(conn.hiv_rel_sev[uid], 2.5)         # Rwanda lt200 rel_sev
    assert np.isclose(conn.hiv_rel_imm[uid], 0.36)


def test_effects_override_validates_shape():
    with pytest.raises(ValueError, match='rel_sus'):
        hpv_hiv_connector(effects={'rel_sus': {'lt200': 2.0}})  # missing keys


def test_user_supplied_analyzer_not_duplicated():
    """Auto-wiring skips a second HIVStratifiedResults when the user supplies one."""
    h = hpv.HIV(beta_m2f=0.0)
    sim = hpv.Sim(n_agents=200, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h],
                  analyzers=[HIVStratifiedResults()])
    sim.init()
    strat = [a for a in sim.analyzers.values() if isinstance(a, HIVStratifiedResults)]
    assert len(strat) == 1
