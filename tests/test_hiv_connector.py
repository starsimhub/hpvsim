import numpy as np
import pytest
import hpvsim as hpv
from hpvsim.hiv import hpv_hiv_connector, HIVStratifiedResults

_DEFAULTS = hpv_hiv_connector().pars


def test_cd4_stratum_boundaries():
    c = hpv_hiv_connector()
    cd4 = np.array([50.0, 199.0, 200.0, 400.0, 594.0, 700.0, 800.0])
    strata = c._cd4_stratum(cd4)
    assert strata.dtype == bool
    # False = lo, True = hi (>=200, including >=500)
    assert list(strata) == [False, False, True, True, True, True, True]


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
    assert np.isclose(hpvmod.rel_sus[uid], _DEFAULTS.rel_sus_lo)
    other = sim.people.auids[1]
    assert np.isclose(hpvmod.rel_sus[other], 1.0)  # HIV- unchanged

    # The stored connector states (consumed by Tasks 4-6) are populated too.
    assert np.isclose(conn.hiv_rel_sus[uid], _DEFAULTS.rel_sus_lo)
    assert np.isclose(conn.hiv_rel_sev[uid], _DEFAULTS.rel_sev_lo)
    assert np.isclose(conn.hiv_rel_imm[uid], _DEFAULTS.rel_imm_lo)
    assert np.isclose(conn.hiv_rel_reactivation[uid], _DEFAULTS.rel_reactivation_lo)
    # Absolute-value anchor (documents the expected lt200 acquisition factor).
    assert np.isclose(hpvmod.rel_sus[uid], 2.2)
    # HIV- agent's stored factors stay neutral.
    assert np.isclose(conn.hiv_rel_sev[other], 1.0)
    assert np.isclose(conn.hiv_rel_imm[other], 1.0)
    assert np.isclose(conn.hiv_rel_reactivation[other], 1.0)


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
    assert np.isclose(hpvmod.rel_sus[uid], _DEFAULTS.rel_sus_hi)
    assert np.isclose(conn.hiv_rel_sev[uid], _DEFAULTS.rel_sev_hi)


def test_cd4_above_500_gets_no_effect():
    """HIV+ agents with CD4 >= 500 fall outside the gt200=[200,500)
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


# --- Configurable pars (location calibration override) ----------------------

_RWANDA_PARS = dict(
    rel_sus_lo=4.75, rel_sus_hi=2.75,
    rel_sev_lo=2.5,  rel_sev_hi=3.5,
    rel_imm_lo=0.36, rel_imm_hi=0.76,
)


def test_effects_override_applied():
    """A connector built with custom pars uses those multipliers, not defaults."""
    h = hpv.HIV(beta_m2f=0.0)
    conn = hpv_hiv_connector(pars=_RWANDA_PARS)
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
    hivmod.cd4[uid] = 100.0  # lo stratum
    hpvmod.rel_sus[sim.people.auids] = 1.0
    conn.step()
    assert np.isclose(hpvmod.rel_sus[uid], 4.75)          # Rwanda lo rel_sus
    assert np.isclose(conn.hiv_rel_sev[uid], 2.5)         # Rwanda lo rel_sev
    assert np.isclose(conn.hiv_rel_imm[uid], 0.36)
    # A partial override (only rel_sus/rel_sev/rel_imm given) leaves
    # rel_reactivation and cd4_threshold at their class defaults -- unlike the
    # old effects= dict, individual pars don't need to be restated.
    assert np.isclose(conn.pars.rel_reactivation_lo, _DEFAULTS.rel_reactivation_lo)
    assert np.isclose(conn.pars.cd4_threshold, _DEFAULTS.cd4_threshold)


def test_unrecognized_par_raises():
    """An unrecognized par name raises (update_pars's built-in guard),
    same as any other hpvsim module -- there is no more effects= dict to
    validate the shape of."""
    with pytest.raises(ValueError, match='bogus_par'):
        hpv_hiv_connector(bogus_par=2.0)


def test_user_supplied_analyzer_not_duplicated():
    """Auto-wiring skips a second HIVStratifiedResults when the user supplies one."""
    h = hpv.HIV(beta_m2f=0.0)
    sim = hpv.Sim(n_agents=200, start=2000, stop=2001, dt=0.25,
                  location='nigeria', genotypes=[16], diseases=[h],
                  analyzers=[HIVStratifiedResults()])
    sim.init()
    strat = [a for a in sim.analyzers.values() if isinstance(a, HIVStratifiedResults)]
    assert len(strat) == 1
