"""HIV-HPV co-infection: HIV_transmit (network-driven) and HIV_incidence
(curve-driven), both built on the shared hpv.HIV base that owns the
CD4-stratified HPV-modulation effects and HIV-stratified results.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starsim as ss
import stisim as sti

import hpvsim as hpv
from hpvsim.network import SexualNetwork
from hpvsim.products import vx as hpv_vx, txvx as hpv_txvx

_RWANDA_HIV_DATA = Path(__file__).parent / 'regression' / 'data' / 'hiv' / 'rwanda'


def _tiny(**kw):
    return dict(n_agents=300, start=2000, stop=2001, dt=0.25, location='nigeria', **kw)


# --------------------------------------------------------------------------- #
# Wiring: gating
# --------------------------------------------------------------------------- #

def test_hiv_gating():
    """HPV._hiv_module() finds the HIV disease when present, None otherwise;
    the CD4-stratified factors default to a no-op (1.0) with no HIV+ agents."""
    sim = hpv.Sim(**_tiny(genotypes=[16], diseases=[hpv.HIV_transmit(beta_m2f=0.0)]))
    sim.init()
    hpvmod = sim.diseases.hpv16
    assert hpvmod._hiv_module() is sim.diseases.hiv
    assert np.allclose(sim.diseases.hiv.hiv_rel_sev.values[:5], 1.0)

    sim_no_hiv = hpv.Sim(**_tiny(genotypes=[16]))
    sim_no_hiv.init()
    assert sim_no_hiv.diseases.hpv16._hiv_module() is None


# --------------------------------------------------------------------------- #
# HIV_transmit: network transmission
# --------------------------------------------------------------------------- #

def test_hiv_transmission():
    """beta is keyed to the SexualNetwork name as [f2m, m2f] (p1=female, p2=male)."""
    h = hpv.HIV_transmit(beta_m2f=0.0035, rel_beta_f2m=0.5)
    sim = hpv.Sim(**_tiny(genotypes=[16], diseases=[h]))
    sim.init()
    net = [n for n in sim.networks.values() if isinstance(n, SexualNetwork)][0]
    betamap = sim.diseases.hiv.validate_beta()
    f2m, m2f = betamap[net.name]
    assert np.isclose(m2f, 0.0035)
    assert np.isclose(f2m, 0.0035 * 0.5)


# --------------------------------------------------------------------------- #
# HIV_incidence: curve-driven infection + plain sti.ART
# --------------------------------------------------------------------------- #

def test_hiv_incidence():
    """HIV_incidence builds a plausible epidemic from a real incidence curve,
    infected agents get a CD4 trajectory, and plain sti.ART (no hiv_art
    shortcut class) both treats a plausible fraction of HIV+ agents and
    reconstitutes their CD4 -- the ti_art=ti+1 fix (see hiv.py) is what makes
    this work with no separate testing-cascade intervention."""
    data = hpv.data.load_hiv_data(_RWANDA_HIV_DATA)
    hiv = hpv.HIV_incidence(incidence=data['incidence'])
    art = sti.ART(coverage=hpv.data.reshape_art_coverage(data['art_coverage']))
    sim = hpv.Sim(location='rwanda', rand_seed=0, genotypes=[16, 18], n_agents=2000,
                  start=1985, stop=2015, dt=0.5, diseases=[hiv], interventions=[art])
    sim.run()
    hivmod = sim.diseases.hiv
    infected = hivmod.infected.uids
    assert len(infected) > 0, 'HIV_incidence infected nobody'
    assert np.isfinite(hivmod.cd4[infected]).any(), 'set_prognoses not wired'

    on_art = infected[hivmod.on_art[infected]]
    off_art = infected[~hivmod.on_art[infected]]
    assert len(on_art) > 0, 'plain sti.ART treated nobody -- ti_art=ti+1 fix may be broken'
    assert len(off_art) > 0
    assert np.nanmean(hivmod.cd4[on_art]) > np.nanmean(hivmod.cd4[off_art]), \
        'CD4 not reconstituted for on-ART agents'

    # Bare sti.ART() forces the ti_art-exact-match path; stratified coverage
    # above only exercises `diagnosed`, not `ti_art`.
    hiv2 = hpv.HIV_incidence(incidence=data['incidence'])
    sim2 = hpv.Sim(location='rwanda', rand_seed=0, genotypes=[16, 18], n_agents=2000,
                  start=1985, stop=2015, dt=0.5, diseases=[hiv2], interventions=[sti.ART()])
    sim2.run()
    assert sim2.diseases.hiv.on_art.uids.__len__() > 0, \
        'bare sti.ART() (exact-match-only path) treated nobody -- ti_art=ti+1 fix broken'


def test_hiv_transmit_auto_diagnoses():
    """HIV_transmit (network transmission, no infect() override, no self-
    diagnose of its own) must still get its agents diagnosed and treated --
    this is HIV.step_state()'s job, shared by both HIV_transmit and
    HIV_incidence, restoring the deleted hiv_art intervention's old
    behavior of diagnosing every living HIV+ agent each step."""
    sim = hpv.Sim(n_agents=2000, start=2000, stop=2020, dt=0.5, location='nigeria',
                  genotypes=[16], rand_seed=0,
                  diseases=[hpv.HIV_transmit(beta_m2f=0.05)],
                  interventions=[sti.ART()])
    sim.run()
    hivmod = sim.diseases.hiv
    infected = hivmod.infected.uids
    assert len(infected) > 0, 'HIV_transmit infected nobody -- check beta_m2f/network'
    assert hivmod.diagnosed[infected].any(), \
        'HIV_transmit agents never got diagnosed -- auto-diagnose fix missing'
    assert hivmod.on_art.uids.__len__() > 0, \
        'HIV_transmit + bare sti.ART() treated nobody -- auto-diagnose fix missing'


def test_hiv_data_loading():
    """load_hiv_data() + reshape_art_coverage() shapes."""
    data = hpv.data.load_hiv_data(_RWANDA_HIV_DATA)
    for key in ('incidence', 'art_coverage', 'init_prev'):
        assert key in data
    assert set(data['incidence'].columns) == {'age', 'sex', 'year', 'incidence'}
    assert set(data['art_coverage'].columns) == {'age', 'sex', 'year', 'coverage'}
    assert isinstance(data['init_prev'], float)

    reshaped = hpv.data.reshape_art_coverage(data['art_coverage'])
    assert set(reshaped.columns) == {'Year', 'Gender', 'AgeBin', 'p_art'}
    assert reshaped['p_art'].between(0, 1).all()

    with pytest.raises(ValueError, match='missing'):
        hpv.data.load_hiv_data(Path(__file__).parent)  # no HIV CSVs here


# --------------------------------------------------------------------------- #
# Effect on HPV/cancer outcomes
# --------------------------------------------------------------------------- #

def test_hiv_effect():
    """Give half the population HIV and test if they develop more cancers."""
    n_agents = 4000
    incidence = pd.DataFrame({'age': [20, 20], 'sex': ['f', 'm'], 'year': [1990, 1990],
                               'incidence': [0.0, 0.0]})
    sim = hpv.Sim(n_agents=n_agents, start=1990, stop=2020, dt=0.5, location='nigeria',
                  genotypes=[16], rand_seed=0, ms_agent_ratio=5,
                  diseases=[hpv.HIV_incidence(incidence=incidence)])
    sim.init()
    females = sim.people.female.uids
    half = ss.uids(females[::2])
    sim.diseases.hiv.infected[half] = True
    sim.diseases.hiv.cd4[half] = 100.0  # lo stratum
    sim.diseases.hiv.step_state()
    sim.diseases.hpv16.set_prognoses(females)
    sim.run()

    res = sim.results.hiv
    for key in ('cancers_with_hiv', 'cancers_no_hiv', 'cancer_incidence_with_hiv',
                'cancer_incidence_no_hiv', 'hpv_prevalence_with_hiv',
                'hpv_prevalence_no_hiv', 'cancer_rate_ratio'):
        assert key in res
    assert np.all((res['hpv_prevalence_with_hiv'] >= 0) & (res['hpv_prevalence_with_hiv'] <= 1))
    # Per-capita incidence, not raw counts: at ms_agent_ratio=5 the HIV- group
    # accumulates far more (lower-weight, fine) agents via demographic growth
    # than the fixed HIV+ cohort, so raw cancers_no_hiv can exceed
    # cancers_with_hiv even though per-capita risk is ~20x higher with HIV.
    assert res['cancer_incidence_with_hiv'].sum() > res['cancer_incidence_no_hiv'].sum()

    # HIV-stratified cancers are a lower bound on the (scale-weighted) total --
    # HIVStratifiedResults' successor runs after step_die, so an agent who
    # turns cancerous and dies from background demographics the same step is
    # counted by all_hpv.new_cancers but missed here.
    assert np.issubdtype(res['cancers_with_hiv'].dtype, np.floating)
    strat_total = res['cancers_with_hiv'].sum() + res['cancers_no_hiv'].sum()
    assert strat_total > 0
    assert strat_total <= sim.results.all_hpv.new_cancers.sum() * 1.001  # float tolerance


def test_hiv_rel_imm_effect():
    """Clearance-conferred AND vaccine/txvx-conferred immunity are both
    reduced by hiv_rel_imm for HIV+ agents.

    Clearance check is deterministic: two sims share a rand_seed and
    manipulate the same female uid identically except HIV status. Starsim's
    per-agent draws are CRN-keyed by uid, so with sero_prob=1.0 the only
    difference in conferred immunity is the HIV rel_imm factor.
    """
    def _build(seed, make_positive):
        sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25, location='nigeria',
                      rand_seed=seed, genotypes=[16],
                      genotype_pars={'hpv16': {'sero_prob': 1.0}},
                      diseases=[hpv.HIV_transmit(beta_m2f=0.0)])
        sim.init()
        hpvmod = sim.diseases.hpv16
        hivmod = sim.diseases.hiv
        uid = sim.people.auids[sim.people.female[sim.people.auids]][0]
        if make_positive:
            hivmod.infected[uid] = True
            hivmod.cd4[uid] = 100.0  # lo stratum
        hivmod.step_state()  # populate hiv_rel_imm before clearance reads it
        hpvmod.infected[uid] = True
        hpvmod.precin[uid] = True
        hpvmod.cin[uid] = False
        hpvmod.cancerous[uid] = False
        hpvmod.nab_imm[uid] = 0.0
        hpvmod.cell_imm[uid] = 0.0
        hpvmod.ti_clearance[uid] = sim.ti
        hpvmod.step_state()
        return float(hpvmod.nab_imm[uid]), float(hpvmod.cell_imm[uid])

    nab_neg, cell_neg = _build(seed=1, make_positive=False)
    nab_pos, cell_pos = _build(seed=1, make_positive=True)
    assert nab_neg > 0 and cell_neg > 0
    factor = hpv.HIV_transmit().pars.rel_imm_lo
    assert np.isclose(nab_pos, nab_neg * factor, rtol=1e-6)
    assert np.isclose(cell_pos, cell_neg * factor, rtol=1e-6)

    # Vaccine / txvx immunity: an HIV+ (lo stratum) agent gets strictly less
    # conferred immunity than an otherwise-identical HIV- agent.
    def _coinfection_sim(product):
        interventions = [ss.treat_num(product=product, prob=0.0)]
        sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25, location='nigeria',
                      genotypes=[16, 18], diseases=[hpv.HIV_transmit(beta_m2f=0.0)],
                      interventions=interventions)
        sim.init()
        hivmod = sim.diseases.hiv
        females = sim.people.auids[sim.people.female[sim.people.auids]]
        uid_pos, uid_neg = females[0], females[1]
        hivmod.infected[uid_pos] = True
        hivmod.cd4[uid_pos] = 100.0
        hivmod.step_state()
        return sim, uid_pos, uid_neg, sim.interventions[0].product

    for product, imm_attr in ((hpv_vx(name='nonavalent', sterilizing_p=1.0), 'vax_imm'),
                              (hpv_txvx(name='txvx1', sterilizing_p=1.0), 'txvx_imm')):
        sim, uid_pos, uid_neg, product = _coinfection_sim(product)
        product.administer(sim.people, np.array([uid_pos, uid_neg]))
        hpvmod = sim.diseases.hpv16
        assert float(getattr(hpvmod, imm_attr)[uid_pos]) < float(getattr(hpvmod, imm_attr)[uid_neg])


def test_hiv_reactivation_effect():
    """rel_reactivation defaults to a no-op (1.0) with no HIV+ agents, and can
    be overridden independently of the other three effects."""
    sim = hpv.Sim(**_tiny(genotypes=[16], diseases=[
        hpv.HIV_transmit(beta_m2f=0.0, pars=dict(rel_reactivation_lo=5.0, rel_reactivation_hi=7.0))]))
    sim.init()
    hivmod = sim.diseases.hiv
    uid_lo, uid_hi = sim.people.auids[0], sim.people.auids[1]
    hivmod.infected[uid_lo] = True
    hivmod.cd4[uid_lo] = 100.0  # lo stratum
    hivmod.infected[uid_hi] = True
    hivmod.cd4[uid_hi] = 350.0  # hi stratum
    hivmod.step_state()
    assert np.isclose(hivmod.hiv_rel_reactivation[uid_lo], 5.0)
    assert np.isclose(hivmod.hiv_rel_reactivation[uid_hi], 7.0)
    assert np.isclose(hivmod.pars.rel_sus_lo, 2.2)  # untouched effects stay default


# --------------------------------------------------------------------------- #
# model_hiv= sim assembly
# --------------------------------------------------------------------------- #

def test_model_hiv_sim_assembly():
    """model_hiv=True/'incidence'/'transmission' construct the right pieces,
    mutual exclusivity with a user-supplied HIV disease raises, nothing is
    auto-wired without model_hiv=, and a raw sti.HIV (vignette B) does NOT get
    the CD4-effect on HPV while hpv.HIV_transmit does."""
    data = dict(
        incidence=pd.DataFrame({'age': [20, 20], 'sex': ['f', 'm'], 'year': [2000, 2000],
                                 'incidence': [0.01, 0.01]}),
        art_coverage=pd.DataFrame({'age': [20, 20], 'sex': ['f', 'm'], 'year': [2000, 2000],
                                    'coverage': [0.5, 0.5]}),
    )
    sim = hpv.Sim(**_tiny(genotypes=[16], model_hiv=True, hiv_data=data))
    sim.init()
    assert isinstance(sim.diseases.hiv, hpv.HIV_incidence)
    assert any(isinstance(iv, sti.ART) for iv in sim.interventions.values())

    sim_t = hpv.Sim(**_tiny(genotypes=[16], model_hiv='transmission'))
    sim_t.init()
    assert isinstance(sim_t.diseases.hiv, hpv.HIV_transmit)
    assert not any(isinstance(iv, sti.ART) for iv in sim_t.interventions.values())

    with pytest.raises(ValueError, match="requires hiv_data"):
        hpv.Sim(**_tiny(genotypes=[16], model_hiv=True))

    with pytest.raises(ValueError, match='mutually exclusive'):
        hpv.Sim(**_tiny(genotypes=[16], model_hiv=True, hiv_data=data,
                        diseases=[hpv.HIV_transmit(beta_m2f=0.0)]))

    sim_no_hiv = hpv.Sim(**_tiny(genotypes=[16]))
    sim_no_hiv.init()
    assert not any(isinstance(d, hpv.HIV_transmit) for d in sim_no_hiv.diseases.values())

    # Vignette B: raw sti.HIV gets no HPV-modulation effect.
    sim_raw = hpv.Sim(**_tiny(genotypes=[16], diseases=[sti.HIV(beta_m2f=0.02)],
                              interventions=[sti.HIVTest(), sti.ART()]))
    sim_raw.init()
    hpvmod = sim_raw.diseases.hpv16
    hivmod = sim_raw.diseases.hiv
    uid = sim_raw.people.auids[0]
    hivmod.infected[uid] = True
    hivmod.cd4[uid] = 50.0
    hpvmod.rel_sus[sim_raw.people.auids] = 1.0
    hpvmod.step_state()
    assert np.isclose(hpvmod.rel_sus[uid], 1.0)  # no cross-effect for a vanilla sti.HIV
