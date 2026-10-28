"""HIV-HPV co-infection: STIsim's HIV, driven by transmission (HIV_transmit)
or an incidence curve (HIV_incidence), and its CD4-stratified effects on HPV
and HPV results by HIV status (hpv.HIVEffects).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starsim as ss

# stisim is the optional [hiv] extra; skip the file rather than fail collection.
sti = pytest.importorskip('stisim')

import hpvsim as hpv
from hpvsim.network import SexualNetwork
from hpvsim.products import vx as hpv_vx

_RWANDA_HIV_DATA = Path(__file__).parent / 'regression' / 'data' / 'hiv' / 'rwanda'

def _tiny(**kw):
    return dict(n_agents=300, start=2000, stop=2001, dt=0.25, location='nigeria', **kw)

def _make_latent(hivmod, uids):
    """Infect these agents with HIV, in latent infection"""
    hivmod.set_prognoses(uids)
    hivmod.acute[uids] = False
    hivmod.latent[uids] = True

# --------------------------------------------------------------------------- #
# Wiring: gating
# --------------------------------------------------------------------------- #

def test_hiv_gating():
    """HPV._hiv_effects() finds HIV's effects (added with the HIV disease) when present, None otherwise;
    the CD4-stratified factors default to a no-op (1.0) with no HIV+ agents."""
    sim = hpv.Sim(**_tiny(genotypes=[16], diseases=[hpv.HIV_transmit(beta_m2f=0.0)]))
    sim.init()
    hpvmod = sim.diseases.hpv16
    assert hpvmod._hiv_effects() is sim.connectors.hiv_effects
    assert np.allclose(sim.connectors.hiv_effects.hiv_rel_sev.values[:5], 1.0)

    sim_no_hiv = hpv.Sim(**_tiny(genotypes=[16]))
    sim_no_hiv.init()
    assert sim_no_hiv.diseases.hpv16._hiv_effects() is None

# --------------------------------------------------------------------------- #
# HIV_transmit: network transmission
# --------------------------------------------------------------------------- #

def test_hiv_transmission():
    """HIV transmits per act on the SexualNetwork, at beta_m2f (and beta_m2f*rel_beta_f2m from women to men)."""
    h = hpv.HIV_transmit(beta_m2f=0.0035, rel_beta_f2m=0.5)
    assert np.isclose(h.pars.beta_m2f, 0.0035)
    assert np.isclose(h.pars.rel_beta_f2m, 0.5)
    assert 'sexualnetwork' in h.hetero_networks
    infections = []
    for beta in [0.0, 0.2]:
        sim = hpv.Sim(**_tiny(genotypes=[16], diseases=[hpv.HIV_transmit(beta_m2f=beta, init_prev=0.2)], rand_seed=1))
        sim.run()
        infections.append(sim.results.hiv.new_infections_sex.values.sum())
    assert infections[0] == 0, 'HIV transmitted with beta_m2f=0'
    assert infections[1] > 0, 'HIV did not transmit on the SexualNetwork'

# --------------------------------------------------------------------------- #
# HIV_incidence: curve-driven infection + plain sti.ART
# --------------------------------------------------------------------------- #

def test_hiv_incidence():
    """HIV_incidence builds a plausible epidemic from a real incidence curve,
    infected agents get a CD4 trajectory, and plain sti.ART treats a plausible
    fraction of them (once a test diagnoses everyone with HIV each step) and
    reconstitutes their CD4."""
    data = hpv.data.load_hiv_data(_RWANDA_HIV_DATA)
    hiv = hpv.HIV_incidence(incidence=data['incidence'])
    dx = sti.HIVTest(test_prob_data=1.0, dt_scale=False)
    art = sti.ART(coverage=hpv.data.reshape_art_coverage(data['art_coverage']))
    sim = hpv.Sim(location='rwanda', rand_seed=0, genotypes=[16, 18], n_agents=2000,
                  start=1985, stop=2015, dt=0.5, diseases=[hiv], interventions=[dx, art])
    sim.run()
    hivmod = sim.diseases.hiv
    infected = hivmod.infected.uids
    assert len(infected) > 0, 'HIV_incidence infected nobody'
    assert np.isfinite(hivmod.cd4[infected]).any(), 'CD4 not set at infection'

    on_art = infected[hivmod.on_art[infected]]
    off_art = infected[~hivmod.on_art[infected]]
    assert len(on_art) > 0, 'plain sti.ART treated nobody: ART start must be scheduled for the step after diagnosis'
    assert len(off_art) > 0
    years_on_art = (hivmod.ti - hivmod.ti_art[on_art])*hivmod.t.dt_year
    settled = on_art[years_on_art >= 1] # On ART for a year or more (those ART put on it last are the lowest CD4 counts)
    assert len(settled) > 0
    assert np.mean(hivmod.cd4[settled]) > np.mean(hivmod.cd4_preart[settled]), \
        'CD4 not reconstituted for on-ART agents'

    # Bare sti.ART() forces the ti_art exact-match path, unlike stratified coverage.
    hiv2 = hpv.HIV_incidence(incidence=data['incidence'])
    sim2 = hpv.Sim(location='rwanda', rand_seed=0, genotypes=[16, 18], n_agents=2000,
                  start=1985, stop=2015, dt=0.5, diseases=[hiv2],
                  interventions=[sti.HIVTest(test_prob_data=1.0, dt_scale=False), sti.ART()])
    sim2.run()
    assert sim2.diseases.hiv.on_art.uids.__len__() > 0, \
        'bare sti.ART() treated nobody: it only treats agents whose ti_art equals the current step'

def test_hiv_transmit_auto_diagnoses():
    """With model_hiv=, agents get diagnosed and treated with no testing cascade
    given: hpv.Sim adds an sti.HIVTest that diagnoses everyone with HIV each step."""
    sim = hpv.Sim(n_agents=2000, start=2000, stop=2020, dt=0.5, location='nigeria',
                  genotypes=[16], rand_seed=0, model_hiv='transmission',
                  hiv_pars=dict(beta_m2f=0.05), interventions=[sti.ART()])
    sim.run()
    hivmod = sim.diseases.hiv
    infected = hivmod.infected.uids
    assert len(infected) > 0, 'HIV_transmit infected nobody -- check beta_m2f/network'
    assert hivmod.diagnosed[infected].any(), \
        'HIV_transmit agents never got diagnosed: model_hiv= adds a test that diagnoses living HIV+ agents each step'
    assert hivmod.on_art.uids.__len__() > 0, \
        'HIV_transmit + bare sti.ART() treated nobody: agents must be diagnosed before ART can treat them'

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

def test_hiv_results_are_scale_weighted():
    """STIsim's HIV results count each agent by its scale.

    Under grow-multiscale a fine agent (scale = 1/ratio) counted as a whole
    person would over-report every HIV stock once multiplied by pop_scale;
    STIsim counts each agent by its weight when weights differ.
    """
    sim = hpv.Sim(location='nigeria', genotypes=[16], n_agents=2000, rand_seed=1, # Seed 1: 4 infected fine agents (about 7 on average)
                  start=1990, stop=2005, dt=1.0, ms_agent_ratio=10,
                  model_hiv='transmission', verbose=0)
    sim.run()
    r, ppl, hiv = sim.results.hiv, sim.people, sim.diseases.hiv
    w, alive = ppl.scale.values, ppl.alive.values
    infected = hiv.infected.values & alive
    scale = sim.pars.pop_scale

    # Multiscale actually engaged, or the test proves nothing.
    assert (w < 1).any(), 'no fine agents; ms_agent_ratio had no effect'

    wtd = float((w * infected).sum())
    raw = float(infected.sum())
    assert wtd < raw, 'fine agents should weigh less than whole people'
    assert np.isclose(float(r['infected'][-1]), wtd * scale, rtol=0.02)
    # The uncorrected value would have been the raw count x pop_scale. The two
    # differ by the share of the infected who are fine agents, x0.9: about 2% on
    # average, and at least 0.25% once any fine agent is infected (0.3% the least
    # in 40 seeds), so 0.1% fails only with none infected (about 1 in 1,000 seeds).
    assert not np.isclose(float(r['infected'][-1]), raw * scale, rtol=0.001)

    # Prevalence is a scale-weighted fraction, and stays a fraction.
    n_alive = float((w * alive).sum())
    assert np.isclose(float(r['prevalence'][-1]), wtd / n_alive, rtol=0.02)
    assert 0 <= r['prevalence'][-1] <= 1
    assert 0 <= r['prevalence_15_49'][-1] <= 1
    # Cumulative flows are rebuilt from the corrected per-step series.
    assert np.isclose(float(r['cum_infections'][-1]),
                      float(np.sum(r['new_infections'])), rtol=1e-6)

def test_hiv_prevalence_by_sex():
    """prevalence_f/_m are scale-weighted adult (15-49) HIV prevalence
    fractions that weight back to prevalence_15_49."""
    sim = hpv.Sim(location='nigeria', genotypes=[16], n_agents=1000, rand_seed=0,
                  start=1990, stop=2000, dt=1.0, ms_agent_ratio=10,
                  model_hiv='transmission', verbose=0)
    sim.run()
    r, ppl = sim.results.hiv, sim.people
    w, alive = ppl.scale.values, ppl.alive.values
    female, age = ppl.female.values, ppl.age.values
    adult = (age >= 15) & (age < 50)

    assert (w < 1).any(), 'no fine agents; ms_agent_ratio had no effect'
    for key in ('prevalence_f', 'prevalence_m'):
        assert ((r[key] >= 0) & (r[key] <= 1)).all()

    n_f = float((w * (alive & female & adult)).sum())
    n_m = float((w * (alive & ~female & adult)).sum())
    combined = (r['prevalence_f'][-1] * n_f + r['prevalence_m'][-1] * n_m) / (n_f + n_m)
    assert np.isclose(combined, float(r['prevalence_15_49'][-1]), rtol=0.02)

def test_hiv_banded_age_data():
    """Age-banded HIV inputs work, not just single years of age.

    UNAIDS/Spectrum outputs come in 5-year bands, so both the incidence
    lookup and the ART reshape must bucket by the bands the data supplies.
    """
    ages = [0, 5, 10, 15, 20]
    inc = pd.DataFrame([(a, s, y, a / 1000) for a in ages for s in 'fm'
                        for y in (1990, 2000)],
                       columns=['age', 'sex', 'year', 'incidence'])
    hiv = hpv.HIV_incidence(incidence=inc)
    sim = hpv.Sim(location='rwanda', rand_seed=0, genotypes=[16], n_agents=500,
                  start=1990, stop=1995, dt=1.0, diseases=[hiv])
    sim.run()  # would IndexError if ages were treated as single years

    # Each agent gets the rate of the band its age falls into. Read the sim's
    # own module, not the local one -- copy_inputs= deep-copies it on the way in.
    mod = sim.diseases.hiv
    mod._rate_cube = {0: np.array([[0, .05, .10, .15, .20]] * 2),
                      1: np.array([[0, .05, .10, .15, .20]] * 2)}
    probe = np.array([0., 4.9, 5., 9., 14., 19., 20., 99.])
    rates = mod._lookup_rates(1990, probe, np.ones(len(probe), bool))
    assert np.allclose(rates, [0, 0, .05, .05, .10, .15, .20, .20])

    # Nearest-year: a year absent from the data resolves to the closest one.
    assert np.allclose(mod._lookup_rates(1991, probe, np.ones(len(probe), bool)), rates)

    # Band upper edges come from the data, not age+1; top band is open-ended.
    art = pd.DataFrame({'age': ages, 'sex': 'f', 'year': 2010,
                        'coverage': [.1, .2, .3, .4, .5]})
    bins = hpv.data.reshape_art_coverage(art)['AgeBin'].tolist()
    assert bins == ['[0,5)', '[5,10)', '[10,15)', '[15,20)', '[20,150)']

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
    _make_latent(sim.diseases.hiv, half)
    sim.diseases.hiv.cd4_latent[half] = 100.0  # lo stratum
    sim.diseases.hiv.step_state()
    sim.diseases.hpv16.set_prognoses(females)
    sim.run()

    res = sim.results.hiv_effects
    all_hpv = sim.results.all_hpv
    for key in ('hpv_prevalence_with_hiv', 'hpv_prevalence_no_hiv'):
        assert key in res
    for key in ('new_cancers_with_hiv', 'new_cancers_no_hiv', 'cancer_incidence_with_hiv',
                'cancer_incidence_no_hiv', 'cancer_rate_ratio'):
        assert key in all_hpv
    assert np.all((res['hpv_prevalence_with_hiv'] >= 0) & (res['hpv_prevalence_with_hiv'] <= 1))
    # Per-capita, not raw counts: the growing HIV- group outnumbers the fixed HIV+ cohort.
    assert all_hpv['cancer_incidence_with_hiv'].sum() > all_hpv['cancer_incidence_no_hiv'].sum()

    # Stratified cancers are a lower bound: stratification runs after step_die,
    # so an agent who turns cancerous and dies the same step is missed here.
    assert np.issubdtype(all_hpv['new_cancers_with_hiv'].dtype, np.floating)
    strat_total = all_hpv['new_cancers_with_hiv'].sum() + all_hpv['new_cancers_no_hiv'].sum()
    assert strat_total > 0
    assert strat_total <= all_hpv.new_cancers.sum() * 1.001  # float tolerance

def test_hiv_cancer_rates_are_annual_and_female():
    """HIV-stratified cancer rates: annualized per-ti rates with female
    denominator. Rate ratio is nan where the HIV- rate is zero."""
    sim = hpv.Sim(location='nigeria', genotypes=[16, 18], n_agents=3000, rand_seed=0,
                  start=1990, stop=2005, dt=0.25, model_hiv='transmission', verbose=0)
    sim.run()
    r = sim.results.all_hpv

    # Non-negative rates; some cancers must occur.
    for k in ('cancer_incidence_with_hiv', 'cancer_incidence_no_hiv'):
        assert (r[k] >= 0).all()
    assert (r['new_cancers_with_hiv'] + r['new_cancers_no_hiv']).sum() > 0

    # annualize() on the per-ti annualized rate collapses quarters to the
    # annual rate; a valid rate for a year with cancers should be > 0.
    ann = r['cancer_incidence_no_hiv'].resample('year')
    assert (ann.values >= 0).all()
    assert (ann.values > 0).any()

    # Rate ratio equals the ratio of the two rates, nan where HIV- is zero.
    with np.errstate(invalid='ignore'):
        ok = r['cancer_incidence_no_hiv'] > 0
        assert np.allclose(r['cancer_rate_ratio'][ok],
                           (r['cancer_incidence_with_hiv'] / r['cancer_incidence_no_hiv'])[ok])
    assert np.isnan(r['cancer_rate_ratio'][~ok]).all()

def test_crude_cancer_incidence():
    """cancer_incidence is the unstandardized companion to the ASR: same
    per-ti annualized-rate treatment, no WHO 2000 age weighting."""
    sim = hpv.Sim(location='nigeria', genotypes=[16], n_agents=2000, rand_seed=1,
                  start=1990, stop=2005, dt=0.25, verbose=0) # Seed 1: 4 cancers (seed 0 has none)
    sim.run()
    r = sim.results.all_hpv

    assert 'cancer_incidence' in r
    assert (r['cancer_incidence'] >= 0).all()
    # Both rates are per-ti annualized; annualize() then averages quarters.
    assert r['cancer_incidence'].max() > 0
    assert r['asr_cancer_incidence'].max() > 0
    # annualize is idempotent on already-annualized-per-ti rates: the annual
    # mean should be within the min/max of the sub-annual quarters.
    ann = r['cancer_incidence'].resample('year')
    per_ti = np.asarray(r['cancer_incidence'])
    assert ann.values.min() >= per_ti.min() - 1e-9
    assert ann.values.max() <= per_ti.max() + 1e-9

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
            _make_latent(hivmod, ss.uids([uid]))
            hivmod.cd4_latent[uid] = 100.0  # lo stratum
        hivmod.step_state()  # their CD4 count
        sim.connectors.hiv_effects.step()  # populate hiv_rel_imm before clearance reads it
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
    factor = hpv.HIVEffects().pars.rel_imm_lo
    assert np.isclose(nab_pos, nab_neg * factor, rtol=1e-6)
    assert np.isclose(cell_pos, cell_neg * factor, rtol=1e-6)

    # Prophylactic vaccine immunity: HIV+ (lo stratum) gets strictly less than HIV-.
    # The therapeutic is not scaled by HIV, matching v2.
    def _coinfection_sim(product):
        interventions = [ss.Delivery(product, coverage=0.0)]
        sim = hpv.Sim(n_agents=400, start=2000, stop=2001, dt=0.25, location='nigeria',
                      genotypes=[16, 18], diseases=[hpv.HIV_transmit(beta_m2f=0.0)],
                      interventions=interventions)
        sim.init()
        hivmod = sim.diseases.hiv
        females = sim.people.auids[sim.people.female[sim.people.auids]]
        uid_pos, uid_neg = females[0], females[1]
        _make_latent(hivmod, ss.uids([uid_pos]))
        hivmod.cd4_latent[uid_pos] = 100.0
        hivmod.step_state()
        sim.connectors.hiv_effects.step()
        return sim, uid_pos, uid_neg, sim.interventions[0].product

    sim, uid_pos, uid_neg, product = _coinfection_sim(hpv_vx(name='nonavalent', sterilizing_p=1.0))
    product.administer(ss.uids([uid_pos, uid_neg]))
    hpvmod = sim.diseases.hpv16
    assert float(hpvmod.vax_imm[uid_pos]) < float(hpvmod.vax_imm[uid_neg])

def test_hiv_reactivation_effect():
    """rel_reactivation defaults to a no-op (1.0) with no HIV+ agents, and can
    be overridden independently of the other three effects."""
    sim = hpv.Sim(**_tiny(genotypes=[16], diseases=[hpv.HIV_transmit(beta_m2f=0.0)],
                          connectors=[hpv.HIVEffects(rel_reactivation_lo=5.0, rel_reactivation_hi=7.0)]))
    sim.init()
    hivmod = sim.diseases.hiv
    effects = sim.connectors.hiv_effects
    uid_lo, uid_hi = (~hivmod.infected).uids[:2] # Not infected at the start, whose CD4 the HIV module sets itself
    _make_latent(hivmod, ss.uids([uid_lo, uid_hi]))
    hivmod.cd4_latent[uid_lo] = 100.0  # lo stratum
    hivmod.cd4_latent[uid_hi] = 350.0  # hi stratum
    hivmod.step_state()
    effects.step()
    assert np.isclose(effects.hiv_rel_reactivation[uid_lo], 5.0)
    assert np.isclose(effects.hiv_rel_reactivation[uid_hi], 7.0)
    assert np.isclose(effects.pars.rel_sus_lo, 2.2)  # untouched effects stay default

# --------------------------------------------------------------------------- #
# model_hiv= sim assembly
# --------------------------------------------------------------------------- #

def test_model_hiv_sim_assembly():
    """model_hiv=True/'incidence'/'transmission' construct the right pieces,
    combining model_hiv= with a user-supplied HIV disease raises, nothing is
    auto-wired without model_hiv=, and HIV's effects on HPV come with any
    HIV, a raw sti.HIV too."""
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
    assert any(isinstance(iv, sti.HIVTest) for iv in sim.interventions.values())
    assert isinstance(sim.connectors.hiv_effects, hpv.HIVEffects)

    sim_t = hpv.Sim(**_tiny(genotypes=[16], model_hiv='transmission'))
    sim_t.init()
    assert isinstance(sim_t.diseases.hiv, sti.HIV) and not isinstance(sim_t.diseases.hiv, hpv.HIV_incidence)
    assert not any(isinstance(iv, sti.ART) for iv in sim_t.interventions.values())

    with pytest.raises(ValueError, match="requires hiv_data"):
        hpv.Sim(**_tiny(genotypes=[16], model_hiv=True))

    with pytest.raises(ValueError, match='mutually exclusive'):
        hpv.Sim(**_tiny(genotypes=[16], model_hiv=True, hiv_data=data,
                        diseases=[hpv.HIV_transmit(beta_m2f=0.0)]))

    sim_no_hiv = hpv.Sim(**_tiny(genotypes=[16]))
    sim_no_hiv.init()
    assert not any(isinstance(d, sti.HIV) for d in sim_no_hiv.diseases.values())
    assert 'hiv_effects' not in sim_no_hiv.connectors

    # Vignette B: a raw sti.HIV gets HIV's effects on HPV too.
    sim_raw = hpv.Sim(**_tiny(genotypes=[16], diseases=[sti.HIV(beta_m2f=0.02)],
                              interventions=[sti.HIVTest(), sti.ART()]))
    sim_raw.init()
    hpvmod = sim_raw.diseases.hpv16
    hivmod = sim_raw.diseases.hiv
    uid = (~hivmod.infected).uids[0]
    _make_latent(hivmod, ss.uids([uid]))
    hivmod.cd4_latent[uid] = 50.0
    hivmod.step_state()
    hpvmod.step_state()  # Resets rel_sus
    sim_raw.connectors.hiv_effects.step()
    assert np.isclose(hpvmod.rel_sus[uid], sim_raw.connectors.hiv_effects.pars.rel_sus_lo)
