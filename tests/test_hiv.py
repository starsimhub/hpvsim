"""HIV-HPV co-infection: HIV_transmit (network-driven) and HIV_incidence
(curve-driven), both built on the shared hpv.HIV base that owns the
CD4-stratified HPV-modulation effects and HIV-stratified results.
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
    infected agents get a CD4 trajectory, and plain sti.ART treats a plausible
    fraction of them and reconstitutes their CD4 with no separate
    testing-cascade intervention."""
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
    assert len(on_art) > 0, 'plain sti.ART treated nobody: ART start must be scheduled for the step after diagnosis'
    assert len(off_art) > 0
    assert np.nanmean(hivmod.cd4[on_art]) > np.nanmean(hivmod.cd4[off_art]), \
        'CD4 not reconstituted for on-ART agents'

    # Bare sti.ART() forces the ti_art exact-match path, unlike stratified coverage.
    hiv2 = hpv.HIV_incidence(incidence=data['incidence'])
    sim2 = hpv.Sim(location='rwanda', rand_seed=0, genotypes=[16, 18], n_agents=2000,
                  start=1985, stop=2015, dt=0.5, diseases=[hiv2], interventions=[sti.ART()])
    sim2.run()
    assert sim2.diseases.hiv.on_art.uids.__len__() > 0, \
        'bare sti.ART() treated nobody: it only treats agents whose ti_art equals the current step'

def test_hiv_transmit_auto_diagnoses():
    """HIV_transmit's agents get diagnosed and treated even though it does no
    self-diagnosis: HIV.step_state() diagnoses every living HIV+ agent each step."""
    sim = hpv.Sim(n_agents=2000, start=2000, stop=2020, dt=0.5, location='nigeria',
                  genotypes=[16], rand_seed=0,
                  diseases=[hpv.HIV_transmit(beta_m2f=0.05)],
                  interventions=[sti.ART()])
    sim.run()
    hivmod = sim.diseases.hiv
    infected = hivmod.infected.uids
    assert len(infected) > 0, 'HIV_transmit infected nobody -- check beta_m2f/network'
    assert hivmod.diagnosed[infected].any(), \
        'HIV_transmit agents never got diagnosed: HIV.step_state() must diagnose living HIV+ agents each step'
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
    """Inherited stisim HIV stocks are recomputed with per-agent scale.

    stisim.utils.count is np.count_nonzero, so under grow-multiscale a fine
    agent (scale = 1/ratio) is counted as a whole person and the result is then
    multiplied by pop_scale, over-reporting every HIV stock. hpv.HIV overrides
    update_results to fix the all-age quantities.
    """
    sim = hpv.Sim(location='nigeria', genotypes=[16], n_agents=1000, rand_seed=0,
                  start=1990, stop=2000, dt=1.0, ms_agent_ratio=10,
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
    assert np.isclose(float(r['n_infected'][-1]), wtd * scale, rtol=0.02)
    # The uncorrected value would have been the raw count x pop_scale.
    assert not np.isclose(float(r['n_infected'][-1]), raw * scale, rtol=0.02)

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
    sim.diseases.hiv.infected[half] = True
    sim.diseases.hiv.cd4[half] = 100.0  # lo stratum
    sim.diseases.hiv.step_state()
    sim.diseases.hpv16.set_prognoses(females)
    sim.run()

    res = sim.results.hiv
    all_hpv = sim.results.all_hpv
    for key in ('hpv_prevalence_with_hiv', 'hpv_prevalence_no_hiv'):
        assert key in res
    for key in ('cancers_with_hiv', 'cancers_no_hiv', 'cancer_incidence_with_hiv',
                'cancer_incidence_no_hiv', 'cancer_rate_ratio'):
        assert key in all_hpv
    assert np.all((res['hpv_prevalence_with_hiv'] >= 0) & (res['hpv_prevalence_with_hiv'] <= 1))
    # Per-capita, not raw counts: the growing HIV- group outnumbers the fixed HIV+ cohort.
    assert all_hpv['cancer_incidence_with_hiv'].sum() > all_hpv['cancer_incidence_no_hiv'].sum()

    # Stratified cancers are a lower bound: stratification runs after step_die,
    # so an agent who turns cancerous and dies the same step is missed here.
    assert np.issubdtype(all_hpv['cancers_with_hiv'].dtype, np.floating)
    strat_total = all_hpv['cancers_with_hiv'].sum() + all_hpv['cancers_no_hiv'].sum()
    assert strat_total > 0
    assert strat_total <= all_hpv.new_cancers.sum() * 1.001  # float tolerance

def test_hiv_cancer_rates_are_annual_and_female():
    """HIV-stratified cancer rates use a female denominator and are annual.

    Cervical cancer only occurs in females, so an all-sex denominator would
    both deflate the rates and bias the rate ratio (HIV prevalence is
    sex-skewed). Rates are also per calendar year, not per timestep.
    """
    sim = hpv.Sim(location='nigeria', genotypes=[16, 18], n_agents=3000, rand_seed=0,
                  start=1990, stop=2005, dt=0.25, model_hiv='transmission', verbose=0)
    sim.run()
    r = sim.results.all_hpv
    years = np.floor(sim.results.timevec.years).astype(int)

    # Recompute independently for one complete year, from the raw accumulators.
    total = sim.analyzers.all_hpv
    y = 2003
    mask = years == y
    expected = {}
    for key, (counts, heads) in {
        'with_hiv': (total._cancers_with_hiv, total._females_with_hiv),
        'no_hiv': (total._cancers_no_hiv, total._females_no_hiv),
    }.items():
        pyrs = heads[mask].mean()
        expected[key] = counts[mask].sum() / pyrs * 1e5 if pyrs else 0.0
    i = np.where(mask)[0][0]
    assert np.isclose(r['cancer_incidence_with_hiv'][i], expected['with_hiv'])
    assert np.isclose(r['cancer_incidence_no_hiv'][i], expected['no_hiv'])

    # The denominator is females only, so it never exceeds the female headcount.
    assert (total._females_with_hiv + total._females_no_hiv).max() <= \
        total._females_by_who_bin.sum(axis=1).max() * 1.001

    # Every ti in a calendar year carries that year's rate (as for the ASR).
    assert len(np.unique(r['cancer_incidence_with_hiv'][mask])) == 1

    # Rate ratio is the ratio of the two rates, or nan where undefined.
    with np.errstate(invalid='ignore'):
        ok = r['cancer_incidence_no_hiv'] > 0
        assert np.allclose(r['cancer_rate_ratio'][ok],
                           (r['cancer_incidence_with_hiv'] / r['cancer_incidence_no_hiv'])[ok])
    assert np.isnan(r['cancer_rate_ratio'][~ok]).all()

def test_crude_cancer_incidence():
    """cancer_incidence is the unstandardized companion to the ASR: same
    female numerator and denominator, no WHO 2000 age weighting."""
    sim = hpv.Sim(location='nigeria', genotypes=[16], n_agents=2000, rand_seed=0,
                  start=1990, stop=2005, dt=0.25, verbose=0)
    sim.run()
    r = sim.results.all_hpv
    total = sim.analyzers.all_hpv
    years = np.floor(sim.results.timevec.years).astype(int)

    assert 'cancer_incidence' in r
    assert (r['cancer_incidence'] >= 0).all()
    mask = years == 2003
    expected = (total._cancers_by_who_bin[mask].sum()
                / total._females_by_who_bin.sum(axis=1)[mask].mean() * 1e5)
    assert np.isclose(r['cancer_incidence'][np.where(mask)[0][0]], expected)
    # Crude and standardized rates differ, since Zambia/Nigeria are younger
    # than the WHO 2000 standard population, but both are the same order.
    assert r['cancer_incidence'].max() > 0
    assert r['asr_cancer_incidence'].max() > 0

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

    # Prophylactic vaccine immunity: HIV+ (lo stratum) gets strictly less than HIV-.
    # The therapeutic is not scaled by HIV, matching v2.
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

    sim, uid_pos, uid_neg, product = _coinfection_sim(hpv_vx(name='nonavalent', sterilizing_p=1.0))
    product.administer(sim.people, np.array([uid_pos, uid_neg]))
    hpvmod = sim.diseases.hpv16
    assert float(hpvmod.vax_imm[uid_pos]) < float(hpvmod.vax_imm[uid_neg])

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
    combining model_hiv= with a user-supplied HIV disease raises, nothing is
    auto-wired without model_hiv=, and a raw sti.HIV gets no CD4-effect on HPV
    while hpv.HIV_transmit does."""
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
