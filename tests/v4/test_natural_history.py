"""
The natural history of the genotypes on Starsim 4: against HPVsim's formulas, and across genotypes
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import hpvsim as hpv

kw = dict(dt=0.25, verbose=0)


def expected_cohort(name, horizon, n=2_000_000, seed=1):
    """ HPVsim's model by Monte Carlo: P(CIN) for a woman newly infected, and P(cancer within the horizon) """
    rng = np.random.default_rng(seed)
    p = hpv.genotype_pars(name)
    def lognorm(dist):
        mean, std = dist.pars['mean'].value, dist.pars['std'].value
        s2 = np.log(1 + (std/mean)**2)
        return rng.lognormal(np.log(mean) - s2/2, np.sqrt(s2), n)
    d_p, d_c = lognorm(p['dur_precin']), lognorm(p['dur_cin'])
    cin = rng.random(n) < hpv.logf2(d_p, p['cin_k'], p['x_infl'], ttc=p['ttc'])
    p_ca = 1 - (1 - p['transform_prob'])**(hpv.intlogf2(d_c, p['cin_k'], p['x_infl'], ttc=p['ttc'])**2)
    cancer = cin & (rng.random(n) < p_ca) & (d_p + d_c <= horizon)
    return cin.mean(), cancer.mean()


@pytest.mark.parametrize('name', ['hpv16', 'hpv18'])
def test_cohort(name):
    sc.heading(f'A cohort of women newly infected with {name}: progression to CIN and cancer as in HPVsim')
    horizon = 60
    sim = ss.Sim(diseases=hpv.genotype(name, init_prev=1.0, beta=0, age_risk=1), networks=ss.MFNet(), n_agents=40_000, start=2000, stop=2000 + horizon, **kw)
    sim.init()
    women, men = sim.people.female.sum(), sim.people.male.sum()
    sim.run()
    res = sim.results[name]
    p_cin, p_cancer = res['precin.CIN'].sum()/women, res['cin.CA'].sum()/women
    exp_cin, exp_cancer = expected_cohort(name, horizon)
    print(f'{name}: P(CIN) {p_cin:.4f} vs {exp_cin:.4f}; P(cancer) {p_cancer:.4f} vs {exp_cancer:.4f}')
    assert np.isclose(p_cin, exp_cin, rtol=0.05)
    assert np.isclose(p_cancer, exp_cancer, rtol=0.2)
    assert res.clear_m.sum() == pytest.approx(men, rel=0.01) # Men clear, and only men
    return sim


def test_family():
    sc.heading('Two genotypes as a family: one cancer per woman, results over both, and cross-immunity that protects')
    def make(cross=True):
        fam = hpv.genotypes(['hpv16', 'hpv18'], init_prev=0.15)
        cons = hpv.cross_immunity(fam) if cross else []
        return ss.Sim(diseases=fam, connectors=cons, networks=ss.MFNet(), n_agents=10_000, start=1990, stop=2030, use_aging=True, **kw)
    sim = make().run()
    none = make(cross=False).run()
    r, d = sim.results, sim.diseases
    assert np.array_equal(r.hpv.n_CA, r.hpv16.n_CA + r.hpv18.n_CA) # Never cancer in both
    assert not np.any(d.hpv16.CA.raw & d.hpv18.CA.raw) and r.hpv.n_Z.max() > 0
    assert np.all(r.hpv.infected <= r.hpv16.infected + r.hpv18.infected) and np.any(r.hpv.infected < r.hpv16.infected + r.hpv18.infected) # A union
    seropos = (d.hpv16.nab.values > 0)
    assert seropos.any() and np.all(d.hpv16.nab.values[seropos] >= 0.5) and not seropos[sim.people.male.values].any() # Antibodies from women's clearance only
    for g in ['hpv16', 'hpv18']:
        assert r[g].infection.sum() < 0.95*none.results[g].infection.sum()
    return sim


def test_seeding():
    sc.heading('Seeding the family from HPVsim\'s curve by age and sex, one genotype per person seeded')
    sim = ss.Sim(diseases=hpv.genotypes(['hpv16', 'hpv18'], init_prev=hpv.init_prev()), networks=ss.MFNet(), n_agents=40_000, start=1990, stop=1991, **kw).init()
    d, ppl = sim.diseases, sim.people
    held = np.array([d[g].P.raw for g in ['hpv16', 'hpv18']])
    age, female = ppl.age.raw, ppl.female.raw
    for lo, hi, f, m in [(0, 12, 0, 0), (17, 24, 0.70, 0.60), (24, 34, 0.25, 0.25), (44, 64, 0.01, 0.01)]:
        band = (age >= lo) & (age < hi) & ppl.alive.raw
        assert np.isclose(held.any(0)[band & female].mean(), f, atol=0.03) and np.isclose(held.any(0)[band & ~female].mean(), m, atol=0.03), (lo, hi)
    assert held.all(0).sum() == 0


def test_network():
    sc.heading('On HPVsim\'s sexual network: male-to-female transmission stronger than female-to-male means more infections of women')
    def ratio(m2f):
        fam = hpv.genotypes(['hpv16'], init_prev=hpv.init_prev(), pars=dict(hpv16=dict(beta=0.02, transm2f=m2f)))
        sim = ss.Sim(diseases=fam, networks=hpv.SexualNetwork(), stratify=dict(sex=['female', 'male']), n_agents=10_000, start=1990, stop=2000, use_aging=True, **kw).run()
        women, men = sim.results.hpv16.infection.by('sex').values.sum(axis=0)
        print(f'transm2f {m2f}: infections of women {women:.0f}, of men {men:.0f}')
        return women/men
    assert ratio(4.0) > 2*ratio(0.25)


def cohort_cancers(**kwargs):
    sim = ss.Sim(diseases=hpv.genotype('hpv18', init_prev=1.0, beta=0, **kwargs), networks=ss.MFNet(), n_agents=40_000, start=2000, stop=2060, **kw).run()
    return sim.results.hpv18['cin.CA'].sum()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='UPSTREAM: a dwell distribution times an attribute (dur_cin*ramp) is drawn from another distribution, even when the attribute is 1')
def test_age_risk_dwell():
    sc.heading('age_risk: a ramp that is 1 for everyone changes nothing')
    plain, ramp = cohort_cancers(age_risk=1), cohort_cancers(age_risk=1 + 1e-9)
    print(f'Cancers {ramp} with the ramp, against {plain} without')
    assert np.isclose(ramp, plain, rtol=0.08) # Seeds 1-3: 18% fewer


@pytest.mark.xfail(strict=True, raises=ss.DuplicateDeclaration, reason='UPSTREAM: the auto-named dwell of precin is dur_precin, the name of the parameter in it')
def test_severity_immunity():
    sc.heading('Severity immunity from cell-mediated immunity: shorter infections, so fewer CIN')
    def make(sev):
        fam = hpv.genotypes(['hpv16', 'hpv18'], init_prev=0.15, sev_matrix=sev)
        return ss.Sim(diseases=fam, networks=ss.MFNet(), n_agents=10_000, start=1990, stop=2030, use_aging=True, **kw).run()
    sim, none = make('default'), make(None)
    assert sim.results.hpv['precin'].sum() > 0 and sim.results.hpv.n_CIN.sum() < none.results.hpv.n_CIN.sum()


@pytest.mark.xfail(strict=True, raises=ss.CapabilityError, reason='UPSTREAM: clocks (P.time in p=) are agents only (Starsim B1), so the natural history has no ODE')
def test_ode():
    sc.heading('The natural history as an ODE, against agents')
    fam = hpv.genotypes(['hpv16', 'hpv18'], init_prev=0.15)
    ss.Sim(diseases=fam, solver='ode', total_pop=10_000, start=1990, stop=2030, **kw).run()


if __name__ == '__main__':
    for name in ['hpv16', 'hpv18']:
        test_cohort(name)
    test_family()
