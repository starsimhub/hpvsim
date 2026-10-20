"""
HPV products and programs on Starsim 4: HPVsim's tables, vaccination, and screen-and-treat
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import hpvsim as hpv

kw = dict(dt=0.25, verbose=0, use_aging=True)
genos = ['hpv16', 'hpv18']


def make(interventions=None, n_agents=10_000, stop=2040, **kwargs):
    fam = hpv.genotypes(genos, init_prev=0.15)
    return ss.Sim(diseases=fam, connectors=hpv.cross_immunity(fam), networks=ss.MFNet(), interventions=interventions,
                  n_agents=n_agents, start=1990, stop=stop, **kw, **kwargs)


@pytest.mark.parametrize('name', ['via', 'hpv', 'colposcopy'])
def test_diagnostic(name):
    sc.heading(f'The test "{name}" from HPVsim\'s table: each result as often as the table says, given each woman\'s states')
    df = hpv.products.table('dx', name)
    order = hpv.dx_results[name]
    seen = sc.objdict()
    def look(sim): # The expected count of each result, from the women's states when tested
        d, u = sim.diseases, (sim.people.female & (sim.people.age >= 25)).uids
        states = {s: [d[g][s].raw[u] for g in genos] for s in ['P', 'CIN', 'CA']}
        tails = np.zeros((len(u), len(order) - 1)) # P(this result or a more severe one)
        def prob(state, g, res):
            rows = df[(df.state == state) & (df.result == res) & df.genotype.isin(['all', g])]
            return float(rows.probability.sum()) if len(rows) else 0.0
        if name == 'hpv': # By genotype: independent, the most severe of each
            miss = np.ones((len(u), len(order) - 1))
            for i, g in enumerate(genos):
                p = np.zeros((len(u), len(order) - 1))
                for s, state in [('P', 'precin'), ('CIN', 'cin'), ('CA', 'cancerous')]:
                    cum = np.cumsum([prob(state, g, r) for r in order[:-1]])
                    p = np.where(states[s][i][:, None], cum, p)
                miss *= 1 - p
            tails = 1 - miss
        else: # Any genotype: the highest over the states each woman is in, or susceptible
            none = ~np.any([x for v in states.values() for x in v], axis=0)
            for state, held in [('susceptible', [none]), ('precin', states['P']), ('cin', states['CIN']), ('cancerous', states['CA'])]:
                cum = np.cumsum([prob(state, 'all', r) for r in order[:-1]])
                tails = np.maximum(tails, np.where(np.any(held, axis=0)[:, None], cum, 0))
        tails = np.column_stack([tails, np.ones(len(u))])
        seen.expected = np.diff(np.column_stack([np.zeros(len(u)), tails]), axis=1).sum(axis=0)
        return True
    test = hpv.diagnostic(name, genos)
    prog = ss.screen(test, where=ss.ppl.female & (ss.ppl.age >= 25), at=[2010], when=look, name='prog')
    sim = make(prog, n_agents=40_000, stop=2011).run()
    k = int(np.flatnonzero(sim.results.prog.delivered)[0])
    got = np.array([sim.results.prog[r][k] for r in order])
    print(f'{name}: {dict(zip(order, got))} vs expected {dict(zip(order, seen.expected.round(1)))}')
    big = seen.expected > 200
    assert np.allclose(got[big], seen.expected[big], rtol=0.1)
    assert got.sum() == sim.results.prog.delivered[k]
    return sim


def test_vaccine():
    sc.heading('A bivalent vaccine from HPVsim\'s table, given to everyone: infections with both genotypes stop')
    vx = hpv.vaccine('bivalent', genos)
    assert vx.efficacy == pytest.approx({'hpv16': 0.9975, 'hpv18': 0.9975})
    base = make(stop=2010).run()
    sim = make(ss.vaccinate(vx, at=[2000], coverage=1.0, name='vx'), stop=2010).run()
    after = slice(int(np.flatnonzero(sim.results.vx.delivered)[0]) + 1, None)
    for g in genos:
        n, n0 = sim.results[g].infection[after].sum(), base.results[g].infection[after].sum()
        print(f'{g}: {n} infections after vaccination, against {n0}')
        assert n < 0.05*n0 # UPSTREAM: efficacy scales the hazard per act, so with ~20 acts a step it leaves ~3% (HPVsim: under 0.3%)
    with pytest.raises(ValueError, match='no efficacy'):
        hpv.vaccine('bivalent', ['hpv16', 'hpv99'])
    return sim


@pytest.mark.xfail(strict=True, raises=ss.ClockOutsideDomain, reason='UPSTREAM: with births, the antibodies of new agents are NaN, and the cross-immunity modifier raises E212')
def test_routine_vx():
    sc.heading('Routine two-dose vaccination of girls at 9, with births: the second dose 6 months after the first, and fewer infections')
    vx = hpv.vaccine('bivalent', genos, doses=[0.8, 1.0])
    demog = lambda: [ss.Births(birth_rate=25), ss.Deaths(death_rate=10)]
    base = make(demographics=demog()).run()
    sim = make(hpv.routine_vx(vx, genos, start=2000), demographics=demog()).run()
    r = sim.results
    d1, d2 = r.routine_vx.delivered, r.routine_vx_next.delivered
    first = int(np.flatnonzero(d1)[0])
    assert d2[:first + 2].sum() == 0 and np.isclose(d2.sum(), 0.9*d1[:-2].sum(), rtol=0.1)
    late = slice(-40, None)
    assert r.hpv16.infection[late].sum() < base.results.hpv16.infection[late].sum()


def test_screen_treat():
    sc.heading('Screen, triage, and treat (HPV DNA, VIA, ablation): the steps follow each other, and lesions and cancers fall')
    base = make().run()
    sim = make(hpv.screen_treat('hpv', 'via', 'ablation', genos, start=2010, coverage=ss.probperyear(0.3))).run()
    r = sim.results
    n = sc.objdict({key: r[key].delivered.sum() for key in ['screening', 'triage', 'treatment']})
    print(n, 'cured', r.treatment.cured.sum())
    assert n.screening > 1000 and r.screening.delivered[:80].sum() == 0 # From 2010
    assert n.triage == r.screening.positive.sum() and n.treatment == r.triage.positive.sum()
    assert r.treatment.cured.sum() == pytest.approx(0.936*(r.treatment.cured.sum() + r.treatment.failed.sum()), rel=0.15)
    women = r.screening.eligible[81] # Those 30-49 at the start: none screened twice in 5 years, so at most 6 times in 30 years
    assert n.screening < 6*women
    cancers = lambda s: sum(s.results[g]['cin.CA'][80:].sum() for g in genos)
    cin = lambda s: s.results.hpv.n_CIN[100:].mean()
    print(f'CIN {cin(sim):.0f} vs {cin(base):.0f}; cancers {cancers(sim)} vs {cancers(base)}')
    assert cin(sim) < 0.95*cin(base) and cancers(sim) < cancers(base) # Seeds 1-3: 0.87-0.90, and 33-38 cancers against 45-50
    return sim


if __name__ == '__main__':
    for name in ['via', 'hpv', 'colposcopy']:
        test_diagnostic(name)
    test_vaccine()
    test_screen_treat()
