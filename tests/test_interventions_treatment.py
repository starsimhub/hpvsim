"""Integration tests for treatment: ss.treat() delivering hpv.tx and hpv.radiation, and the conditions on cancer."""
import numpy as np
import starsim as ss
import hpvsim as hpv

genotypes = ('hpv16', 'hpv18', 'hi5', 'ohr')


def _four_genotype_sim_with(intvs, analyzers=None):
    return hpv.Sim(
        n_agents=500, start=2020, stop=2025, location='nigeria',
        diseases=[hpv.HPV(g) for g in genotypes],
        interventions=intvs, analyzers=analyzers, verbose=0,
    )


def _run_with_cancers(treat, n=5):
    """Give some women cancer of hpv16 after init, run, and return the sim, those women, and the agents the treatment reached"""
    reached = []
    def track(sim):
        reached.extend(sim.interventions.rx.reached().tolist())
    sim = _four_genotype_sim_with([treat], analyzers=track)
    sim.init()
    uids = sim.people.female.uids[:n]
    sim.diseases.hpv16.cancerous[uids] = True
    sim.diseases.hpv16.ti_dead_cancer[uids] = np.inf # Not dying of it during the test
    sim.run()
    return sim, uids, set(reached)


def test_cancer_conditions_are_complements():
    sim = _four_genotype_sim_with([])
    sim.init()
    uids = sim.people.alive.uids[:5]
    sim.diseases.hpv18.cancerous[uids] = True
    has, no = np.asarray(hpv.has_cancer(sim)), np.asarray(hpv.no_cancer(sim))
    assert has[uids].all() and not no[uids].any()
    assert (has ^ no)[sim.people.alive.uids].all()


def test_treatment_excludes_cancer_with_no_cancer():
    """A CIN treatment given where=hpv.no_cancer does not reach women with cancer."""
    treat = ss.treat(hpv.tx('excision'), where=ss.ppl.female & hpv.no_cancer, coverage=1.0, at=['2022'], name='rx')
    sim, uids, reached = _run_with_cancers(treat)
    assert reached, 'nobody was treated'
    assert not set(uids.tolist()) & reached


def test_radiation_only_treats_cancer_with_has_cancer():
    """Radiation given where=hpv.has_cancer reaches the women with cancer, and nobody else."""
    treat = ss.treat(hpv.radiation(), where=ss.ppl.female & hpv.has_cancer, coverage=1.0, at=['2022'], name='rx')
    sim, uids, reached = _run_with_cancers(treat)
    alive = uids[sim.people.alive[uids]] # Those who have not died of other causes
    assert len(alive) and set(alive.tolist()) <= reached
    cancerous = set(np.concatenate([sim.diseases[g].cancerous.uids for g in genotypes]).tolist())
    assert reached <= cancerous | set(uids.tolist())


def test_treatment_capacity_respected():
    """With a capacity per round (in people, not agents), no more than that many are treated per round."""
    capacity = 1e6
    treat = ss.treat(hpv.tx('excision'), where=ss.ppl.female & hpv.no_cancer, capacity=capacity, interval=ss.years(1), name='rx')
    sim = _four_genotype_sim_with([treat])
    sim.run()
    delivered = np.asarray(sim.results.rx.delivered)
    rounds = (delivered > 0).sum()
    assert delivered.sum() <= capacity*rounds + sim.pars.pop_scale # Within one agent: a capacity of a fraction of an agent carries over
    assert delivered.sum() > 0


def test_ablation_cures_cin():
    """Ablation of women with CIN cures 93.6% of them (products_tx.csv)."""
    treat = ss.treat(hpv.tx('ablation'), where=ss.ppl.female, coverage=1.0, at=['2021'], name='rx')
    sim = hpv.Sim(n_agents=4000, start=2020, stop=2022, location='nigeria', genotypes=[16], interventions=[treat], verbose=0)
    sim.init()
    women = sim.people.female.uids
    sim.diseases.hpv16.infected[women] = True
    sim.diseases.hpv16.susceptible[women] = False
    sim.diseases.hpv16.cin[women] = True
    sim.diseases.hpv16.ti_cin[women] = 0
    sim.diseases.hpv16.ti_clearance[women] = np.inf
    sim.run()
    r = sim.results.rx
    cured, failed = np.asarray(r.cured).sum(), np.asarray(r.failed).sum()
    assert 0.9 < cured/(cured + failed) < 0.97
