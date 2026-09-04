"""Unit tests for hpv.dx — per-genotype multinomial classifier."""
import numpy as np
import pytest
import starsim as ss
import hpvsim as hpv
from hpvsim.products import dx as hpv_dx


def _four_genotype_sim():
    sim = hpv.Sim(
        n_agents=200, start=2020, stop=2021, location='nigeria',
        diseases=[hpv.HPV(g) for g in ('hpv16', 'hpv18', 'hi5', 'ohr')],
    )
    return sim


def _attach_dx_and_init(sim, dx_instance):
    """Attach the dx product to a stub treat_num, init the sim, and return
    the initialized copy of the product.

    Starsim deep-copies pars inputs on Sim construction, so the object stored
    on the intervention after init may differ from the one passed in. Callers
    must use the returned product for all administer/result_dist calls.
    """
    # prob=0 attaches the dx to a sim that will init without ever firing it.
    stub = ss.treat_num(product=dx_instance, prob=0.0)
    sim.pars['interventions'] = [stub]
    sim.init()
    return sim.interventions[0].product


def _force_state(sim, uids, state, genotype):
    """Manually set a per-genotype BoolState for testing."""
    mod = sim.diseases[genotype]
    arr = getattr(mod, state)
    arr[uids] = True


def test_dx_via_hierarchy_default():
    """The 'via' product default hierarchy is positive > inadequate > negative."""
    d = hpv_dx(name='via')
    assert d.hierarchy == ['positive', 'inadequate', 'negative']


def test_dx_unknown_name_raises():
    with pytest.raises(ValueError, match='Unknown dx product name'):
        hpv_dx(name='nope')


def test_dx_name_and_df_gives_custom_name_to_df_built_product():
    """Passing both name= and df= keeps df as the row table and uses name
    as the module name — so two df-built dx products can share a sim."""
    import pandas as pd
    df = pd.DataFrame([
        {'name': 'x', 'state': 'susceptible', 'genotype': 'all', 'result': 'positive', 'probability': 0.0},
        {'name': 'x', 'state': 'susceptible', 'genotype': 'all', 'result': 'negative', 'probability': 1.0},
    ])
    d = hpv_dx(name='my_dx', df=df)
    assert d.name == 'my_dx'
    assert list(d.df.result.unique()) == ['positive', 'negative']


def test_dx_neither_name_nor_df_raises():
    with pytest.raises(ValueError, match='at least one'):
        hpv_dx()


def test_two_df_built_dx_products_coexist_in_a_sim():
    """Regression: two df-built dx products with distinct names must both
    init on the same sim without colliding on the class-default 'dx' name."""
    import pandas as pd
    def _mkdf(prob_pos):
        return pd.DataFrame([
            {'name': 'x', 'state': 'susceptible', 'genotype': 'all', 'result': 'positive', 'probability': prob_pos},
            {'name': 'x', 'state': 'susceptible', 'genotype': 'all', 'result': 'negative', 'probability': 1 - prob_pos},
        ])
    d1 = hpv_dx(name='dx_screen', df=_mkdf(0.1))
    d2 = hpv_dx(name='dx_triage', df=_mkdf(0.9))
    sim = _four_genotype_sim()
    sim.pars['interventions'] = [
        ss.treat_num(name='stub_a', product=d1, prob=0.0),
        ss.treat_num(name='stub_b', product=d2, prob=0.0),
    ]
    sim.init()  # would raise 'Module dx already added' before the fix


def test_dx_all_genotype_mode_classifies_susceptibles():
    """A 'via' test (genotype='all' rows) on fully-susceptible agents
    returns the result drawn from the 'susceptible' probability row."""
    sim = _four_genotype_sim()
    d_init = _attach_dx_and_init(sim, hpv_dx(name='via'))
    alive = sim.people.alive.uids[:5]
    out = d_init.administer(alive)
    assert set(out.keys()) == {'positive', 'inadequate', 'negative'}
    total = sum(len(v) for v in out.values())
    assert total == 5


def test_dx_latent_state_silently_empty():
    """The 'hpv' product CSV has latent rows; with no latent agents (no-op
    state), those rows contribute zero classified people (no error)."""
    sim = _four_genotype_sim()
    d_init = _attach_dx_and_init(sim, hpv_dx(name='hpv'))
    alive = sim.people.alive.uids[:10]
    out = d_init.administer(alive)
    # hierarchy is ['positive', 'inadequate', 'negative']
    assert set(out.keys()) == {'positive', 'inadequate', 'negative'}


def test_dx_per_genotype_classifies_precin_hpv16():
    """A precin-on-hpv16 agent under the 'hpv_type' (per-genotype) product
    should classify into one of the hierarchy values."""
    sim = _four_genotype_sim()
    d_init = _attach_dx_and_init(sim, hpv_dx(name='hpv_type'))
    uid = sim.people.alive.uids[0:1]
    _force_state(sim, uid, 'precin', 'hpv16')
    out = d_init.administer(uid)
    classified = next(k for k, v in out.items() if len(v) == 1)
    assert classified in d_init.hierarchy


def test_dx_empty_uids_returns_empty_dict():
    sim = _four_genotype_sim()
    d_init = _attach_dx_and_init(sim, hpv_dx(name='via'))
    out = d_init.administer(ss.uids())
    assert all(len(v) == 0 for v in out.values())


def test_dx_from_df_initializes_in_sim():
    """A df-built product has no product name, so it must keep the class default.

    A None module name breaks people.add_module() at sim.init().
    """
    import pandas as pd
    df = pd.DataFrame(dict(
        state=['susceptible', 'precin'],
        genotype=['all', 'all'],
        result=['negative', 'positive'],
        probability=[1.0, 1.0],
    ))
    d = hpv_dx(df=df, hierarchy=['positive', 'negative'])
    assert d.name == 'dx'
    _attach_dx_and_init(_four_genotype_sim(), d)
