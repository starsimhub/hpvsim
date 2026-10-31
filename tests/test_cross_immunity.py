"""Unit tests for the cross-immunity Connector and matrix-builder."""
import numpy as np
import pytest
import starsim as ss

import hpvsim as hpv
from hpvsim.parameters import GENOTYPE_KEYS


def _default_matrices(**overrides):
    """Test helper: hpv.CrossImmunity().make_cross_immunity() with optional
    par overrides applied via update_pars before matrix construction."""
    conn = hpv.CrossImmunity()
    if overrides:
        conn.pars.update({k: v for k, v in overrides.items() if k != 'keys'})
    return conn.make_cross_immunity(keys=overrides.get('keys'))


def test_genotype_keys_are_canonical_four():
    """GENOTYPE_KEYS pins the 4-genotype default ordering."""
    assert GENOTYPE_KEYS == ('hpv16', 'hpv18', 'hi5', 'ohr')


def test_genotype_registries_agree_on_canonical_keys():
    """All six genotype-key registries scattered across the package must agree
    on the canonical key set. Adding a fifth genotype is a one-entry edit per
    registry; this test prevents drift.

    Registries:
      - ``hpvsim.parameters.GENOTYPE_KEYS``       (tuple, canonical ordering)
      - ``hpvsim.parameters.genotype_aliases``    (dict: canonical -> aliases)
      - ``hpvsim.parameters._GENOTYPE_DEFAULTS``  (dict: canonical -> defaults)
      - ``hpvsim.cross_genotype._FULL_OWN_IMM_KEYS`` (frozenset; subset of canonical)
      - ``hpvsim.hpv._KNOWN_GENOTYPES``           (tuple)
      - ``hpvsim.seeding._INIT_PREV``             (dict; includes 'total' meta-key)
    """
    from hpvsim.parameters import (
        GENOTYPE_KEYS, genotype_aliases, _GENOTYPE_DEFAULTS,
    )
    from hpvsim.cross_genotype import _FULL_OWN_IMM_KEYS
    from hpvsim.hpv import _KNOWN_GENOTYPES
    from hpvsim.seeding import _INIT_PREV

    canonical = set(GENOTYPE_KEYS)
    assert set(genotype_aliases.keys()) == canonical, \
        'genotype_aliases keys do not match GENOTYPE_KEYS'
    assert set(_GENOTYPE_DEFAULTS.keys()) == canonical, \
        '_GENOTYPE_DEFAULTS keys do not match GENOTYPE_KEYS'
    assert set(_KNOWN_GENOTYPES) == canonical, \
        '_KNOWN_GENOTYPES does not match GENOTYPE_KEYS'
    assert _FULL_OWN_IMM_KEYS <= canonical, \
        '_FULL_OWN_IMM_KEYS contains keys not in GENOTYPE_KEYS'
    # _INIT_PREV carries an extra 'total' key for the any-HPV prevalence curve.
    assert set(_INIT_PREV.keys()) - {'total'} == canonical, \
        '_INIT_PREV genotype keys do not match GENOTYPE_KEYS'


def test_get_cross_immunity_default_shape_and_diagonal():
    """Default cross-immunity matrices are (4, 4) float32; diagonal = 1.0
    for hpv16/hpv18, 0.9 (v2 own_imm_hr default) for hi5/ohr."""
    m_sus, m_sev = _default_matrices()
    keys = ('hpv16', 'hpv18', 'hi5', 'ohr')
    idx = {k: i for i, k in enumerate(keys)}
    expected_diag = [1.0, 1.0, 0.9, 0.9]
    for m in (m_sus, m_sev):
        assert m.shape == (4, 4)
        assert m.dtype == np.float32
        assert np.allclose(np.diag(m), expected_diag)


def test_get_cross_immunity_default_values():
    """Defaults: cross_imm_sus_med=0.3, cross_imm_sus_high=0.5,
    cross_imm_sev_med=0.5, cross_imm_sev_high=0.7, own_imm_hr=0.9."""
    m_sus, m_sev = _default_matrices()
    keys = ('hpv16', 'hpv18', 'hi5', 'ohr')
    idx = {k: i for i, k in enumerate(keys)}
    # Off-diagonal hpv16<->hpv18 = high (0.5 sus, 0.7 sev); both directions.
    assert m_sus[idx['hpv16'], idx['hpv18']] == pytest.approx(0.5)
    assert m_sus[idx['hpv18'], idx['hpv16']] == pytest.approx(0.5)
    assert m_sev[idx['hpv16'], idx['hpv18']] == pytest.approx(0.7)
    # hpv16 -> hi5 = med
    assert m_sus[idx['hpv16'], idx['hi5']] == pytest.approx(0.3)
    assert m_sev[idx['hpv16'], idx['hi5']] == pytest.approx(0.5)
    # Diagonal: hpv16/hpv18 = 1.0 (canonical own-immunity); hi5/ohr = 0.9.
    assert m_sus[idx['hpv16'], idx['hpv16']] == pytest.approx(1.0)
    assert m_sus[idx['hi5'], idx['hi5']] == pytest.approx(0.9)
    assert m_sev[idx['ohr'], idx['ohr']] == pytest.approx(0.9)


def test_get_cross_immunity_own_imm_hr_override():
    """own_imm_hr kwarg overrides the 0.9 default for non-canonical genotypes."""
    m_sus, _ = _default_matrices(own_imm_hr=0.7)
    keys = ('hpv16', 'hpv18', 'hi5', 'ohr')
    idx = {k: i for i, k in enumerate(keys)}
    # hpv16/hpv18 still 1.0; hi5/ohr now use the override.
    assert m_sus[idx['hpv16'], idx['hpv16']] == pytest.approx(1.0)
    assert m_sus[idx['hi5'], idx['hi5']] == pytest.approx(0.7)
    assert m_sus[idx['ohr'], idx['ohr']] == pytest.approx(0.7)


def test_get_cross_immunity_custom_keys():
    """Caller-supplied genotype ordering controls matrix layout."""
    m_sus, _ = _default_matrices(keys=('hi5', 'hpv16'))
    assert m_sus.shape == (2, 2)
    # m_sus[1, 0] is "from hi5 source to hpv16 target" — medium scalar.
    assert m_sus[1, 0] == pytest.approx(0.3)


def test_genotype_pars_imm_init_is_distribution():
    """GenotypePars.imm_init becomes a beta-mean distribution."""
    gp = hpv.get_genotype_pars('hpv16')
    assert hasattr(gp.imm_init, 'rvs'), \
        f'imm_init should be a Dist, got {type(gp.imm_init)}'


def test_cross_immunity_connector_collects_hpv_modules():
    """CrossImmunity.init_pre populates hpv_modules from sim.diseases in registration order."""
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
    )
    sim.init()
    conn = sim.connectors.crossimmunity   # Starsim auto-snake-cases class name
    assert len(conn.hpv_modules) == 1
    assert conn.hpv_modules[0].genotype == 'hpv16'
    assert conn.genotype_index == {'hpv16': 0}


def test_cross_immunity_connector_default_matrices():
    """If matrices not supplied, init_pre populates from get_cross_immunity for the discovered genotype set."""
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
    )
    sim.init()
    conn = sim.connectors.crossimmunity   # Starsim copies; check the live instance
    assert conn.cross_imm_sus.shape == (1, 1)
    assert conn.cross_imm_sus.dtype == np.float32
    assert conn.cross_imm_sus[0, 0] == pytest.approx(1.0)


def test_cross_immunity_connector_rejects_diagonal_outside_unit_interval():
    """Diagonal entries must be in [0, 1] (own-immunity is a probability);
    init_pre raises if outside that range. Per-key diagonal values < 1.0
    (e.g. own_imm_hr=0.9 for hi5/ohr) are now allowed.

    Drops down to ss.Sim since hpv.Sim appends a default CrossImmunity that
    would mask a user-supplied one.
    """
    from hpvsim.cross_genotype import CrossImmunity
    bad = np.array([[1.5]], dtype=np.float32)
    conn = CrossImmunity(cross_imm_sus=bad, cross_imm_sev=bad)
    sim = ss.Sim(
        n_agents=10, start=ss.years(1990), stop=ss.years(1991), dt=ss.years(1.0),
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[conn],
    )
    with pytest.raises(ValueError, match='diagonal'):
        sim.init()


def test_cross_immunity_connector_accepts_per_key_diagonal():
    """Per-key diagonal: hpv16/hpv18=1.0, hi5/ohr=0.9. init_pre must accept this."""
    sim = hpv.Sim(
        n_agents=100, start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    )
    sim.init()
    conn = sim.connectors.crossimmunity
    assert np.allclose(np.diag(conn.cross_imm_sus), [1.0, 1.0, 0.9, 0.9])
    assert np.allclose(np.diag(conn.cross_imm_sev), [1.0, 1.0, 0.9, 0.9])


def test_cross_immunity_connector_rejects_shape_mismatch():
    """Matrix dim must match number of HPV modules.

    Drops down to ss.Sim since hpv.Sim appends a default CrossImmunity that
    would mask a user-supplied one.
    """
    from hpvsim.cross_genotype import CrossImmunity
    bad = np.eye(2, dtype=np.float32)
    conn = CrossImmunity(cross_imm_sus=bad, cross_imm_sev=bad)
    sim = ss.Sim(
        n_agents=10, start=ss.years(1990), stop=ss.years(1991), dt=ss.years(1.0),
        diseases=[hpv.HPV(genotype='hpv16')],
        connectors=[conn],
    )
    with pytest.raises(ValueError, match='shape'):
        sim.init()


def test_cross_immunity_step_identity_for_single_genotype():
    """1x1 identity matrix: rel_sus = 1 - nab_imm, sev_imm = cell_imm."""
    sim = hpv.Sim(
        n_agents=10, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
    )
    sim.init()
    mod = sim.diseases.hpv16
    conn = sim.connectors.crossimmunity
    mod.nab_imm.values[:5] = 0.4
    mod.cell_imm.values[:5] = 0.3
    conn.step()
    # Identity multiply: rel_sus = 1 - 0.4 = 0.6 for first five, 1.0 elsewhere.
    rel = np.asarray(mod.rel_sus.values)
    sev = np.asarray(mod.sev_imm.values)
    assert np.allclose(rel[:5], 0.6)
    assert np.allclose(rel[5:], 1.0)
    assert np.allclose(sev[:5], 0.3)
    assert np.allclose(sev[5:], 0.0)


def test_cross_immunity_step_two_genotype_hand_computed():
    """2-genotype case: rel_sus and sev_imm match hand-computed dot product.

    The auto-default 2-genotype matrices match the values used here:
    cross_imm_sus = [[1.0, 0.5], [0.5, 1.0]], cross_imm_sev = [[1.0, 0.7], [0.7, 1.0]].
    """
    sim = hpv.Sim(
        n_agents=4, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
    )
    sim.init()
    h16 = sim.diseases.hpv16
    h18 = sim.diseases.hpv18
    conn = sim.connectors.crossimmunity
    # Agent 0: had hpv16 (nab=0.4, cell=0.3); no hpv18 history.
    h16.nab_imm.values[0] = 0.4
    h16.cell_imm.values[0] = 0.3
    # Agent 1: had hpv18 (nab=0.6, cell=0.5); no hpv16 history.
    h18.nab_imm.values[1] = 0.6
    h18.cell_imm.values[1] = 0.5
    conn.step()
    # rel_sus[t] = 1 - sum_s cross_imm_sus[t, s] * nab_imm[s]
    assert h16.rel_sus.values[0] == pytest.approx(0.6, abs=1e-6)
    assert h18.rel_sus.values[0] == pytest.approx(0.8, abs=1e-6)
    assert h16.rel_sus.values[1] == pytest.approx(0.7, abs=1e-6)
    assert h18.rel_sus.values[1] == pytest.approx(0.4, abs=1e-6)
    # sev_imm[t] = sum_s cross_imm_sev[t, s] * cell_imm[s]
    assert h16.sev_imm.values[0] == pytest.approx(0.3, abs=1e-6)
    assert h18.sev_imm.values[1] == pytest.approx(0.5, abs=1e-6)


def test_cross_immunity_step_four_genotype_hand_computed():
    """4-genotype: matrix multiply produces correct rel_sus / sev_imm.

    Detects index-orientation bugs that 2-genotype cases can't catch (the 2x2
    diagonal-symmetric matrix is invariant under target/source swap; 4x4 with
    asymmetric source-vs-target structure is not).

    Uses the default 4-genotype matrices:
        cross_imm_sus = [[1.0, 0.5, 0.3, 0.3],
                         [0.5, 1.0, 0.3, 0.3],
                         [0.3, 0.3, 0.9, 0.3],
                         [0.3, 0.3, 0.3, 0.9]]
        cross_imm_sev = [[1.0, 0.7, 0.5, 0.5],
                         [0.7, 1.0, 0.5, 0.5],
                         [0.5, 0.5, 0.9, 0.5],
                         [0.5, 0.5, 0.5, 0.9]]
    where row = target, col = source. Convention: immunity[target_idx,
    source_idx] = pars[label_target][label_source].
    """
    from hpvsim.cross_genotype import CrossImmunity
    sim = hpv.Sim(
        n_agents=8, start=1990, stop=1991, dt=1.0, rand_seed=0,
        genotypes=['hpv16', 'hpv18', 'hi5', 'ohr'],
    )
    sim.init()
    h16 = sim.diseases.hpv16
    h18 = sim.diseases.hpv18
    hi5 = sim.diseases.hi5
    ohr = sim.diseases.ohr
    conn = sim.connectors.crossimmunity

    # Agent 0: cleared hpv16 only (nab=0.4, cell=0.3).
    h16.nab_imm.values[0] = 0.4
    h16.cell_imm.values[0] = 0.3
    # Agent 1: cleared hi5 only (nab=0.5, cell=0.4).
    hi5.nab_imm.values[1] = 0.5
    hi5.cell_imm.values[1] = 0.4
    # Agent 2: cleared hpv16 (nab=0.4) AND hi5 (nab=0.5); cell=0.3 each.
    h16.nab_imm.values[2] = 0.4
    h16.cell_imm.values[2] = 0.3
    hi5.nab_imm.values[2] = 0.5
    hi5.cell_imm.values[2] = 0.3
    # Agent 3: cleared ohr only (nab=0.6, cell=0.5).
    ohr.nab_imm.values[3] = 0.6
    ohr.cell_imm.values[3] = 0.5

    conn.step()

    # --- Agent 0 (cleared hpv16 only) ---
    # rel_sus[t] = 1 - cross_imm_sus[t, hpv16] * 0.4
    assert h16.rel_sus.values[0] == pytest.approx(0.60, abs=1e-6)
    assert h18.rel_sus.values[0] == pytest.approx(0.80, abs=1e-6)
    assert hi5.rel_sus.values[0] == pytest.approx(0.88, abs=1e-6)
    assert ohr.rel_sus.values[0] == pytest.approx(0.88, abs=1e-6)
    # sev_imm[t] = cross_imm_sev[t, hpv16] * 0.3
    assert h16.sev_imm.values[0] == pytest.approx(0.30, abs=1e-6)
    assert h18.sev_imm.values[0] == pytest.approx(0.21, abs=1e-6)
    assert hi5.sev_imm.values[0] == pytest.approx(0.15, abs=1e-6)
    assert ohr.sev_imm.values[0] == pytest.approx(0.15, abs=1e-6)

    # --- Agent 1 (cleared hi5 only) ---
    # rel_sus[t] = 1 - cross_imm_sus[t, hi5] * 0.5
    assert h16.rel_sus.values[1] == pytest.approx(0.85, abs=1e-6)
    assert h18.rel_sus.values[1] == pytest.approx(0.85, abs=1e-6)
    assert hi5.rel_sus.values[1] == pytest.approx(0.55, abs=1e-6)
    assert ohr.rel_sus.values[1] == pytest.approx(0.85, abs=1e-6)
    # sev_imm[t] = cross_imm_sev[t, hi5] * 0.4
    assert h16.sev_imm.values[1] == pytest.approx(0.20, abs=1e-6)
    assert hi5.sev_imm.values[1] == pytest.approx(0.36, abs=1e-6)
    assert ohr.sev_imm.values[1] == pytest.approx(0.20, abs=1e-6)

    # --- Agent 2 (cleared hpv16 AND hi5) — additivity test ---
    # rel_sus[t] = 1 - (cross_imm_sus[t, hpv16]*0.4 + cross_imm_sus[t, hi5]*0.5)
    assert h16.rel_sus.values[2] == pytest.approx(0.45, abs=1e-6)
    assert h18.rel_sus.values[2] == pytest.approx(0.65, abs=1e-6)
    assert hi5.rel_sus.values[2] == pytest.approx(0.43, abs=1e-6)
    assert ohr.rel_sus.values[2] == pytest.approx(0.73, abs=1e-6)

    # --- Agent 3 (cleared ohr only) — non-canonical source asymmetry ---
    # rel_sus[t] = 1 - cross_imm_sus[t, ohr] * 0.6
    assert h16.rel_sus.values[3] == pytest.approx(0.82, abs=1e-6)
    assert hi5.rel_sus.values[3] == pytest.approx(0.82, abs=1e-6)
    assert ohr.rel_sus.values[3] == pytest.approx(0.46, abs=1e-6)


def test_cross_immunity_step_clips_to_unit_interval():
    """sus_imm and sev_imm are clipped to [0, 1] (np.minimum cap)."""
    sim = hpv.Sim(
        n_agents=4, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16'), hpv.HPV(genotype='hpv18')],
    )
    sim.init()
    sim.diseases.hpv16.nab_imm.values[0] = 0.9
    sim.diseases.hpv18.nab_imm.values[0] = 0.9
    sim.connectors.crossimmunity.step()
    # sus_imm to hpv16 = 1*0.9 + 0.5*0.9 = 1.35, clipped to 1.0
    assert sim.diseases.hpv16.rel_sus.values[0] == pytest.approx(0.0, abs=1e-6)


def test_cross_immunity_step_writes_survive_dead_agents():
    """Connector writes must use Arr __setitem__ via auids, not .values[:].

    With dead agents present (raw.size != auids.size), FloatArr.values returns
    a copy; writing to .values[:] would be silently discarded. Verify writes
    actually land in raw.
    """
    sim = hpv.Sim(
        n_agents=20, start=1990, stop=1991, dt=1.0, rand_seed=0,
        diseases=[hpv.HPV(genotype='hpv16')],
    )
    sim.init()
    mod = sim.diseases.hpv16
    conn = sim.connectors.crossimmunity
    # Kill agents 0 and 1 to force raw.size != auids.size.
    uids_to_kill = ss.uids([0, 1])
    sim.people.ti_dead[uids_to_kill] = -1   # mark as died before current step
    sim.people.alive[uids_to_kill] = False
    sim.people.remove_dead()                 # refresh active-uid arrays
    assert mod.rel_sus.raw.size != mod.rel_sus.auids.size
    live_auids = mod.rel_sus.auids
    target_auid = int(live_auids[0])
    mod.nab_imm[ss.uids([target_auid])] = 0.4
    mod.cell_imm[ss.uids([target_auid])] = 0.3
    conn.step()
    # The write must land in raw, not an ephemeral .values copy.
    assert float(mod.rel_sus.raw[target_auid]) == pytest.approx(0.6, abs=1e-6)
    assert float(mod.sev_imm.raw[target_auid]) == pytest.approx(0.3, abs=1e-6)


def test_cross_immunity_top_level_import():
    """hpv.CrossImmunity + its make_cross_immunity factory are reachable from the root."""
    assert hasattr(hpv, 'CrossImmunity')
    assert hasattr(hpv.CrossImmunity(), 'make_cross_immunity')
    assert hasattr(hpv, 'GENOTYPE_KEYS')
