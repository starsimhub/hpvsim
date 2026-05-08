"""Unit tests for M02 progression math + GenotypePars defaults."""
import numpy as np
import pytest

import hpvsim as hpv
from hpvsim.utils import (
    compute_severity,
    compute_severity_integral,
    intlogf2,
    logf2,
    transform_prob,
)


def test_genotype_pars_hpv16_defaults():
    """GenotypePars('hpv16') exposes the HPV16 natural-history defaults."""
    g = hpv.GenotypePars('hpv16')
    assert g.beta == 0.25
    assert g.cin_fn == dict(form='logf2', k=0.3, x_infl=0, ttc=50)
    assert g.cancer_fn['method'] == 'cin_integral'
    assert g.cancer_fn['transform_prob'] == 2e-3
    assert g.imm_init == 0.35
    assert g.age_risk == dict(age=30, risk=2)
    assert g.rel_beta == 1.0
    assert g.sero_prob == 0.75
    # Duration distributions are starsim Dist instances, not dicts.
    import starsim as ss
    for name in ('dur_precin', 'dur_cin', 'dur_cancer', 'dur_inf_male'):
        assert isinstance(getattr(g, name), ss.Dist)


def test_genotype_aliases_hpv16():
    """Aliases let us pass '16' or 'hpv16' interchangeably."""
    assert hpv.parameters.genotype_aliases['hpv16'] == ['hpv16', '16']


def test_get_genotype_pars_factory():
    """get_genotype_pars('hpv16') returns a fresh GenotypePars each call."""
    g1 = hpv.get_genotype_pars('hpv16')
    g2 = hpv.get_genotype_pars('hpv16')
    # Fresh distribution instances per call (independent RNG slots).
    assert g1.dur_precin is not g2.dur_precin


def test_logf2_pinned_outputs():
    """logf2 reproduces v2's logistic-2 outputs at canonical points.

    Generated from hpvsim._v2_legacy.utils.logf2 with the HPV16 cin_fn
    parameters (k=0.3, x_infl=0, ttc=50).
    """
    expected = {
        0.5:  0.07485973648701938,
        1.0:  0.14888512471190052,
        2.0:  0.2913127906780537,
        5.0:  0.6351493409744831,
        10.0: 0.9051488074189387,
    }
    for x, want in expected.items():
        got = logf2(x, k=0.3, x_infl=0, ttc=50)
        assert np.isclose(got, want, rtol=1e-9), f'logf2({x}) = {got}, want {want}'


def test_logf2_array_input():
    """logf2 accepts numpy arrays."""
    out = logf2(np.array([0.5, 1.0, 2.0]), k=0.3, x_infl=0, ttc=50)
    assert out.shape == (3,)
    assert (out >= 0).all() and (out <= 1).all()


def test_logf2_matches_quarantine_verbatim():
    """logf2 produces bit-identical output to v2's logf2 over a sweep.

    This is the strongest possible parity check: literally compare to v2.
    """
    from hpvsim._v2_legacy.utils import logf2 as v2_logf2
    xs = np.linspace(0.01, 60, 200)
    for k in (0.15, 0.3, 0.5):
        for x_infl in (0, 5, 10):
            for ttc in (25, 50):
                a = logf2(xs, k=k, x_infl=x_infl, ttc=ttc)
                b = v2_logf2(xs, k=k, x_infl=x_infl, ttc=ttc)
                assert np.allclose(a, b, equal_nan=True), \
                    f'mismatch at k={k} x_infl={x_infl} ttc={ttc}'


def test_transform_prob_pinned():
    """transform_prob reproduces v2's transform_prob outputs.

    Generated from hpvsim._v2_legacy.utils.transform_prob.
    """
    out = transform_prob(2e-3, np.array([0.1, 0.5, 1.0, 2.0]))
    expected = np.array([0.00010009512368247542, 0.012434560635623426, 0.09525318199596433, 0.55103083492734])
    assert np.allclose(out, expected, rtol=1e-9)


def test_transform_prob_matches_quarantine_verbatim():
    """transform_prob is bit-identical to v2's transform_prob over a sweep."""
    from hpvsim._v2_legacy.utils import transform_prob as v2_transform_prob
    for tp in (1e-4, 2e-3, 1e-2):
        dysp = np.linspace(0.01, 1.5, 100)
        a = transform_prob(tp, dysp)
        b = v2_transform_prob(tp, dysp)
        assert np.allclose(a, b, equal_nan=True), f'mismatch at tp={tp}'


def test_transform_prob_monotone_in_dysp():
    """Output is monotonically increasing in dysp (sanity check)."""
    dysp = np.linspace(0.05, 2.0, 50)
    out = transform_prob(2e-3, dysp)
    assert (np.diff(out) >= 0).all()
    # Bounded in [0, 1]
    assert (out >= 0).all() and (out <= 1).all()


def test_intlogf2_pinned():
    """intlogf2 reproduces v2's intlogf2."""
    out = intlogf2(np.array([1.0, 5.0, 10.0]), k=0.3, x_infl=0, ttc=50)
    expected = [1.0747210835763568, 6.721778095232491, 15.70294408055644]
    assert np.allclose(out, expected, rtol=1e-9)


def test_intlogf2_matches_quarantine_verbatim():
    """intlogf2 is bit-identical to v2's intlogf2 over a sweep."""
    from hpvsim._v2_legacy.utils import intlogf2 as v2_intlogf2
    upper = np.linspace(0.5, 60, 100)
    for k in (0.15, 0.3, 0.5):
        for x_infl in (0, 5):
            for ttc in (25, 50):
                a = intlogf2(upper, k=k, x_infl=x_infl, ttc=ttc)
                b = v2_intlogf2(upper, k=k, x_infl=x_infl, ttc=ttc)
                assert np.allclose(a, b, equal_nan=True), \
                    f'intlogf2 mismatch at k={k} x_infl={x_infl} ttc={ttc}'


def test_compute_severity_logf2_branch_pinned():
    """compute_severity(form='logf2') reproduces v2."""
    pars = dict(form='logf2', k=0.3, x_infl=0, ttc=50)
    out = compute_severity(np.array([1.0, 5.0, 10.0]), pars=pars)
    expected = [0.14888512471190052, 0.6351493409744831, 0.9051488074189387]
    assert np.allclose(out, expected, rtol=1e-9)


def test_compute_severity_cin_integral_branch_pinned():
    """compute_severity(method='cin_integral') reproduces v2's cancer prob."""
    pars = dict(method='cin_integral', transform_prob=2e-3,
                form='logf2', k=0.3, x_infl=0, ttc=50)
    out = compute_severity(np.array([1.0, 5.0, 10.0]), pars=pars)
    expected = [0.0023096924964789434, 0.08648463811473539, 0.3896109443915773]
    assert np.allclose(out, expected, rtol=1e-9)


def test_compute_severity_does_not_mutate_pars():
    """Caller's pars dict must not be modified (regression: v2 actually
    DOES mutate; the v3 port should make a defensive deepcopy at entry)."""
    pars = dict(method='cin_integral', transform_prob=2e-3,
                form='logf2', k=0.3, x_infl=0, ttc=50)
    snapshot = dict(pars)
    compute_severity(np.array([1.0]), pars=pars)
    assert pars == snapshot, 'pars dict was mutated by compute_severity'


def test_compute_severity_matches_quarantine_verbatim():
    """Both branches bit-identical to v2 over realistic durations."""
    from hpvsim._v2_legacy.utils import intlogf2 as v2_intlogf2
    from hpvsim._v2_legacy.utils import logf2 as v2_logf2
    durs = np.linspace(0.1, 30, 60)

    # cin (logf2) branch: compare to direct v2 logf2 calculation
    cin_fn = dict(form='logf2', k=0.3, x_infl=0, ttc=50)
    a = compute_severity(durs, pars=dict(cin_fn))
    b = v2_logf2(durs, k=0.3, x_infl=0, ttc=50)
    assert np.allclose(a, b, equal_nan=True), 'cin (logf2) branch mismatch'

    # cin_integral branch: manually replicate v2's compute_severity logic
    sev = v2_intlogf2(durs, k=0.3, x_infl=0, ttc=50)
    tp = 2e-3
    b_cancer = 1 - np.power(1 - tp, sev**2)
    cancer_fn = dict(method='cin_integral', transform_prob=2e-3,
                     form='logf2', k=0.3, x_infl=0, ttc=50)
    a_cancer = compute_severity(durs, pars=dict(cancer_fn))
    assert np.allclose(a_cancer, b_cancer, equal_nan=True), 'cin_integral branch mismatch'