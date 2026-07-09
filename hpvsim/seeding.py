"""Initial HPV-prevalence curves and multi-genotype seeding Connector.

The age-banded init_prev curves and ``_make_init_prev_fn`` sampler are used by
both per-genotype ``HPV`` modules (for ``init_seeding='independent'``) and by
``_ExclusiveSeeder`` (for ``init_seeding='exclusive'``, which uses the 'total'
curve as the any-HPV Bernoulli before genotype assignment).
"""

import numpy as np
import starsim as ss

from .network import SexualNetwork


# Initial HPV prevalence curves by age bracket and sex. Brackets are
# inclusive lower bounds; the last bracket extends to age 150.
#
# 'total': total-HPV prevalence curve, used by _ExclusiveSeeder
# for the per-agent any-HPV Bernoulli before genotype assignment.
#
# Per-genotype curves ('hpv16', 'hpv18', 'hi5', 'ohr') are used in
# 'independent' init_seeding mode, where each genotype seeds from its own
# curve. hpv16's per-genotype curve aliases 'total' since hpv16 dominates
# the total in the source v2 data. hpv18 is 0.6x of hpv16; hi5/ohr are 0.4x.
_INIT_HPV_PREV_AGE_BRACKETS = np.array([12, 17, 24, 34, 44, 64, 80, 150])

_INIT_PREV = {
    'total': {
        'm': np.array([0.0, 0.25, 0.60, 0.25, 0.05, 0.01, 0.0005, 0.0]),
        'f': np.array([0.0, 0.35, 0.70, 0.25, 0.05, 0.01, 0.0005, 0.0]),
    },
    'hpv18': {
        'm': np.array([0.0, 0.15, 0.36, 0.15, 0.03, 0.006, 0.0003, 0.0]),
        'f': np.array([0.0, 0.21, 0.42, 0.15, 0.03, 0.006, 0.0003, 0.0]),
    },
    'hi5': {
        'm': np.array([0.0, 0.10, 0.24, 0.10, 0.02, 0.004, 0.0002, 0.0]),
        'f': np.array([0.0, 0.14, 0.28, 0.10, 0.02, 0.004, 0.0002, 0.0]),
    },
    'ohr': {
        'm': np.array([0.0, 0.10, 0.24, 0.10, 0.02, 0.004, 0.0002, 0.0]),
        'f': np.array([0.0, 0.14, 0.28, 0.10, 0.02, 0.004, 0.0002, 0.0]),
    },
}
_INIT_PREV['hpv16'] = _INIT_PREV['total']


def _make_init_prev_fn(key):
    """Return the per-uid init-prev sampler for a given _INIT_PREV key.

    Accepts a genotype key ('hpv16', 'hpv18', 'hi5', 'ohr') for per-genotype
    seeding or 'total' for the v2-equivalent total-HPV curve.
    """
    curves = _INIT_PREV[key]
    f_curve = curves['f']
    m_curve = curves['m']

    def _age_stratified(module, sim, uids):
        age = sim.people.age[uids]
        is_female = sim.people.female[uids]
        bin_idx = np.searchsorted(_INIT_HPV_PREV_AGE_BRACKETS, age, side='right')
        bin_idx = np.clip(bin_idx, 0, len(f_curve) - 1)
        out = np.zeros(len(uids))
        out[is_female] = f_curve[bin_idx[is_female]]
        out[~is_female] = m_curve[bin_idx[~is_female]]
        return out
    return _age_stratified


class _ExclusiveSeeder(ss.Connector):
    """Coordinated initial seeding for multi-genotype HPV.

    Registered as an ``ss.Connector`` so the seeding Dists go through the
    standard ``define_pars`` -> ``init_pre`` -> ``init_dists`` lifecycle.

    On the first invocation of any per-genotype callback, computes the
    global assignment (per-agent total-prevalence Bernoulli + per-infected
    -agent genotype choice) and caches it. Each genotype's callback
    returns 1.0 for uids assigned to it and 0.0 otherwise; an
    ``ss.bernoulli`` with those probabilities deterministically yields
    exactly the assigned uids in ``init_prev.filter()``.
    """

    def __init__(self, genotype_keys, init_hpv_dist=None, **kwargs):
        super().__init__(**kwargs)
        self.keys = tuple(genotype_keys)
        weights = None
        if init_hpv_dist is not None:
            weights = np.array(
                [init_hpv_dist[k] for k in self.keys], dtype=float
            )
            weights = weights / weights.sum()
        self.define_pars(
            seed_bern=ss.bernoulli(p=0.0),
            seed_choice=ss.choice(a=len(self.keys), p=weights),
        )
        # Per-genotype assigned uids, aligned with self.keys. Populated lazily
        # on the first init_prev callback.
        self._assigned_uids = None

    def step(self):
        return  # No per-step logic; compute fires lazily on first init_prev callback.

    def for_genotype(self, key):
        """Return an init_prev callback for ``ss.bernoulli(p=callback)``.

        Returns 1.0 for uids assigned to this genotype, 0.0 otherwise.
        The first invocation triggers the shared lazy compute.

        The callback resolves the LIVE seeder from ``sim.connectors`` rather
        than capturing ``self``. A deep-copy (``sc.dcp`` / ``ss.Calibration`` /
        ``MultiSim`` / parallel) copies the connector-registered seeder and the
        closure-captured seeder as SEPARATE objects; only the connector copy
        goes through ``init_dists``, so a callback bound to the captured copy
        would ``force``-reinit ``seed_bern``/``seed_choice`` onto a different
        (and, on starsim >=3.5, platform-dependent) seed -> non-deterministic,
        non-portable seeding. Resolving the connector instance keeps a single
        properly-initialised source of truth, making seeding copy-stable and
        platform-stable.
        """
        gen_idx = self.keys.index(key)

        def callback(module, sim, uids):
            seeder = _live_seeder(sim)
            if seeder._assigned_uids is None:
                seeder._compute(sim)
            return np.isin(uids, seeder._assigned_uids[gen_idx]).astype(float)

        return callback

    def _compute(self, sim):
        """Compute per-agent genotype assignment (called once on first callback).

        Steps:
          1. Per-agent total HPV probability using the 'total' curve.
          2. Zero out agents who are not yet past sexual debut.
          3. Bernoulli draw: which agents get any HPV at all.
          4. Per-infected-agent genotype assignment (uniform or weighted).
        """
        people = sim.people
        auids = people.auids

        # Step 1: total-HPV probability per alive agent.
        p_per_uid = _make_init_prev_fn('total')(None, sim, auids)

        # Step 2: gate on sexual debut where a SexualNetwork is present.
        net = None
        if sim.networks is not None:
            net = next(
                (n for n in sim.networks.values() if isinstance(n, SexualNetwork)),
                None,
            )
        if net is not None:
            p_per_uid[~net.active(people)[auids]] = 0.0

        # Step 3: Bernoulli draw — who gets any HPV at all.
        # Note: when ss.Calibration deep-copies the sim, the _ExclusiveSeeder
        # stored as sim._seeder may be a different object from the one in
        # sim.connectors (which is the one init_dists initialises). Ensure
        # seed_bern is initialised before use so both paths work.
        self.pars.seed_bern.set(p=p_per_uid)
        if not self.pars.seed_bern.initialized:
            self.pars.seed_bern.init(sim=sim, force=True)
        infected_uids = self.pars.seed_bern.filter(auids)

        # Step 4: per-infected-agent genotype assignment. ss.choice draws
        # independent per-uid uniforms via its auto-generated unique trace.
        if len(infected_uids):
            if not self.pars.seed_choice.initialized:
                self.pars.seed_choice.init(sim=sim, force=True)
            gen_choices = self.pars.seed_choice.rvs(infected_uids)
            self._assigned_uids = tuple(
                infected_uids[gen_choices == i] for i in range(len(self.keys))
            )
        else:
            # No infections — empty assignment for each genotype.
            self._assigned_uids = tuple(infected_uids for _ in self.keys)


def _live_seeder(sim):
    """Return the _ExclusiveSeeder registered in ``sim.connectors``.

    init_prev callbacks resolve the seeder through this rather than capturing
    ``self`` so they always use the connector instance that went through
    ``init_dists`` — see ``_ExclusiveSeeder.for_genotype``. Exactly one is
    auto-wired by ``hpv.Sim`` when ``init_seeding='exclusive'``.
    """
    for c in sim.connectors.values():
        if isinstance(c, _ExclusiveSeeder):
            return c
    raise RuntimeError('_ExclusiveSeeder not found in sim.connectors; '
                       'exclusive-seeding init_prev callback cannot resolve it.')