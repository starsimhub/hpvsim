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
# the total prevalence. hpv18 is 0.6x of hpv16; hi5/ohr are 0.4x.
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
    seeding or 'total' for the total-HPV curve.
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
    """Mutually-exclusive initial seeding for multi-genotype HPV (v2-faithful).

    In ``init_seeding='exclusive'`` each seeded agent starts with AT MOST ONE
    genotype: a per-agent any-HPV draw (the 'total' prevalence curve) picks who
    is infected, then a single genotype is assigned per infected agent. This
    differs from ``'independent'`` seeding, where every genotype draws from its
    own curve and co-infection at t=0 is possible.

    Why a Connector (rather than seeding all genotypes in one place)? Two
    starsim constraints:
      * Each genotype is a separate ``HPV`` disease module, and starsim seeds
        every disease independently in ``Disease.init_post`` via
        ``init_prev.filter()`` (which also records ``_n_initial_cases`` so the
        seeded cases are excluded from first-step incidence results). Reusing
        that path keeps exclusive seeding consistent with the independent path
        and gets the bookkeeping for free.
      * We therefore need the assignment to be *shared* across those modules.
        A Connector is the natural home for the two seeding Dists (so they go
        through the standard ``define_pars`` -> ``init_pre`` -> ``init_dists``
        lifecycle) and for the shared assignment cache. Note connectors run
        ``init_post`` BEFORE diseases, so the seeder cannot seed directly in its
        own ``init_post`` (rel_sev is not sampled yet) — instead it computes
        lazily on the first per-genotype callback, which fires from within the
        diseases' ``init_post``.

    Mechanism: on the first per-genotype callback, ``_compute`` fixes the global
    assignment (any-HPV Bernoulli + per-infected genotype choice) and caches it.
    Each genotype's callback returns 1.0 for its assigned uids and 0.0 otherwise,
    so ``ss.bernoulli(p=callback).filter()`` yields exactly those uids.
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

        Returns 1.0 for uids assigned to this genotype, 0.0 otherwise; the
        first invocation triggers the shared lazy compute.

        The callback resolves the live seeder from ``sim.connectors`` (via
        ``_live_seeder``) instead of capturing ``self`` so it stays correct
        under deep-copy (``ss.Calibration`` / ``MultiSim`` / parallel). A copy
        clones the connector-registered seeder and any closure-captured seeder
        as separate objects, and only the connector copy goes through
        ``init_dists`` — so a callback bound to the captured copy would use
        uninitialised/force-reinitialised Dists and seed a different set of
        agents. Resolving the connector keeps one initialised source of truth.
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
        # _live_seeder routes callbacks to the connector-registered seeder, but
        # guard anyway: init seed_bern if a copy left it uninitialised.
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