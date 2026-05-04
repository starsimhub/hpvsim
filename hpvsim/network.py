"""HPVsim sexual partnership network.

Lift-and-shift of v2 hpvsim's two-layer (marital, casual) sexual network.
The pair-formation algorithm is ported from
``hpvsim/_v2_legacy/population.py:create_edgelist`` (lines 281-379) with
adaptations for Starsim's idioms (UID-indexed arrays, isinstance-filtered
sibling iteration for cross-layer concurrency). One class instantiated
twice, one per layer; inherits scaffolding (debut, participant, duration
tracking, end_pairs, net_beta) from ss.SexualNetwork.
"""

import numpy as np
import starsim as ss


_KNOWN_LAYERS = ('m', 'c')


class SexualNetwork(ss.SexualNetwork):
    """One layer of HPVsim's heterosexual partnership network.

    Args:
        layer: one of 'm' (marital/long-term), 'c' (casual).
        pars: dict of layer parameters; see hpvsim.data.load_country for
            the expected shape (partners, mixing, layer_probs, cross_layer,
            duration, acts).
    """

    def __init__(self, layer='m', pars=None, **kwargs):
        if layer not in _KNOWN_LAYERS:
            raise ValueError(
                f'Unknown layer {layer!r}. Known: {list(_KNOWN_LAYERS)}.'
            )
        self.layer = layer
        kwargs.setdefault('name', layer)
        super().__init__()
        # Pars (when supplied via hpv.data.load_country):
        #   partners, duration, acts: ss.Dist instances (sampled via .rvs(uids))
        #   partners is per-sex: {'m': ss.Dist, 'f': ss.Dist}
        #   cross_layer is per-sex scalar: {'m': float, 'f': float}
        #   mixing, layer_probs: 2D ndarrays
        # Defaults are minimal placeholders; tests/scaffold paths short-
        # circuit add_pairs when the full pars set isn't supplied.
        self.define_pars(
            partners=None,
            mixing=None,
            layer_probs=None,
            cross_layer=None,
            duration=None,
            acts=None,
            debut=None,   # per-sex {'f': ss.Dist, 'm': ss.Dist} for sexual debut age
        )
        self.update_pars(pars=pars, **kwargs)
        # Per-agent desired partner count (sampled once when agent enters
        # the network).
        self.define_states(
            ss.FloatArr('partners_target', default=np.nan,
                        label='Desired partner count for this layer'),
        )

        # CRN-safe shuffle/sampling streams used inside add_pairs (one stream
        # per layer instance, so cross-layer order doesn't couple their RNG):
        # _dist_bin_order: shuffles the order in which female age-bins are
        #     processed when forming pairs (matches v2's randomized loop order).
        # _dist_f_select: when there aren't enough males in the male age-bin
        #     to satisfy a female bin, picks the female subset to drop.
        # _dist_cross_f/_dist_cross_m: Bernoulli for cross-layer concurrency
        #     filtering (replaces v2 hpu.binomial_filter, now CRN-safe).
        # _dist_participate: Bernoulli for age-bin participation filtering
        #     (replaces v2 hpu.participation_filter, now CRN-safe).
        # _dist_choose_m: weighted random choice for male selection per bin
        #     (replaces v2 hpu.choose_w, now CRN-safe).
        self._dist_bin_order = ss.choice(name=f'{layer}_bin_order', replace=False)
        self._dist_f_select = ss.choice(name=f'{layer}_f_select', replace=False)
        self._dist_cross_f = ss.bernoulli(p=0.5, name=f'{layer}_cross_f')
        self._dist_cross_m = ss.bernoulli(p=0.5, name=f'{layer}_cross_m')
        self._dist_participate = ss.bernoulli(p=0.5, name=f'{layer}_participate')
        self._dist_choose_m = ss.choice(a=2, replace=False, name=f'{layer}_choose_m')
        # Record formation timestep per edge so the partnership-equivalence
        # test can compute age-at-formation (matching v2's age_f/age_m in
        # to_df) and reconstruct original-duration (matching v2's stored dur).
        self.meta.start_ti = float

    def _other_layer_partner_uids(self):
        """ss.uids of agents currently partnered in any OTHER hpv.SexualNetwork
        layer. Filtered to ``hpv.SexualNetwork`` siblings via ``isinstance`` so
        non-sexual networks (e.g., maternal, environmental) don't contribute.
        Used by ``add_pairs`` to gate cross-layer concurrency.
        """
        endpoints = []
        for other in self.sim.networks():
            if other is self or not isinstance(other, SexualNetwork) or len(other) == 0:
                continue
            endpoints.append(other.edges.p1)
            endpoints.append(other.edges.p2)
        if not endpoints:
            return ss.uids()
        return ss.uids(np.unique(np.concatenate(endpoints)))

    def _init_partners_target(self, people):
        """Sample each agent's desired partner count for this layer, and
        their debut age (matches v2's per-sex debut distribution).

        v2 sampled the desired count once per agent at population creation
        and stored it in a static array. We do the same via the
        ``partners_target`` FloatArr state. We sample for any agent whose
        target is still NaN (uninitialized). Newly-born
        agents start as NaN and get a sample on first add_pairs after their
        birth. Debut is sampled into the parent ss.SexualNetwork's ``debut``
        FloatArr at the same time.
        """
        unset = self.partners_target.isnan.uids
        if not len(unset):
            return
        is_female = people.female[unset]
        f_uids = unset[is_female]
        m_uids = unset[~is_female]
        self.partners_target[f_uids] = self.pars.partners['f'].rvs(f_uids)
        self.partners_target[m_uids] = self.pars.partners['m'].rvs(m_uids)
        # Sample debut age per-sex (matches v2: female mean 15, male mean 17.6).
        # ss.SexualNetwork.active() requires people.age > self.debut, so this
        # gates young agents out of pairing.
        if self.pars.debut is not None:
            self.debut[f_uids] = self.pars.debut['f'].rvs(f_uids)
            self.debut[m_uids] = self.pars.debut['m'].rvs(m_uids)
        self.participant[unset] = True

    def _own_n_partners(self):
        """Per-UID count of own-layer edges, restricted to agents who have
        any.

        Returns ``(uids, counts)``: ``ss.uids`` of agents with at least one
        edge in this layer, and an int ndarray of their edge counts (in the
        same order). Used by ``add_pairs`` to identify agents at or above
        their per-layer ``partners_target``.
        """
        if not len(self.edges.p1):
            return ss.uids(), np.array([], dtype=int)
        endpoints = np.concatenate([np.asarray(self.edges.p1), np.asarray(self.edges.p2)])
        partnered = ss.uids(np.unique(endpoints))
        counts = np.bincount(endpoints)[partnered]
        return partnered, counts

    def _participation_filter(self, inds, age, layer_probs, bins):
        """Apply age-specific participation filter using Bernoulli draws.

        Replaces the v2 ``hpu.participation_filter`` helper. Uses
        ``self._dist_participate`` (a linked ``ss.bernoulli``) so draws are
        CRN-safe within the sim's RNG stream.

        Args:
            inds (ss.uids): candidate agent UIDs
            age (FloatArr): per-agent ages (indexed by UID)
            layer_probs (ndarray): per-bin participation probability (scalar per bin)
            bins (ndarray): age-bin boundaries

        Returns:
            ndarray of UIDs that passed the Bernoulli draw
        """
        if not len(inds):
            return np.array([], dtype=int)
        age_bins = np.digitize(age[inds], bins=bins) - 1
        participating = np.array([], dtype=int)
        for ab in np.unique(age_bins):
            bin_inds = inds[age_bins == ab]
            self._dist_participate.set(p=float(layer_probs[ab]))
            participating = np.concatenate(
                [participating, self._dist_participate.filter(bin_inds)]
            )
        return participating

    def add_pairs(self):
        """Form new partnerships in this layer for one timestep.

        Ported from v2 hpvsim/_v2_legacy/population.py:create_edgelist
        (lines 281-379). Adaptations from v2:

        - lno (layer index) -> self.layer (informational only)
        - current_partners[lno, :] count -> self._own_n_partners()
        - current_partners[other_layers, :].any(axis=0) -> self._other_layer_partner_uids()
        - current_partners updates -> self.append(...)
        - cluster (multi-cluster on hpvsim.People) -> single cluster on stock ss.People

        Females are placed in p1, males in p2, matching v2's convention.
        """
        # If this network was constructed with no pars (e.g. in scaffold
        # tests that only exercise the helper), there is nothing to do.
        if self.pars.partners is None or self.pars.layer_probs is None:
            return

        people = self.sim.people

        # Sample desired partner count for any agent that doesn't have one
        # yet (covers initial population AND newly-born agents on subsequent
        # timesteps).
        self._init_partners_target(people)

        # Eligible-for-new-partnership = active in this layer AND wants any
        # partners (``partners_target > 0``; v2's casual layer uses a plain
        # ``poisson`` so a large fraction of agents draw target=0) AND
        # own-layer edge count is below target. ``.asnew()`` is the
        # BoolArr-preserving copy (``.copy()`` would downgrade to a plain
        # ndarray).
        eligible = (self.active(people) & (self.partners_target > 0)).asnew()
        own_uids, own_counts = self._own_n_partners()
        if len(own_uids):
            saturated = own_uids[own_counts >= self.partners_target[own_uids]]
            eligible[saturated] = False

        # Cross-layer concurrency eligibility (mirror v2 lines 318-326). Of
        # agents currently partnered in OTHER hpv.SexualNetwork layers, only
        # those who pass the per-step cross_layer probability remain eligible.
        # cross_layer is an ss.prob (annual); ``.to_prob(dt)`` converts to a
        # dt-correct per-step probability (mirrors v2 sim step lines 461-469).
        dt = self.t.dt
        other_uids = self._other_layer_partner_uids()
        if len(other_uids):
            other_is_female = np.asarray(people.female[other_uids])
            other_f = other_uids[other_is_female]
            other_m = other_uids[~other_is_female]
            f_cross_p = self.pars.cross_layer['f'].to_prob(dt)
            m_cross_p = self.pars.cross_layer['m'].to_prob(dt)
            self._dist_cross_f.set(p=f_cross_p)
            self._dist_cross_m.set(p=m_cross_p)
            f_winners = self._dist_cross_f.filter(other_f)
            m_winners = self._dist_cross_m.filter(other_m)
            cross_winners = np.concatenate([f_winners, m_winners])
            cross_losers = ss.uids(np.setdiff1d(other_uids, cross_winners))
            eligible[cross_losers] = False

        # Split eligible by sex.
        elig_uids = eligible.uids
        elig_is_female = np.asarray(people.female[elig_uids])
        f_eligible_uids = elig_uids[elig_is_female]
        m_eligible_uids = elig_uids[~elig_is_female]

        # Bin participants by age (mirror v2 lines 330-339).
        # layer_probs is a dict with annual ss.prob arrays for f/m and a
        # plain ndarray for bins; convert to per-step probabilities here.
        bins = self.pars.layer_probs['bins']
        f_part_p = self.pars.layer_probs['f'].to_prob(dt)
        m_part_p = self.pars.layer_probs['m'].to_prob(dt)
        age = people.age
        m_participants = ss.uids(self._participation_filter(
            m_eligible_uids, age, m_part_p, bins,
        ))
        if len(m_participants) == 0:
            return  # no males available for pairing in any age bin

        age_bins_m = np.digitize(age[m_participants], bins=bins) - 1

        # Single-cluster handling: stock ss.People has no cluster array.
        # v2's loop `for cl in cluster_range` collapses to one iteration;
        # add_mixing[cl, cluster[m_participants]] reduces to a constant 1.
        f_cl = ss.uids(self._participation_filter(
            f_eligible_uids, age, f_part_p, bins,
        ))

        # ``paired_m`` tracks males already selected this timestep so they
        # aren't picked again across age-bin iterations (replaces the v2
        # raw-sized ``m_probs`` scratch buffer).
        paired_m = ss.BoolArr(people=people)

        # Accumulate selected pairs across age bins
        f_arr = np.array([], dtype=int)
        m_arr = np.array([], dtype=int)

        if len(f_cl) > 0:
            age_bins_f = np.digitize(age[f_cl], bins=bins) - 1
            bin_range_f, males_needed = np.unique(age_bins_f, return_counts=True)
            n_bins = len(bin_range_f)
            if n_bins > 1:
                self._dist_bin_order.set(a=np.arange(n_bins))
                bin_order = np.asarray(self._dist_bin_order.rvs(n_bins))
            else:
                bin_order = np.arange(n_bins)

            for ab, nm in zip(bin_range_f[bin_order], males_needed[bin_order]):
                # Female-of-age `ab` preferences over male age bins.
                # Weight each m_participant by mixing prob; males already paired
                # this timestep contribute 0 (v2 line 343 + the m_probs=0 reset).
                male_dist = self.pars.mixing[:, ab + 1]
                available_m = (~paired_m[m_participants]).astype(float)
                this_weighting = available_m * male_dist[age_bins_m]
                if this_weighting.sum() <= 0:
                    continue
                males_nonzero = this_weighting.nonzero()[0]
                this_weighting_nonzero = this_weighting[males_nonzero]
                f_inds = f_cl[age_bins_f == ab]
                if nm > len(this_weighting_nonzero):
                    # Not enough males — drop a CRN-safe random subset of females.
                    self._dist_f_select.set(a=f_inds)
                    f_selected = np.asarray(
                        self._dist_f_select.rvs(len(this_weighting_nonzero))
                    )
                    nm = f_selected.size
                else:
                    f_selected = f_inds
                norm_w = this_weighting_nonzero / this_weighting_nonzero.sum()
                self._dist_choose_m.set(a=len(this_weighting_nonzero), p=norm_w)
                m_selected = m_participants[
                    males_nonzero[np.asarray(self._dist_choose_m.rvs(nm))]
                ]
                paired_m[ss.uids(m_selected)] = True
                m_arr = np.concatenate((m_arr, m_selected))
                f_arr = np.concatenate((f_arr, f_selected))

        # Sample partnership durations and per-pair acts; append edges.
        # v2 placed females in p1, males in p2 (line 376 of create_edgelist).
        n_new = len(f_arr)
        if n_new == 0:
            return
        f_uids = ss.uids(f_arr.astype(int))
        m_uids = ss.uids(m_arr.astype(int))
        # ss.Dist.rvs takes uids; sample over the female-side uids of the
        # newly-formed pairs (per-pair, not per-agent — uids array length
        # = number of pairs).
        # v2's dur_pship is sampled in YEARS once per pair; Starsim's
        # DynamicNetwork decrements edges.dur by 1 each step (units of dt).
        # Divide here rather than dist-level unit-wrapping: nbinom (used
        # for marital dur) only supports predraw rate scaling, which would
        # inflate per-pair sample variance and break v2 equivalence.
        dur_years = self.pars.duration.rvs(f_uids)
        dur = dur_years / float(self.t.dt)
        acts = self.pars.acts.rvs(f_uids)
        beta = np.ones(n_new)
        start_ti = np.full(n_new, float(self.t.ti))
        self.append(
            p1=f_uids,
            p2=m_uids,
            beta=beta,
            dur=dur,
            acts=acts,
            start_ti=start_ti,
        )
        return n_new