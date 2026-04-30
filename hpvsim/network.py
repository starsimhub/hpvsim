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

import hpvsim.utils as hpu


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
        # the network). FloatArr with default=NaN gives an unambiguous
        # "unsampled" sentinel; using IntArr with default=0 would collide
        # with legitimate Poisson(λ) draws that legitimately yield 0
        # (re-sampling would bias the distribution upward and break CRN
        # reproducibility). v2's partner-count distributions yield integers,
        # which we cast at sample-time.
        self.define_states(
            ss.FloatArr('partners_target', default=np.nan,
                        label='Desired partner count for this layer'),
        )
        # CRN-safe Dists for ordering/selection inside add_pairs. Each
        # ss.choice gets a unique seed via Starsim's per-Dist isolation;
        # configured per-call via .set(a=...) (MFNet pattern).
        self._dist_bin_order = ss.choice(name=f'{layer}_bin_order', replace=False)
        self._dist_f_select = ss.choice(name=f'{layer}_f_select', replace=False)

    def _n_partners_elsewhere(self):
        """Count current partnerships each agent has in OTHER hpv.SexualNetwork
        layers. Returns an int array sized at len(sim.people) — i.e., one
        entry per currently-alive agent's UID slot. Used by external code
        and tests; add_pairs uses _n_partners_elsewhere_uid_space directly.

        Returns:
            np.ndarray of int, shape (len(sim.people),). All zeros if no
            sibling SexualNetwork instances exist.
        """
        return self._n_partners_elsewhere_uid_space(len(self.sim.people))

    def _n_partners_elsewhere_uid_space(self, n_uids):
        """Same as _n_partners_elsewhere but returns an array sized at n_uids
        (the underlying agent storage size, which covers all live UIDs).
        Used by add_pairs to keep all per-agent arrays the same shape."""
        n = np.zeros(n_uids, dtype=int)
        for other in self.sim.networks():
            if other is self:
                continue
            if not isinstance(other, SexualNetwork):
                continue
            if len(other) == 0:
                continue
            n[other.edges.p1] += 1
            n[other.edges.p2] += 1
        return n

    def _init_partners_target(self, people):
        """Sample each agent's desired partner count for this layer, and
        their debut age (matches v2's per-sex debut distribution).

        v2 sampled the desired count once per agent at population creation
        and stored it in a static array. We do the same via the
        ``partners_target`` FloatArr state, which Starsim auto-resizes on
        births/deaths. We sample for any agent whose target is still NaN
        (uninitialized) — using a NaN sentinel rather than 0 because v2's
        Poisson(λ=1) for casual partners legitimately yields 0 ~36% of the
        time, and re-sampling those would bias the distribution. Newly-born
        agents start as NaN and get a sample on first add_pairs after their
        birth. Debut is sampled into the parent ss.SexualNetwork's ``debut``
        FloatArr at the same time.
        """
        unset_uids = self.partners_target.isnan.uids
        # Limit to currently-alive agents (Starsim's nan-mask covers all
        # ever-allocated UIDs, including dead ones we shouldn't sample for).
        alive_unset = unset_uids[np.asarray(people.alive[unset_uids])]
        if not len(alive_unset):
            return
        is_female_unset = people.female[alive_unset]
        f_uids = alive_unset[is_female_unset]
        m_uids = alive_unset[~is_female_unset]
        if len(f_uids):
            self.partners_target[f_uids] = self.pars.partners['f'].rvs(f_uids)
        if len(m_uids):
            self.partners_target[m_uids] = self.pars.partners['m'].rvs(m_uids)
        # Sample debut age per-sex (matches v2: female mean 15, male mean 17.6).
        # ss.SexualNetwork.active() requires people.age > self.debut, so this
        # gates young agents out of pairing.
        if self.pars.debut is not None:
            if len(f_uids):
                self.debut[f_uids] = self.pars.debut['f'].rvs(f_uids)
            if len(m_uids):
                self.debut[m_uids] = self.pars.debut['m'].rvs(m_uids)
        self.participant[unset_uids] = True
        return

    def _own_n_partners(self):
        """Number of edges in *this* layer touching each currently-alive agent.

        Returns an int array indexed in the same UID space as the network's
        ``partners_target`` FloatArr — i.e., the underlying agent storage
        size, which always covers all live UIDs.
        """
        n_uids = len(self.partners_target.raw)
        n = np.zeros(n_uids, dtype=int)
        if len(self.edges.p1):
            np.add.at(n, np.asarray(self.edges.p1), 1)
            np.add.at(n, np.asarray(self.edges.p2), 1)
        return n

    def add_pairs(self):
        """Form new partnerships in this layer for one timestep.

        Ported from v2 hpvsim/_v2_legacy/population.py:create_edgelist
        (lines 281-379). Adaptations from v2:

        - lno (layer index) -> self.layer (informational only)
        - current_partners[lno, :] count -> self._own_n_partners()
        - current_partners[other_layers, :].any(axis=0) -> self._n_partners_elsewhere() > 0
        - current_partners updates -> self.append(...)
        - cluster (multi-cluster on hpvsim.People) -> single cluster on stock ss.People

        Females are placed in p1, males in p2, matching v2's convention.
        """
        # If this network was constructed with no pars (e.g. in scaffold
        # tests that only exercise the helper), there is nothing to do.
        if self.pars.partners is None or self.pars.layer_probs is None:
            return

        people = self.sim.people
        n_agents = len(people)

        # Sample desired partner count for any agent that doesn't have one
        # yet (covers initial population AND newly-born agents on subsequent
        # timesteps). FloatArr auto-resizes; _init samples only the unset.
        self._init_partners_target(people)

        # Work in UID space so newly-born / dead agents resize cleanly.
        n_uids = len(self.partners_target.raw)
        is_female = np.asarray(people.female.raw, dtype=bool)
        # ss.SexualNetwork.active() returns a BoolArr indexed by alive uids;
        # project it into a full-size ndarray via the BoolArr's True UIDs.
        active_mask = np.zeros(n_uids, dtype=bool)
        active_mask[self.active(people).uids] = True

        f_active = is_female & active_mask
        m_active = ~is_female & active_mask
        underpartnered = self._own_n_partners() < np.asarray(self.partners_target.raw)

        # Cross-layer concurrency eligibility (mirror v2 lines 318-326).
        # In v2 every layer has its own current_partners row; we use the
        # SexualNetwork-only sibling count, which is equivalent for M01.
        n_elsewhere = self._n_partners_elsewhere_uid_space(n_uids)
        other_partners = n_elsewhere > 0
        f_with_other = (other_partners & is_female).nonzero()[0]
        m_with_other = (other_partners & ~is_female).nonzero()[0]
        f_cross_inds = hpu.binomial_filter(self.pars.cross_layer['f'], f_with_other)
        m_cross_inds = hpu.binomial_filter(self.pars.cross_layer['m'], m_with_other)
        cross_layer_bools = np.zeros(n_uids, dtype=bool)
        cross_layer_bools[f_cross_inds] = True
        cross_layer_bools[m_cross_inds] = True

        f_eligible = f_active & underpartnered & (~other_partners | cross_layer_bools)
        m_eligible = m_active & underpartnered & (~other_partners | cross_layer_bools)

        # Bin participants by age (mirror v2 lines 330-339).
        layer_probs = self.pars.layer_probs
        bins = layer_probs[0, :]
        age = np.asarray(people.age.raw)
        m_eligible_inds = m_eligible.nonzero()[0]
        m_participants = hpu.participation_filter(
            m_eligible_inds, age, layer_probs[2, :], bins=bins,
        )
        if len(m_participants) == 0:
            return  # no males available for pairing in any age bin

        age_bins_m = np.digitize(age[m_participants], bins=bins) - 1
        m_probs = np.ones(n_uids)  # equal initial weighting (v2 line 343)

        # Single-cluster handling: stock ss.People has no cluster array.
        # v2's loop `for cl in cluster_range` collapses to one iteration;
        # add_mixing[cl, cluster[m_participants]] reduces to a constant 1.
        f_eligible_inds = f_eligible.nonzero()[0]
        f_cl = hpu.participation_filter(
            f_eligible_inds, age, layer_probs[1, :], bins=bins,
        )

        # Accumulate selected pairs across age bins
        f_arr = np.array([], dtype=int)
        m_arr = np.array([], dtype=int)

        if len(f_cl) > 0:
            age_bins_f = np.digitize(age[f_cl], bins=bins) - 1
            bin_range_f, males_needed = np.unique(age_bins_f, return_counts=True)
            # CRN-safe permutation of age-bin processing order via ss.choice.
            n_bins = len(bin_range_f)
            if n_bins > 1:
                self._dist_bin_order.set(a=np.arange(n_bins))
                bin_order = np.asarray(self._dist_bin_order.rvs(n_bins))
            else:
                bin_order = np.arange(n_bins)

            for ab, nm in zip(bin_range_f[bin_order], males_needed[bin_order]):
                # Female-of-age `ab` preferences over male age bins
                male_dist = self.pars.mixing[:, ab + 1]
                this_weighting = m_probs[m_participants] * male_dist[age_bins_m]
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
                m_selected = m_participants[
                    males_nonzero[hpu.choose_w(this_weighting_nonzero, nm)]
                ]
                m_probs[m_selected] = 0  # remove males that just got paired
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
        dur = self.pars.duration.rvs(f_uids)
        acts = self.pars.acts.rvs(f_uids)
        beta = np.ones(n_new)
        self.append(
            p1=f_uids,
            p2=m_uids,
            beta=beta,
            dur=dur,
            acts=acts,
        )
        return n_new