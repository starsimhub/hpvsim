"""Test-only SexualNetwork variant using v2's RNG plumbing.

Mirrors hpvsim.network.SexualNetwork.add_pairs and _init_partners_target
algorithmically, but swaps every Starsim-managed source of randomness for
the equivalent numpy / hpvsim.utils.sample call:

  - ss.poisson / ss.lognorm_ex / ss.nbinom (.rvs)  -> hpu.sample(**v2_dict)
  - ss.choice (.set, .rvs)                          -> np.random.choice / shuffle

This isolates the RNG-plumbing dimension of v2/v3 divergence: if running
the partnership-equivalence test with this network produces *closer*
agreement to v2 than the production network, the divergence is
RNG-stream-driven (not an algorithmic bug). If divergence is unchanged
(or larger), an algorithmic difference is responsible.

NOT for production use — bypasses Starsim's CRN-safe Dist machinery.
"""

import numpy as np

import hpvsim.utils as hpu
import hpvsim.parameters as _P
from hpvsim.network import SexualNetwork


class LegacyRngSexualNetwork(SexualNetwork):
    """v2-RNG-equivalent variant of hpv.SexualNetwork.

    Constructor reads raw v2-format distribution dicts from
    ``hpvsim.parameters.make_pars`` directly (bypassing
    ``hpv.data.load_country``'s ss.Dist conversion). All sampling inside
    ``add_pairs`` and ``_init_partners_target`` uses ``hpu.sample`` and
    ``np.random.*`` against numpy's global RNG, matching v2's
    ``create_edgelist`` plumbing.

    Args:
        layer: 'm' or 'c' (matching v2's two layers).
        location: passed to v2's ``parameters.make_pars`` for raw v2 pars.
    """

    def __init__(self, layer='m', location='nigeria', **kwargs):
        # parameters.make_pars already populates ['mixing'] per-layer;
        # no need to call get_mixing separately (which returns a tuple).
        v2_pars = _P.make_pars(location=location)
        # Stash raw v2-format dicts; the parent SexualNetwork stores
        # ss.Dist instances in self.pars but we override every sample site.
        self._v2 = dict(
            partners={
                'm': v2_pars['m_partners'][layer],
                'f': v2_pars['f_partners'][layer],
            },
            duration=v2_pars['dur_pship'][layer],
            acts=v2_pars['acts'][layer],
            debut={
                'm': v2_pars['debut']['m'],
                'f': v2_pars['debut']['f'],
            },
            layer_probs=v2_pars['layer_probs'][layer],
            mixing=v2_pars['mixing'][layer],
            cross_layer={
                'm': v2_pars['m_cross_layer'],
                'f': v2_pars['f_cross_layer'],
            },
        )
        # Pass an "always populated" pars dict so the parent's add_pairs
        # short-circuit `if pars.partners is None` never trips.
        pars = dict(
            partners=self._v2['partners'],   # not actually sampled from
            mixing=self._v2['mixing'],
            layer_probs=self._v2['layer_probs'],
            cross_layer=self._v2['cross_layer'],
            duration=self._v2['duration'],
            acts=self._v2['acts'],
            debut=self._v2['debut'],
        )
        super().__init__(layer=layer, pars=pars, **kwargs)

    def _init_partners_target(self, people):
        """v2-RNG version: hpu.sample for partner counts and debut."""
        unset_alive = (people.alive & self.partners_target.isnan).uids
        if not len(unset_alive):
            return
        is_female = people.female[unset_alive]
        f_uids = unset_alive[is_female]
        m_uids = unset_alive[~is_female]
        if len(f_uids):
            self.partners_target[f_uids] = hpu.sample(
                size=len(f_uids), **self._v2['partners']['f']
            )
            self.debut[f_uids] = hpu.sample(
                size=len(f_uids), **self._v2['debut']['f']
            )
        if len(m_uids):
            self.partners_target[m_uids] = hpu.sample(
                size=len(m_uids), **self._v2['partners']['m']
            )
            self.debut[m_uids] = hpu.sample(
                size=len(m_uids), **self._v2['debut']['m']
            )
        self.participant[unset_alive] = True

    def add_pairs(self):
        """v2-RNG version of SexualNetwork.add_pairs.

        Algorithmically identical to the production add_pairs but swaps
        every Starsim-managed source of randomness for an equivalent
        numpy / hpu.sample call.
        """
        people = self.sim.people
        self._init_partners_target(people)

        is_female = people.female.raw
        active_mask = self.active(people).raw
        f_active = is_female & active_mask
        m_active = ~is_female & active_mask
        underpartnered = self._own_n_partners() < self.partners_target.raw

        n_uids = len(self.partners_target.raw)
        n_elsewhere = self._n_partners_elsewhere()
        other_partners = n_elsewhere > 0
        f_with_other = (other_partners & is_female).nonzero()[0]
        m_with_other = (other_partners & ~is_female).nonzero()[0]
        f_cross_inds = hpu.binomial_filter(self._v2['cross_layer']['f'], f_with_other)
        m_cross_inds = hpu.binomial_filter(self._v2['cross_layer']['m'], m_with_other)
        cross_layer_bools = np.zeros(n_uids, dtype=bool)
        cross_layer_bools[f_cross_inds] = True
        cross_layer_bools[m_cross_inds] = True

        f_eligible = f_active & underpartnered & (~other_partners | cross_layer_bools)
        m_eligible = m_active & underpartnered & (~other_partners | cross_layer_bools)

        layer_probs = self._v2['layer_probs']
        bins = layer_probs[0, :]
        age = people.age.raw
        m_eligible_inds = m_eligible.nonzero()[0]
        m_participants = hpu.participation_filter(
            m_eligible_inds, age, layer_probs[2, :], bins=bins,
        )
        if len(m_participants) == 0:
            return

        age_bins_m = np.digitize(age[m_participants], bins=bins) - 1
        m_probs = np.ones(n_uids)

        f_eligible_inds = f_eligible.nonzero()[0]
        f_cl = hpu.participation_filter(
            f_eligible_inds, age, layer_probs[1, :], bins=bins,
        )

        f_arr = np.array([], dtype=int)
        m_arr = np.array([], dtype=int)

        if len(f_cl) > 0:
            age_bins_f = np.digitize(age[f_cl], bins=bins) - 1
            bin_range_f, males_needed = np.unique(age_bins_f, return_counts=True)
            bin_order = np.arange(len(bin_range_f))
            np.random.shuffle(bin_order)  # v2 plumbing: np.random global

            for ab, nm in zip(bin_range_f[bin_order], males_needed[bin_order]):
                male_dist = self._v2['mixing'][:, ab + 1]
                this_weighting = m_probs[m_participants] * male_dist[age_bins_m]
                if this_weighting.sum() <= 0:
                    continue
                males_nonzero = this_weighting.nonzero()[0]
                this_weighting_nonzero = this_weighting[males_nonzero]
                f_inds = f_cl[age_bins_f == ab]
                if nm > len(this_weighting_nonzero):
                    # v2 plumbing: np.random.choice on the global stream
                    f_selected = f_inds[np.random.choice(
                        len(f_inds), len(this_weighting_nonzero), replace=False,
                    )]
                    nm = f_selected.size
                else:
                    f_selected = f_inds
                m_selected = m_participants[
                    males_nonzero[hpu.choose_w(this_weighting_nonzero, nm)]
                ]
                m_probs[m_selected] = 0
                m_arr = np.concatenate((m_arr, m_selected))
                f_arr = np.concatenate((f_arr, f_selected))

        n_new = len(f_arr)
        if n_new == 0:
            return
        # v2 plumbing for duration and acts: hpu.sample from raw dicts.
        dur = hpu.sample(size=n_new, **self._v2['duration'])
        acts = hpu.sample(size=n_new, **self._v2['acts'])
        beta = np.ones(n_new)
        import starsim as ss
        f_uids = ss.uids(f_arr.astype(int))
        m_uids = ss.uids(m_arr.astype(int))
        start_ti = np.full(n_new, float(self.t.ti))
        self.append(
            p1=f_uids, p2=m_uids, beta=beta,
            dur=dur, acts=acts, start_ti=start_ti,
        )
        return n_new