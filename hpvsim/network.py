"""HPVsim sexual partnership network.

Multi-layer heterosexual partnership network. A single ``SexualNetwork``
instance holds all partnership layers (e.g. marital ``m`` + casual ``c``)
in one ``edges`` table tagged by ``layer_id``. ``debut`` and ``participant``
are inherited from ``ss.SexualNetwork`` as single per-agent values shared
across all layers. ``step()`` dissolves all pairs (single ``end_pairs`` on
the combined edges table) and then forms new pairs per layer in sequence,
so no sibling-network coordination is needed.
"""

import numpy as np
import starsim as ss


def _age_scale_acts(acts, age_act_pars, age_f, age_m, debut_f, debut_m):
    """Age-scaled acts: piecewise-linear modulation by couple's average age.

    Below peak: ramp from ``debut_ratio`` at debut to 1.0 at peak.
    Above peak (below retirement): ramp from 1.0 to ``retirement_ratio``.
    Above retirement: 0 (default-initialized).
    """
    avg_age = (age_f + age_m) / 2.0
    avg_debut = (debut_f + debut_m) / 2.0

    dr = age_act_pars['debut_ratio']
    peak = age_act_pars['peak']
    rr = age_act_pars['retirement_ratio']
    retire = age_act_pars['retirement']

    below = avg_age <= peak
    above = (avg_age > peak) & (avg_age < retire)

    scaled = np.zeros(len(acts))
    if below.any():
        scaled[below] = acts[below] * (
            dr + (1 - dr) / (peak - avg_debut[below]) * (avg_age[below] - avg_debut[below])
        )
    if above.any():
        scaled[above] = acts[above] * (
            rr + (1 - rr) / (peak - retire) * (avg_age[above] - retire)
        )
    return scaled


class SexualNetwork(ss.SexualNetwork):
    """Multi-layer heterosexual partnership network.

    Args:
        layer_pars: dict ``{layer_name: layer_pars_dict}``. Each layer dict
            (when supplied via ``hpv.data.load_country``) carries:
            ``partners`` (per-sex Dist dict), ``mixing`` (2D ndarray),
            ``layer_probs`` (dict with bins/f/m), ``cross_layer`` (per-sex
            ss.prob dict), ``duration``, ``acts`` (Dist instances).
        debut: per-sex debut-age distribution dict ``{'f': Dist, 'm': Dist}``.
            Sampled once per agent and shared across all layers.

    Overrides ``net_beta`` to compound the per-act probability across the
    per-edge ``acts`` count: ``1 - (1 - edges.beta * disease_beta)^acts``.
    The starsim base treats ``edges.beta`` as a final scalar
    (``edges.beta * (1 - (1 - disease_beta)^acts)``); this override treats
    it as a per-act probability multiplier, which is the v2 convention.
    """

    def net_beta(self, disease_beta=None, inds=None, disease=None):
        if inds is None:
            inds = Ellipsis
        p_per_act = self.edges.beta[inds] * disease_beta
        return 1 - (1 - p_per_act) ** self.edges.acts[inds]

    def __init__(self, layer_pars=None, debut=None, **kwargs):
        super().__init__()
        self.define_pars(
            layer_pars=layer_pars,
            debut=debut,
        )
        self.update_pars(**kwargs)
        # Layer ordering: tuple keys in insertion order. Empty when the
        # network is constructed without layer_pars (scaffold tests).
        self.layers = tuple(self.pars.layer_pars.keys()) if self.pars.layer_pars else ()
        self._layer_idx = {lkey: i for i, lkey in enumerate(self.layers)}
        # Per-layer partners_target — partners are sampled with independent
        # Poisson draws per layer, so per-layer state is required.
        if self.layers:
            target_states = [
                ss.FloatArr(f'partners_target_{lkey}', default=np.nan,
                            label=f'Desired partner count ({lkey})')
                for lkey in self.layers
            ]
            self.define_states(*target_states)
        # Edge metadata: per-edge layer tag + formation timestep. start_ti
        # lets diagnostics reconstruct age-at-formation and original duration.
        self.meta.layer_id = int
        self.meta.start_ti = float
        # CRN-safe distributions used inside ``_add_pairs_for_layer``. Each
        # layer gets its own set so layers' RNG state is independent;
        # sharing a single set would couple them and inflate cross-layer
        # covariance.
        self._dists = {
            lkey: dict(
                bin_order=ss.choice(name=f'{lkey}_bin_order', replace=False),
                f_select=ss.choice(name=f'{lkey}_f_select', replace=False),
                cross_f=ss.bernoulli(p=0.5, name=f'{lkey}_cross_f'),
                cross_m=ss.bernoulli(p=0.5, name=f'{lkey}_cross_m'),
                participate=ss.bernoulli(p=0.5, name=f'{lkey}_participate'),
                choose_m=ss.choice(a=2, replace=False, name=f'{lkey}_choose_m'),
            )
            for lkey in self.layers
        }

    # ------------------------------------------------------------------ #
    # Public per-layer accessors (used by tests / diagnostics)             #
    # ------------------------------------------------------------------ #

    def edges_for_layer(self, lkey):
        """Boolean mask into ``self.edges`` selecting edges in layer ``lkey``."""
        if lkey not in self._layer_idx:
            raise KeyError(f'Unknown layer {lkey!r}; known: {list(self.layers)}')
        return np.asarray(self.edges.layer_id) == self._layer_idx[lkey]

    def n_pairs_in_layer(self, lkey):
        """Active pair count in layer ``lkey``."""
        return int(self.edges_for_layer(lkey).sum())

    # ------------------------------------------------------------------ #
    # Lifecycle hooks                                                      #
    # ------------------------------------------------------------------ #

    def init_post(self):
        super().init_post()
        self.set_network_states()

    def set_network_states(self):
        """Sample ``debut``, ``participant``, and per-layer ``partners_target``
        for any alive agent without initialized state (``participant=False``).

        Runs once at init for the starting population and once per step to
        handle newly-added agents (births, AgeMigration immigrants).
        """
        if not self.layers:
            return
        unset = (~self.participant).uids
        if not len(unset):
            return
        people = self.sim.people
        is_female = people.female[unset]
        f_uids = unset[is_female]
        m_uids = unset[~is_female]

        # Per-layer partners_target — independent samples per layer.
        for lkey in self.layers:
            lpars = self.pars.layer_pars[lkey]
            partners = lpars.get('partners') if lpars else None
            if partners is None:
                continue
            arr = getattr(self, f'partners_target_{lkey}')
            arr[f_uids] = partners['f'].rvs(f_uids)
            arr[m_uids] = partners['m'].rvs(m_uids)

        # Single shared debut sample per agent (across layers).
        if self.pars.debut is not None:
            self.debut[f_uids] = self.pars.debut['f'].rvs(f_uids)
            self.debut[m_uids] = self.pars.debut['m'].rvs(m_uids)

        self.participant[unset] = True

    def step(self):
        """Dissolve all pairs (single ``end_pairs`` on the combined edges
        table), then form new pairs per layer in sequence.
        """
        self.end_pairs()
        self.set_network_states()
        for lkey in self.layers:
            self._add_pairs_for_layer(lkey)

    # ------------------------------------------------------------------ #
    # Per-layer helpers                                                    #
    # ------------------------------------------------------------------ #

    def _own_n_partners_in_layer(self, lkey):
        """``(uids, counts)`` for agents with at least one edge in layer ``lkey``."""
        mask = self.edges_for_layer(lkey)
        if not mask.any():
            return ss.uids(), np.array([], dtype=int)
        p1 = np.asarray(self.edges.p1)[mask]
        p2 = np.asarray(self.edges.p2)[mask]
        endpoints = np.concatenate([p1, p2])
        partnered = ss.uids(np.unique(endpoints))
        counts = np.bincount(endpoints)[partnered]
        return partnered, counts

    def _other_layer_partner_uids(self, lkey):
        """ss.uids of agents partnered in any layer OTHER than ``lkey``.

        Called after ``end_pairs`` so cross-layer eligibility doesn't see
        dissolved pairs.
        """
        mask = ~self.edges_for_layer(lkey)
        if not mask.any():
            return ss.uids()
        p1 = np.asarray(self.edges.p1)[mask]
        p2 = np.asarray(self.edges.p2)[mask]
        return ss.uids(np.unique(np.concatenate([p1, p2])))

    def _participation_filter(self, inds, age, layer_probs, bins, dist):
        """Apply age-bin Bernoulli participation filter using ``dist`` (a
        per-layer CRN-safe ``ss.bernoulli``).
        """
        if not len(inds):
            return np.array([], dtype=int)
        age_bins = np.digitize(age[inds], bins=bins) - 1
        participating = np.array([], dtype=int)
        for ab in np.unique(age_bins):
            bin_inds = inds[age_bins == ab]
            dist.set(p=float(layer_probs[ab]))
            participating = np.concatenate(
                [participating, dist.filter(bin_inds)]
            )
        return participating

    def _add_pairs_for_layer(self, lkey):
        """Form new partnerships in layer ``lkey`` for one timestep.

        Females are placed in ``p1``, males in ``p2``.
        """
        lpars = self.pars.layer_pars[lkey]
        # Scaffolding short-circuit: layers with no pars (test fixtures) skip.
        if not lpars or lpars.get('partners') is None or lpars.get('layer_probs') is None:
            return

        people = self.sim.people
        target = getattr(self, f'partners_target_{lkey}')
        dists = self._dists[lkey]
        dt = self.t.dt
        dt_yr = float(dt)

        # Eligibility: alive & past debut & participant & wants partners
        # & not already at target.
        eligible = (self.active(people) & (target > 0)).asnew()
        own_uids, own_counts = self._own_n_partners_in_layer(lkey)
        if len(own_uids):
            saturated = own_uids[own_counts >= target[own_uids]]
            eligible[saturated] = False

        # Cross-layer concurrency filter: agents partnered in OTHER layers
        # only stay eligible if they pass the per-step cross_layer Bernoulli.
        # cross_layer is annual ``ss.prob``; ``.to_prob(dt)`` converts to
        # dt-correct per-step probability.
        other_uids = self._other_layer_partner_uids(lkey)
        if len(other_uids):
            other_is_female = people.female[other_uids]
            other_f = other_uids[other_is_female]
            other_m = other_uids[~other_is_female]
            f_cross_p = lpars['cross_layer']['f'].to_prob(dt)
            m_cross_p = lpars['cross_layer']['m'].to_prob(dt)
            dists['cross_f'].set(p=f_cross_p)
            dists['cross_m'].set(p=m_cross_p)
            f_winners = dists['cross_f'].filter(other_f)
            m_winners = dists['cross_m'].filter(other_m)
            cross_winners = np.concatenate([f_winners, m_winners])
            cross_losers = ss.uids(np.setdiff1d(other_uids, cross_winners))
            eligible[cross_losers] = False

        # Split eligible by sex.
        elig_uids = eligible.uids
        elig_is_female = people.female[elig_uids]
        f_eligible_uids = elig_uids[elig_is_female]
        m_eligible_uids = elig_uids[~elig_is_female]

        # Bin participants by age.
        bins = lpars['layer_probs']['bins']
        f_part_p = lpars['layer_probs']['f'].to_prob(dt)
        m_part_p = lpars['layer_probs']['m'].to_prob(dt)
        age = people.age
        m_participants = ss.uids(self._participation_filter(
            m_eligible_uids, age, m_part_p, bins, dists['participate'],
        ))
        if len(m_participants) == 0:
            return  # no males to pair in this layer this step

        age_bins_m = np.digitize(age[m_participants], bins=bins) - 1
        f_cl = ss.uids(self._participation_filter(
            f_eligible_uids, age, f_part_p, bins, dists['participate'],
        ))

        # Tracks males already paired during this step's bin loop.
        paired_m = ss.BoolArr(people=people)
        f_arr = np.array([], dtype=int)
        m_arr = np.array([], dtype=int)

        if len(f_cl) > 0:
            age_bins_f = np.digitize(age[f_cl], bins=bins) - 1
            bin_range_f, males_needed = np.unique(age_bins_f, return_counts=True)
            n_bins = len(bin_range_f)
            if n_bins > 1:
                dists['bin_order'].set(a=np.arange(n_bins))
                bin_order = dists['bin_order'].rvs(n_bins)
            else:
                bin_order = np.arange(n_bins)

            for ab, nm in zip(bin_range_f[bin_order], males_needed[bin_order]):
                male_dist = lpars['mixing'][:, ab + 1]
                available_m = (~paired_m[m_participants]).astype(float)
                this_weighting = available_m * male_dist[age_bins_m]
                if this_weighting.sum() <= 0:
                    continue
                males_nonzero = this_weighting.nonzero()[0]
                this_weighting_nonzero = this_weighting[males_nonzero]
                f_inds = f_cl[age_bins_f == ab]
                if nm > len(this_weighting_nonzero):
                    # Not enough males: drop a CRN-safe random subset of females.
                    dists['f_select'].set(a=f_inds)
                    f_selected = dists['f_select'].rvs(len(this_weighting_nonzero))
                    nm = f_selected.size
                else:
                    f_selected = f_inds
                norm_w = this_weighting_nonzero / this_weighting_nonzero.sum()
                dists['choose_m'].set(a=len(this_weighting_nonzero), p=norm_w)
                m_selected = m_participants[
                    males_nonzero[dists['choose_m'].rvs(nm)]
                ]
                paired_m[ss.uids(m_selected)] = True
                m_arr = np.concatenate((m_arr, m_selected))
                f_arr = np.concatenate((f_arr, f_selected))

        n_new = len(f_arr)
        if n_new == 0:
            return
        f_uids = ss.uids(f_arr.astype(int))
        m_uids = ss.uids(m_arr.astype(int))
        # Duration is sampled in years per pair; ``ss.DynamicNetwork.end_pairs``
        # decrements ``edges.dur`` by 1 each step (units of dt). Divide here
        # rather than at dist-level — nbinom only supports predraw rate
        # scaling, which would inflate per-pair variance.
        dur_years = lpars['duration'].rvs(f_uids)
        dur = dur_years / dt_yr
        # Sample raw per-year acts, then apply age-based modulation, then
        # scale to per-step. age_act_pars is optional — older test fixtures
        # may not supply it, in which case we skip the modulation (equivalent
        # to age=peak for everyone).
        raw_acts = lpars['acts'].rvs(f_uids)
        age_pars = lpars.get('age_act_pars')
        if age_pars is not None:
            age_f = people.age[f_uids]
            age_m = people.age[m_uids]
            debut_f = self.debut[f_uids]
            debut_m = self.debut[m_uids]
            raw_acts = _age_scale_acts(
                raw_acts, age_pars, age_f, age_m, debut_f, debut_m
            )
        acts = raw_acts * dt_yr
        beta = np.ones(n_new)
        start_ti = np.full(n_new, float(self.t.ti))
        layer_id = np.full(n_new, self._layer_idx[lkey], dtype=int)
        self.append(
            p1=f_uids,
            p2=m_uids,
            beta=beta,
            dur=dur,
            acts=acts,
            start_ti=start_ti,
            layer_id=layer_id,
        )
        return n_new