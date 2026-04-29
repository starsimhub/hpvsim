"""HPVsim sexual partnership network.

Lift-and-shift of v2 hpvsim's two-layer (marital, casual) sexual network.
One class instantiated twice, one per layer; cross-layer concurrency
resolved at add_pairs time via isinstance-filtered iteration of sibling
networks. Inherits scaffolding (debut, participant, duration tracking,
end_pairs, net_beta) from ss.SexualNetwork.

Task 5 (this commit): class scaffold + cross-layer helper.
Task 6: port v2's create_edgelist into add_pairs.

Note on layer count: v2's default network has only two layers (m, c). An
earlier plan draft assumed a third 'o' (one-off) layer based on a
misleading comment in v2's parameters.py; verification confirmed only m
and c exist in v2 code.
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
        self.define_pars(
            partners=ss.poisson(lam=1),
            mixing=None,
            layer_probs=None,
            cross_layer=0.0,
            duration=ss.lognorm_ex(mean=ss.years(5)),
            acts=ss.poisson(lam=ss.freqperyear(50)),
        )
        self.update_pars(pars=pars, **kwargs)

    def _n_partners_elsewhere(self):
        """Count current partnerships each agent has in OTHER hpv.SexualNetwork
        layers. Used by add_pairs (Task 6) for cross-layer concurrency
        eligibility.

        Returns:
            np.ndarray of int, shape (n_agents,). Returns all zeros if no
            sibling SexualNetwork instances exist.
        """
        n = np.zeros(len(self.sim.people), dtype=int)
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

    def add_pairs(self):
        """Pair-formation logic - implemented in Task 6."""
        return