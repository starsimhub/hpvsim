"""
Connector for HPVsim which unites results and attributes across genotypes
"""

import starsim as ss

__all__ = ["hpv_hiv_connector"]


class hpv_hiv_connector(ss.Connector):
    def __init__(self, hpv=None, hiv=None, pars=None, **kwargs):
        super().__init__()
        if not isinstance(hpv, ss.Connector):
            print("We are expecting the HPV superconnector here")
            self.hpv = hpv
        else:
            self.hpv = hpv
        self.hiv = hiv
        self.define_pars(
            rel_sus_hpv=lambda cd4: [2.2 if i < 200 else 1.5 for i in cd4],
            rel_sev_hpv=lambda cd4: [2.2 if i < 200 else 1.5 for i in cd4],
        )
        self.update_pars(pars, **kwargs)

    def step(self):
        hiv_inds = self.hiv.infected.true()
        self.hpv.rel_sev[hiv_inds] = self.pars.rel_sev_hpv(self.hiv.cd4[hiv_inds])
        self.hpv.rel_sus[hiv_inds] = self.pars.rel_sus_hpv(self.hiv.cd4[hiv_inds])
        return

