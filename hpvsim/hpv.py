"""
Module for HPV
"""

import numpy as np
import starsim as ss
import sciris as sc
import stisim as sti
import hpvsim as hpv

__all__ = ["HPV", "HPV16", "HPV18"]


class HPV(sti.BaseSTI):
    """
    Base class for a single genotype of HPV
    """

    def __init__(self, name="hpv", genotype="16", pars=None, **kwargs):
        super().__init__(name=f"{name}{genotype}")

        self.define_pars(
            unit="month",
            init_prev=ss.bernoulli(p=0.2),
            beta=0.1,
            beta_m2f=1,
            rel_beta_f2m=0.27,
            beta_m2c=0,
            eff_condom=0.5,
            dur_cancer=ss.lognorm_ex(ss.dur(8, "year"), ss.dur(3, "year")),
            dur_infection_male=ss.lognorm_ex(ss.dur(1, "year"), ss.dur(1, "year")),
            sero_prob=ss.bernoulli(p=0.75),
            init_imm=hpv.beta(a=2, b=2),
            init_cell_imm=hpv.beta(a=2, b=2),
            rel_beta=1,
            dur_precin=None,    # Set for individual genotypes by derived classes
            dur_cin=None,       # Set for individual genotypes by derived classes
            cin_fn=None,        # Set for individual genotypes by derived classes
            transform_prob=None,  # Set for individual genotypes by derived classes
            cin_prob=ss.bernoulli(p=0),     # placeholder, gets reset
            cancer_prob=ss.bernoulli(p=0),  # placeholder, gets reset
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            # States
            ss.State("latent", label="latent"),
            ss.State("precin", label="precin"),
            ss.State("cin", label="cin"),
            ss.State("cancerous", label="cancerous"),

            # Duration and timestep of states
            ss.FloatArr("dur_precin", label="Duration of precin"),
            ss.FloatArr("dur_cin", label="Duration of cin"),
            ss.FloatArr("dur_cancer", label="Duration of cancer"),
            ss.FloatArr("ti_cancer", label="Timestep of cancer"),
            ss.FloatArr("ti_cancer_death", label="Timestep of cancer death"),
            ss.FloatArr("ti_cin", label="Timestep of CIN"),
            ss.FloatArr("ti_clearance", label="Timestep of clearance"),

            # Immunity states
            ss.FloatArr("rel_sev", default=1, label="relative severity"),
            ss.FloatArr("sus_imm", default=0, label="immunity to infection"),
            ss.FloatArr("sev_imm", default=0, label="immunity to severe disease"),
        )

    def init_results(self):
        super().init_results()
        results = [
            ss.Result("cins", label="CINs"),
            ss.Result("cancers", label="Cancers"),
            ss.Result("cancer_incidence", label="Cancer incidence", scale=False),
            ss.Result("cancer_deaths", label="Cancer deaths"),
        ]
        self.define_results(*results)
        return

    def step_state(self):
        """
        Update states prior to transmission
        """
        self.rel_sus[:] = 1
        self.rel_sev[:] = 1
        self.update_infection()
        return

    def set_prognoses(self, uids, sources=None):
        """
        Set the prognoses for people infected with HPV
        """

        # First separate out men and women
        m_uids = uids.intersect(self.sim.people.male.uids)
        f_uids = uids.intersect(self.sim.people.female.uids)
        self.infectious[uids] = True
        self.precin[f_uids] = True
        self.susceptible[uids] = False

        # Deal with men first
        dur_inf_male = self.pars.dur_infection_male.rvs(m_uids)
        self.ti_clearance[m_uids] = self.ti + dur_inf_male

        # Set the duration of precin and determine who will progress to CIN
        dur_precin = self.pars.dur_precin.rvs(f_uids)  # * self.sev_imm[uids]
        self.dur_precin[f_uids] = dur_precin
        cin_probs = self.get_cin_prob(f_uids)

        self.pars.cin_prob.set(cin_probs)
        cin, no_cin = self.pars.cin_prob.split(f_uids)
        self.ti_cin[cin] = self.ti + self.dur_precin[cin]
        self.ti_clearance[no_cin] = self.ti + self.dur_precin[no_cin]

        # Set the duration of CIN and determine who will progress to cancer
        dur_cin = self.pars.dur_cin.rvs(cin)  # * self.sev_imm[cin]
        self.dur_cin[cin] = dur_cin
        cancer_probs = self.get_cancer_prob(cin)
        self.pars.cancer_prob.set(cancer_probs)
        cancer, no_cancer = self.pars.cancer_prob.split(cin)
        self.ti_cancer[cancer] = self.ti + self.dur_cin[cancer]
        self.ti_clearance[no_cancer] = self.ti + self.dur_cin[no_cancer]

        # Set duration of cancer and time of cancer mortality
        dur_cancer = self.pars.dur_cancer.rvs(cancer)
        self.dur_cancer[cancer] = dur_cancer
        self.ti_cancer_death[cancer] = self.ti + dur_cancer
        return

    def update_immunity(self, uids):
        """
        Update immunity states
        """
        sero_converted = self.pars.sero_prob.filter(uids)
        self.sus_imm[sero_converted] = self.pars.init_imm.rvs(sero_converted)
        self.sev_imm[sero_converted] = self.pars.init_cell_imm.rvs(sero_converted)
        return

    def update_infection(self):
        """
        Update infection states
        """
        ti = self.ti

        # Find men who clear infection
        new_clearance = (
            self.infectious & self.sim.people.male & (self.ti_clearance <= ti)
        ).uids
        if len(new_clearance):
            self.infectious[new_clearance] = False
            self.susceptible[new_clearance] = True
            self.ti_clearance[new_clearance] = ti
            self.update_immunity(new_clearance)

        # Find women who clear infection
        new_clearance = (self.precin & (self.ti_clearance <= ti)).uids
        if len(new_clearance):
            self.precin[new_clearance] = False
            self.infectious[new_clearance] = False
            self.susceptible[new_clearance] = True
            self.ti_clearance[new_clearance] = ti
            self.update_immunity(new_clearance)

        # Find those who progress to CIN
        new_progression = (self.precin & (self.ti_cin <= ti)).uids
        if len(new_progression):
            self.precin[new_progression] = False
            self.cin[new_progression] = True
            self.ti_cin[new_progression] = ti
            self.results.cins[ti] += len(new_progression)

        # Find those who clear CIN
        new_clearance = (self.cin & (self.ti_clearance <= ti)).uids
        if len(new_clearance):
            self.cin[new_clearance] = False
            self.precin[new_clearance] = False
            self.infectious[new_clearance] = False
            self.susceptible[new_clearance] = True
            self.ti_clearance[new_clearance] = ti
            self.update_immunity(new_clearance)

        # Find those who progress to cancer
        new_cancers = (self.cin & (self.ti_cancer <= ti)).uids
        if len(new_cancers):
            self.cin[new_cancers] = False
            self.infectious[new_cancers] = False
            self.cancerous[new_cancers] = True
            self.ti_cancer[new_cancers] = ti
            self.results.cancers[ti] += len(new_cancers)

        # Find those who die of cancer
        new_deaths = (self.cancerous & (self.ti_cancer_death <= ti)).uids
        if len(new_deaths):
            self.cancerous[new_deaths] = False
            self.ti_cancer_death[new_deaths] = ti
            self.sim.people.request_death(new_deaths)
            self.results.cancer_deaths[ti] += len(new_deaths)

        return

    def get_cancer_prob(self, uids):
        """
        Get the probability of progressing to cancer

        """
        dur_cin = self.dur_cin[uids]
        sev = hpv.compute_severity_integral(
            dur_cin, rel_sev=self.rel_sev[uids], pars=self.pars.cin_fn
        )
        cancer_probs = 1 - np.power(1 - self.pars.transform_prob, sev**2)
        return cancer_probs

    def get_cin_prob(self, uids):
        """
        Get the probability of progressing to CIN
        """
        return hpv.logf2(self.dur_precin[uids] * self.rel_sev[uids], **self.pars.cin_fn)

    def update_results(self):
        super().update_results()
        ti = self.ti
        women = (self.sim.people.age >= 15) & self.sim.people.female
        ages = [15, 25, 35, 45, 55]

        def cond_prob(num, denom):
            n_num = np.count_nonzero(num & denom)
            n_denom = np.count_nonzero(denom)
            return sc.safedivide(n_num, n_denom)

        self.results["prevalence"][ti] = cond_prob(
            (self.infectious & self.sim.people.female), women
        )

        # Calculate cancer incidence
        scale_factor = 1e5
        new_cancers = self.results.cancers[ti]
        sus_pop = (
            (self.sim.people.age >= 15) & (self.sim.people.female) & (~self.cancerous)
        )
        denominator = np.count_nonzero(sus_pop) / scale_factor
        self.results["cancer_incidence"][ti] = sc.safedivide(new_cancers, denominator)
        return


class HPV16(HPV):
    def __init__(self, pars=None, **kwargs):
        super().__init__(genotype="16")
        self.define_pars(
            rel_beta=1.0,
            dur_precin=ss.lognorm_ex(ss.dur(3, "year"), ss.dur(9, "year")),
            dur_cin=ss.lognorm_ex(ss.dur(5, "year"), ss.dur(20, "year")),
            cin_fn=dict(k=0.3, x_infl=0, y_max=0.5, ttc=50),
            transform_prob=2e-3,
        )
        self.update_pars(pars, **kwargs)
        return


class HPV18(HPV):
    def __init__(self, pars=None, **kwargs):
        super().__init__(genotype="18")
        self.define_pars(
            beta=0.75,
            dur_precin=ss.lognorm_ex(ss.dur(2.5, "year"), ss.dur(9, "year")),
            dur_cin=ss.lognorm_ex(ss.dur(5, "year"), ss.dur(20, "year")),
            cin_fn=dict(k=0.35, x_infl=0, y_max=0.5, ttc=50),
            transform_prob=2e-3,
        )
        self.update_pars(pars, **kwargs)
        return



