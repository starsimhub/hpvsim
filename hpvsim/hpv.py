"""
Module for HPV
"""

import numpy as np
import starsim as ss
import sciris as sc
import stisim as sti
import hpvsim as hpv

__all__ = ["HPV"]


class HPV(sti.BaseSTI):

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
            dur_precin=None,
            dur_cin=None,
            cin_prob=ss.bernoulli(p=0.5),  # placeholder, gets reset
            cancer_prob=ss.bernoulli(p=0.5),  # placeholder, gets reset
            cin_fn=None,
            transform_prob=None,
        )

        genotype_pars = self.get_genotype_pars(genotype)
        pars = sc.mergedicts(genotype_pars, pars)
        self.update_pars(pars, **kwargs)
        self.define_states(
            # States
            ss.State("latent", default=False, label="latent"),
            ss.State("precin", default=False, label="precin"),
            ss.State("cin", default=False, label="cin"),
            ss.State("cancerous", default=False, label="cancerous"),
            # Duration and timestep of states
            ss.FloatArr("dur_precin", default=0, label="duration of precin"),
            ss.FloatArr("dur_cin", default=0, label="duration of cin"),
            ss.FloatArr("dur_cancer", default=0, label="duration of cancer"),
            ss.FloatArr("ti_cancer", default=np.nan, label="time of cancer"),
            ss.FloatArr(
                "ti_cancer_death", default=np.nan, label="time of cancer mortality"
            ),
            ss.FloatArr("ti_cin", default=np.nan, label="time of cin"),
            ss.FloatArr("ti_clearance", default=np.nan, label="time of clearance"),
            # Immunity states
            ss.FloatArr("rel_sev", default=1, label="relative severity"),
            ss.FloatArr("sus_imm", default=0, label="immunity to infection"),
            ss.FloatArr("sev_imm", default=0, label="immunity to severe disease"),
        )

    def get_genotype_pars(self, genotype):
        """
        Get genotype-specific parameters
        """
        if genotype == "16":
            pars = sc.objdict(
                rel_beta=1.0,
                dur_precin=ss.lognorm_ex(ss.dur(3, "year"), ss.dur(9, "year")),
                dur_cin=ss.lognorm_ex(ss.dur(5, "year"), ss.dur(20, "year")),
                cin_fn=dict(k=0.3, x_infl=0, y_max=0.5, ttc=50),
                transform_prob=2e-3,
            )
        elif genotype == "18":
            pars = sc.objdict(
                beta=0.75,
                dur_precin=ss.lognorm_ex(ss.dur(2.5, "year"), ss.dur(9, "year")),
                dur_cin=ss.lognorm_ex(ss.dur(5, "year"), ss.dur(20, "year")),
                cin_fn=dict(k=0.35, x_infl=0, y_max=0.5, ttc=50),
                transform_prob=2e-3,
            )
        else:
            raise ValueError(f"Genotype {genotype} not recognized")
        return pars

    def init_results(self):
        super().init_results()
        results = [
            ss.Result("cins", label="CINs"),
            ss.Result("cancers", label="Cancers"),
            ss.Result("cancer_incidence", label="Cancer incidence", scale=False),
            ss.Result("cancer_deaths", label="Cancer deaths"),
            ss.Result("prevalence_15_24", label="Prevalence 15-24", scale=False),
            ss.Result("prevalence_25_34", label="Prevalence 25-34", scale=False),
            ss.Result("prevalence_35_44", label="Prevalence 35-44", scale=False),
            ss.Result("prevalence_45_54", label="Prevalence 45-54", scale=False),
            ss.Result("prevalence_55_64", label="Prevalence 55-64", scale=False),
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
        for age in ages:
            age_group = (
                (self.sim.people.female)
                & (self.sim.people.age >= age)
                & (self.sim.people.age < age + 10)
            )
            infectious_age = self.infectious & age_group
            self.results[f"prevalence_{age}_{age+9}"][ti] = cond_prob(
                infectious_age, age_group
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
