import starsim as ss
import sciris as sc
import numpy as np


__all__ = ["screen", "treat"]  # , "vaccinate"]


class screen(ss.Intervention):
    """
    Screning intervention to detect pre-cancerous lesions
    """

    def __init__(
        self,
        *args,
        start_year=None,
        pars=None,
        eligibility=None,
        modules=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.modules = sc.promotetolist(modules)
        self.screen_results = None

        self.define_pars(
            unit="month",
            # Screening effects
            start_year=start_year,  # Day to start screening
            stop_year=None,  # Day to stop screening
            p_seek_care=ss.bernoulli(p=0.9),  # Distribution of care-seeking behavior
            screen_age=[30, 50],  # Age range for screening
            screen_interval=ss.dur(5, "years"),  # Interval between screenings
            screen_sensitivity=dict(
                # susceptible=ss.bernoulli(p=0.05),
                precin=ss.bernoulli(p=0.45),
                cin=ss.bernoulli(p=0.95),
                cancerous=ss.bernoulli(p=0.99),
            ),
        ),
        self.update_pars(pars, **kwargs)
        self.eligibility = eligibility

        self.define_states(
            ss.FloatArr("ti_screened"),
        )

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.pars.stop_year is None:
            self.pars.stop_year = sim.t.npts - 1
        else:
            ti = sc.findfirst(sim.t.yearvec, self.pars.stop_year)
            self.pars.stop_year = ti

        if self.pars.start_year:
            ti = sc.findfirst(sim.t.yearvec, self.pars.start_year)
            self.pars.start_year = ti

        return

    def init_results(self):
        super().init_results()
        results = [
            ss.Result("new_screens", dtype=int, label="New screens administered"),
            ss.Result("new_screened", dtype=int, label="New people screened"),
        ]
        self.define_results(*results)
        return

    def check_eligibility(self):
        adult_females = self.sim.people.female
        in_age_range = (self.sim.people.age >= self.pars.screen_age[0]) & (
            self.sim.people.age <= self.pars.screen_age[1]
        )
        conditions = adult_females & in_age_range
        unscreened = np.isnan(self.ti_screened).astype(bool)

        screened_in_interval_range = (
            self.sim.ti - self.ti_screened
        ) >= self.pars.screen_interval.values
        screen_eligible = unscreened | screened_in_interval_range
        conditions = conditions & screen_eligible
        if self.eligibility is not None:
            other_eligible = sc.promotetoarray(self.eligibility(self.sim))
            conditions = other_eligible & conditions
        return ss.uids(conditions)

    def step(self):
        modules = self.modules
        if self.pars.stop_year >= self.ti >= self.pars.start_year:
            # Identify eligible agents for treatment
            eligible_inds = self.check_eligibility()
            seeks_care = self.pars.p_seek_care.filter(eligible_inds)
            if len(seeks_care):
                self.ti_screened[seeks_care] = self.ti
                self.results["new_screens"][self.ti] += len(seeks_care)
                self.results["new_screened"][self.ti] += len(seeks_care)
                test_pos_inds = []
                for state, sens in self.pars.screen_sensitivity.items():
                    for module in modules:
                        screen_pos = sens.filter(
                            seeks_care.intersect(getattr(module, state).uids)
                        )
                        test_pos_inds += screen_pos.tolist()
                test_pos_inds = list(set(test_pos_inds))
                self.screen_results = dict(
                    positive=ss.uids(test_pos_inds),
                    negative=np.setdiff1d(seeks_care, test_pos_inds),
                )

        return


class treat(ss.Intervention):
    """
    Treatment intervention to treat detected pre-cancerous lesions
    """

    def __init__(
        self,
        *args,
        start_year=None,
        pars=None,
        eligibility=None,
        modules=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.modules = sc.promotetolist(modules)

        self.define_pars(
            unit="month",
            # Treatment effects
            start_year=start_year,  # Day to start treatment
            stop_year=None,  # Day to stop treatment
            p_seek_care=ss.bernoulli(p=0.9),  # Distribution of care-seeking behavior
            treat_efficacy=dict(
                susceptible=ss.bernoulli(p=0),
                precin=ss.bernoulli(p=0),
                cin=ss.bernoulli(p=0.81),
                cancerous=ss.bernoulli(p=0),
            ),
        ),
        self.update_pars(pars, **kwargs)
        self.eligibility = eligibility

        self.define_states(
            ss.FloatArr("ti_treated"),
        )

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.pars.stop_year is None:
            self.pars.stop_year = sim.t.npts - 1
        else:
            ti = sc.findfirst(sim.t.yearvec, self.pars.stop_year)
            self.pars.stop_year = ti

        if self.pars.start_year:
            ti = sc.findfirst(sim.t.yearvec, self.pars.start_year)
            self.pars.start_year = ti

        return

    def init_results(self):
        super().init_results()
        results = [
            ss.Result("new_treatments", dtype=int, label="New treatments administered"),
        ]
        self.define_results(*results)
        return

    def check_eligibility(self):
        if self.eligibility is not None:
            conditions = sc.promotetoarray(self.eligibility(self.sim))
            return ss.uids(conditions)
        else:
            return ss.uids([])

    def step(self):
        modules = self.modules
        if self.pars.stop_year >= self.ti >= self.pars.start_year:
            # Identify eligible agents for treatment
            eligible_inds = self.check_eligibility()
            seeks_care = self.pars.p_seek_care.filter(eligible_inds)
            if len(seeks_care):
                self.ti_treated[seeks_care] = self.ti
                self.results["new_treatments"][self.ti] += len(seeks_care)
                treated_inds = []
                for state, efficacy in self.pars.treat_efficacy.items():
                    for module in modules:
                        treated_effectively = efficacy.filter(
                            seeks_care.intersect(getattr(module, state).uids)
                        )
                        if len(treated_effectively):
                            module.ti_clearance[treated_effectively] = self.ti
                        treated_inds += treated_effectively.tolist()
                treated_inds = list(set(treated_inds))

        return
