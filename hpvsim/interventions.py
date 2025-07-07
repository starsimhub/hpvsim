import starsim as ss
import sciris as sc
import numpy as np


__all__ = ["screen", "treat"]  # , "vaccinate"]


# %% Template classes for routine and campaign delivery

__all__ += ['RoutineDelivery', 'CampaignDelivery']

class RoutineDelivery(ss.Intervention):
    """
    Base class for any intervention that uses routine delivery; handles interpolation of input years.
    """

    def __init__(self, *args, years=None, start_year=None, end_year=None, prob=None, annual_prob=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.years = years
        self.start_year = start_year
        self.end_year = end_year
        self.prob = sc.promotetoarray(prob)
        self.annual_prob = annual_prob  # Determines whether the probability is annual or per timestep
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder - initialize delivery
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Validate inputs
        if (self.years is not None) and (self.start_year is not None or self.end_year is not None):
            errormsg = 'Provide either a list of years or a start year, not both.'
            raise ValueError(errormsg)

        # If start_year and end_year are not provided, figure them out from the provided years or the sim
        if self.years is None:
            if self.start_year is None: self.start_year = sim.pars.start
            if self.end_year is None:   self.end_year = sim.pars.stop
        else:
            self.years = sc.promotetoarray(self.years)
            self.start_year = self.years[0]
            self.end_year = self.years[-1]

        # More validation
        yearvec = sim.t.yearvec
        if not(any(np.isclose(self.start_year, yearvec)) and any(np.isclose(self.end_year, yearvec))):
            errormsg = 'Years must be within simulation start and end dates.'
            raise ValueError(errormsg)

        # Adjustment to get the right end point
        dt = sim.pars.dt # TODO: need to eventually replace with own timestep, but not initialized yet since super().init_pre() hasn't been called
        adj_factor = int(1/dt) - 1 if dt < 1 else 1

        # Determine the timepoints at which the intervention will be applied
        self.start_point = sc.findfirst(yearvec, self.start_year)
        self.end_point   = sc.findfirst(yearvec, self.end_year) + adj_factor
        self.years       = sc.inclusiverange(self.start_year, self.end_year)
        self.timepoints  = sc.inclusiverange(self.start_point, self.end_point)
        self.yearvec     = np.arange(self.start_year, self.end_year + adj_factor, dt) # TODO: integrate with self.t

        # Get the probability input into a format compatible with timepoints
        if len(self.years) != len(self.prob):
            if len(self.prob) == 1:
                self.prob = np.array([self.prob[0]] * len(self.timepoints))
            else:
                errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
                raise ValueError(errormsg)
        else:
            self.prob = sc.smoothinterp(self.yearvec, self.years, self.prob, smoothness=0)

        # Lastly, adjust the probability by the sim's timestep, if it's an annual probability
        if self.annual_prob: self.prob = 1 - (1 - self.prob) ** dt

        return


class CampaignDelivery(ss.Intervention):
    """
    Base class for any intervention that uses campaign delivery; handles interpolation of input years.
    """

    def __init__(self, *args, years=None, interpolate=None, prob=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.years = sc.promotetoarray(years)
        self.interpolate = True if interpolate is None else interpolate
        self.prob = sc.promotetoarray(prob)
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Decide whether to apply the intervention at every timepoint throughout the year, or just once.
        self.timepoints = sc.findnearest(sim.timevec, self.years)

        if len(self.prob) == 1:
            self.prob = np.array([self.prob[0]] * len(self.timepoints))

        if len(self.prob) != len(self.years):
            errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
            raise ValueError(errormsg)

        return


# %% Screening and triage

__all__ += ['BaseTest', 'BaseScreening', 'routine_screening', 'campaign_screening', 'BaseTriage', 'routine_triage',
            'campaign_triage']

class BaseTest(ss.Intervention):
    """
    Base class for screening and triage.

    Args:
         product        (Product)       : the diagnostic to use
         prob           (float/arr)     : annual probability of eligible people receiving the diagnostic
         eligibility    (inds/callable) : indices OR callable that returns inds
         kwargs         (dict)          : passed to Intervention()
    """

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.screened = ss.BoolArr('screened')
        self.screens = ss.FloatArr('screens', default=0)
        self.ti_screened = ss.FloatArr('ti_screened')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        return

    def deliver(self):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        sim = self.sim
        ti = sc.findinds(self.timepoints, sim.ti)[0]
        prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
        eligible_uids = self.check_eligibility()  # Check eligibility
        self.coverage_dist.set(p=prob)
        accept_uids = self.coverage_dist.filter(eligible_uids)
        if len(accept_uids):
            self.outcomes = self.product.administer(accept_uids)  # Actually administer the diagnostic
        return accept_uids

    def check_eligibility(self):
        raise NotImplementedError


class BaseScreening(BaseTest):
    """
    Base class for screening.

    Args:
        kwargs (dict): passed to BaseTest
    """
    def check_eligibility(self):
        """
        Check eligibility
        """
        raise NotImplementedError

    def step(self):
        """
        Perform screening by finding who's eligible, finding who accepts, and applying the product.
        """
        sim = self.sim
        accept_uids = ss.uids()
        if sim.ti in self.timepoints: # TODO: change to self.ti
            accept_uids = self.deliver()
            self.screened[accept_uids] = True
            self.screens[accept_uids] += 1
            self.ti_screened[accept_uids] = sim.ti
            self.results['n_screened'][sim.ti] = len(accept_uids)
            self.results['n_dx'][sim.ti] = len(self.outcomes['positive'])

        return accept_uids


class BaseTriage(BaseTest):
    """
    Base class for triage.

    Args:
        kwargs (dict): passed to BaseTest
    """
    def check_eligibility(self):
        return sc.promotetoarray(self.eligibility(self.sim))

    def step(self):
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        accept_inds = ss.uids()
        if self.sim.t in self.timepoints: accept_inds = self.deliver() # TODO: not robust for timestep
        return accept_inds


class routine_screening(BaseScreening, RoutineDelivery):
    """
    Routine screening - an instance of base screening combined with routine delivery.
    See base classes for a description of input arguments.

    **Examples**::

        screen1 = ss.routine_screening(product=my_prod, prob=0.02) # Screen 2% of the eligible population every year
        screen2 = ss.routine_screening(product=my_prod, prob=0.02, start_year=2020) # Screen 2% every year starting in 2020
        screen3 = ss.routine_screening(product=my_prod, prob=np.linspace(0.005,0.025,5), years=np.arange(2020,2025)) # Scale up screening over 5 years starting in 2020
    """
    pass


class campaign_screening(BaseScreening, CampaignDelivery):
    """
    Campaign screening - an instance of base screening combined with campaign delivery.
    See base classes for a description of input arguments.

    **Examples**::

        screen1 = ss.campaign_screening(product=my_prod, prob=0.2, years=2030) # Screen 20% of the eligible population in 2020
        screen2 = ss.campaign_screening(product=my_prod, prob=0.02, years=[2025,2030]) # Screen 20% of the eligible population in 2025 and again in 2030
    """
    pass


class routine_triage(BaseTriage, RoutineDelivery):
    """
    Routine triage - an instance of base triage combined with routine delivery.
    See base classes for a description of input arguments.

    **Example**:
        # Example: Triage positive screens into confirmatory testing
        screened_pos = lambda sim: sim.interventions.screening.outcomes['positive']
        triage = ss.routine_triage(product=my_triage, eligibility=screen_pos, prob=0.9, start_year=2030)
    """
    pass


class campaign_triage(BaseTriage, CampaignDelivery):
    """
    Campaign triage - an instance of base triage combined with campaign delivery.
    See base classes for a description of input arguments.

    **Examples**:
        # Example: In 2030, triage all positive screens into confirmatory testing
        screened_pos = lambda sim: sim.interventions.screening.outcomes['positive']
        triage1 = ss.campaign_triage(product=my_triage, eligibility=screen_pos, prob=0.9, years=2030)
    """
    pass


#%% Treatment interventions

__all__ += ['BaseTreatment', 'treat_num']

class BaseTreatment(ss.Intervention):
    """
    Base treatment class.

    Args:
         product        (str/Product)   : the treatment product to use
         prob           (float/arr)     : probability of treatment aong those eligible
         eligibility    (inds/callable) : indices OR callable that returns inds
         kwargs         (dict)          : passed to Intervention()
    """
    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.outcomes = {k: np.array([], dtype=int) for k in ['unsuccessful', 'successful']} # Store outcomes on each timestep
        return

    def get_accept_inds(self):
        """
        Get indices of people who will acccept treatment; these people are then added to a queue or scheduled for receiving treatment
        """
        accept_uids = ss.uids()
        eligible_uids = self.check_eligibility()  # Apply eligiblity
        if len(eligible_uids):
            self.coverage_dist.set(p=self.prob[0])
            accept_uids = self.coverage_dist.filter(eligible_uids)
        return accept_uids

    def get_candidates(self):
        """
        Get candidates for treatment on this timestep. Implemented by derived classes.
        """
        raise NotImplementedError

    def step(self):
        """
        Perform treatment by getting candidates, checking their eligibility, and then treating them.
        """
        # Get indices of who will get treated
        treat_candidates = self.get_candidates()  # NB, this needs to be implemented by derived classes
        still_eligible = self.check_eligibility()
        treat_uids = treat_candidates.intersect(still_eligible)
        if len(treat_uids):
            self.outcomes = self.product.administer(treat_uids)
        return treat_uids


class treat_num(BaseTreatment):
    """
    Treat a fixed number of people each timestep.

    Args:
         max_capacity (int): maximum number who can be treated each timestep
    """
    def __init__(self, max_capacity=None, **kwargs):
        super().__init__(**kwargs)
        self.queue = []
        self.max_capacity = max_capacity
        return

    def add_to_queue(self):
        """
        Add people who are willing to accept treatment to the queue
        """
        accept_inds = self.get_accept_inds()
        if len(accept_inds): self.queue += accept_inds.tolist()
        return

    def get_candidates(self):
        """
        Get the indices of people who are candidates for treatment
        """
        treat_candidates = np.array([], dtype=int)
        if len(self.queue):
            if self.max_capacity is None or (self.max_capacity > len(self.queue)):
                treat_candidates = self.queue[:]
            else:
                treat_candidates = self.queue[:self.max_capacity]
        return ss.uids(treat_candidates) # TODO: Check

    def step(self):
        """
        Apply treatment. On each timestep, this method will add eligible people who are willing to accept treatment to a
        queue, and then will treat as many people in the queue as there is capacity for.
        """
        self.add_to_queue()
        treat_inds = BaseTreatment.step(self) # Apply method from BaseTreatment class
        self.queue = [e for e in self.queue if e not in treat_inds] # Recreate the queue, removing people who were treated
        return treat_inds


#%% Vaccination

__all__ += ['BaseVaccination', 'routine_vx', 'campaign_vx']

class BaseVaccination(ss.Intervention):
    """
    Base vaccination class for determining who will receive a vaccine.

    Args:
         product        (str/Product)   : the vaccine to use
         prob           (float/arr)     : annual probability of eligible population getting vaccinated
         label          (str)           : the name of vaccination strategy
         kwargs         (dict)          : passed to Intervention()
    """
    def __init__(self, *args, product=None, prob=None, label=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.label = label
        self._parse_product(product)
        self.vaccinated = ss.BoolArr('vaccinated')
        self.n_doses = ss.FloatArr('doses', default=0)
        self.ti_vaccinated = ss.FloatArr('ti_vaccinated')
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        return

    def step(self):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        sim = self.sim
        accept_uids = np.array([])
        if sim.ti in self.timepoints:

            ti = sc.findinds(self.timepoints, sim.ti)[0]
            prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
            is_eligible = self.check_eligibility()  # Check eligibility
            self.coverage_dist.set(p=prob)
            accept_uids = self.coverage_dist.filter(is_eligible)

            if len(accept_uids):
                self.product.administer(sim.people, accept_uids)

                # Update people's state and dates
                self.vaccinated[accept_uids] = True
                self.ti_vaccinated[accept_uids] = sim.ti
                self.n_doses[accept_uids] += 1

        return accept_uids


class routine_vx(BaseVaccination, RoutineDelivery):
    """
    Routine vaccination - an instance of base vaccination combined with routine delivery.
    See base classes for a description of input arguments.
    """
    pass


class campaign_vx(BaseVaccination, CampaignDelivery):
    """
    Campaign vaccination - an instance of base vaccination combined with campaign delivery.
    See base classes for a description of input arguments.
    """
    pass


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
