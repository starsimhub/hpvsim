"""HPV-specific interventions.

Screening, triage, treatment, and vaccination are Starsim's deliveries of
HPVsim's products (see ``hpvsim.products``)::

    women = ss.ppl.female & (ss.ppl.age >= 30) & (ss.ppl.age < 50)
    screening = ss.screen(hpv.dx('via'), where=women, coverage=0.6, interval=ss.years(1), start=2025)
    ablation = ss.treat(hpv.tx('ablation'), who=screening.positive & hpv.no_cancer, coverage=0.9)
    vaccination = ss.vaccinate(hpv.vx('bivalent'), where=ss.ppl.female, at_age=9, coverage=0.9, start=2025)

This module has the conditions on cancer that treatments use, and ``dynamic_pars``.
"""
import numpy as np
import starsim as ss

from hpvsim.utils import any_genotype_cancer

__all__ = ['has_cancer', 'no_cancer', 'dynamic_pars']


def has_cancer(sim):
    """Whether each person has cancer of any genotype: for a delivery's ``where=`` or ``who=``,
    e.g. ``ss.treat(hpv.radiation(), who=screening.positive & hpv.has_cancer)``"""
    return any_genotype_cancer(sim)


def no_cancer(sim):
    """Whether each person has no cancer of any genotype: for a delivery's ``where=`` or ``who=``,
    e.g. ``ss.treat(hpv.tx('ablation'), who=screening.positive & hpv.no_cancer)``"""
    return ~any_genotype_cancer(sim)


class dynamic_pars(ss.Intervention):
    """Time-varying parameter editor.

    pars: dict mapping dotted-path strings to {'years': [...], 'vals': [...]}
    schedules. Each step, the resolved parameter is set to the interpolated
    (default) or stepwise (interpolate=False) value for the current year.

    Dotted-path resolution order: sim.diseases > sim.interventions > sim.pars.
    Schedule keys are epoch years (e.g. 2020, 2030) — `np.interp` clamps to
    the first / last value outside the schedule.
    """

    def __init__(self, pars=None, interpolate=True, **kwargs):
        super().__init__(**kwargs)
        self.par_schedules = pars or {}
        self.interpolate = interpolate

    def step(self):
        year = self.sim.t.now('year')
        for dotted_path, schedule in self.par_schedules.items():
            years = np.asarray(schedule['years'], dtype=float)
            vals = np.asarray(schedule['vals'], dtype=float)
            if self.interpolate:
                val = float(np.interp(year, years, vals))
            else:
                idx = int(np.searchsorted(years, year, side='right')) - 1
                if idx < 0:
                    continue
                val = float(vals[idx])
            self._set_dotted(self.sim, dotted_path, val)

    @staticmethod
    def _set_dotted(sim, dotted_path, value):
        """Resolve a dotted-path string against (sim.diseases, sim.interventions, sim.pars) and set it.

        Top-level segment is looked up in:
          1. sim.diseases (by key)         e.g. 'hpv16.beta' -> sim.diseases['hpv16'].pars.beta
          2. sim.interventions (by name)   e.g. 'screen.prob' -> sim.interventions['screen'].prob
          3. sim.pars (by key)             e.g. 'rand_seed' -> sim.pars.rand_seed
        Diseases and interventions require at least one tail segment (the attribute
        to set on the resolved module). sim.pars accepts a single segment (set
        directly on sim.pars). Raises KeyError if the head doesn't resolve anywhere.
        """
        parts = dotted_path.split('.')
        head, tail = parts[0], parts[1:]

        if head in sim.diseases:
            if not tail:
                raise KeyError(
                    f'Path {dotted_path!r}: missing attribute name after disease '
                    f'key {head!r}.'
                )
            # Navigate into the disease module's .pars by default.
            target = sim.diseases[head].pars
            for seg in tail[:-1]:
                target = getattr(target, seg)
            setattr(target, tail[-1], value)
            return

        if head in sim.interventions:
            if not tail:
                raise KeyError(
                    f'Path {dotted_path!r}: missing attribute name after '
                    f'intervention name {head!r}.'
                )
            target = sim.interventions[head]
            for seg in tail[:-1]:
                target = getattr(target, seg)
            setattr(target, tail[-1], value)
            return

        # Fall back to sim.pars.
        if not hasattr(sim.pars, head):
            raise KeyError(
                f'Cannot resolve dotted path {dotted_path!r}: head segment '
                f'{head!r} is not a sim.diseases / sim.interventions / sim.pars key.'
            )
        if not tail:
            setattr(sim.pars, head, value)
            return
        target = getattr(sim.pars, head)
        for seg in tail[:-1]:
            target = getattr(target, seg)
        setattr(target, tail[-1], value)
