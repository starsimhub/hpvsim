"""
HPVsim simulation

``hpv.Sim(location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'])`` builds the default model: one
declared HPV disease per genotype, the sexual network, births, deaths and migration for the
location, cross-genotype immunity, and results over all genotypes (``sim.results.all_hpv``).

``connectors=`` and ``analyzers=`` add to the defaults; ``diseases=`` (HPV genotypes, or other
diseases such as HIV), ``networks=``, ``demographics=`` and ``people=`` replace them.
"""

import warnings
import numpy as np
import sciris as sc
import starsim as ss

from .cross_genotype import HPVTotal, CrossImmunity
from .data.country import load_country
from .demographics import AgeMigration, AnnualBirths, Births
from .hiv import HIV, HIV_incidence, ART
from .hpv import HPV, _normalize_genotype
from .network import SexualNetwork
from .parameters import SimPars, get_genotype_pars
from .products import products_in
from .seeding import init_prevs


__all__ = ['Sim']


class Sim(ss.Sim):
    """
    HPVsim simulation

    Args:
        location (str): a country, for its population, births, deaths and migration; None (default) for uniform ages 0-60 and no demography
        genotypes (list): the genotypes, e.g. [16, 18, 'hi5', 'ohr'] (default [16])
        genotype_pars (dict): parameters by genotype, e.g. {'hpv16': dict(cin_fn=dict(k=0.4))}
        init_seeding (str): 'exclusive' (default; the total prevalence curve shared between genotypes) or 'independent' (each genotype's own curve)
        init_hpv_dist (dict): with 'exclusive', the share of each genotype (default equal)
        n_agents (int): the number of agents
        start, stop (float): the first and last years
        dt (float): the step, in years
        total_pop (float): the population the agents represent (default: the location's)
        model_hiv (bool/str): add HIV: True or 'transmission' (network-transmitted `hpv.HIV`), or 'incidence' (from hiv_data)
        hiv_data (dict/str): for model_hiv='incidence': {'incidence', 'art_coverage', 'init_prev'}, or a folder (see `hpv.data.load_hiv_data`)
        hiv_pars (dict): parameters of the HIV disease
        nw_pars (dict): parameters of the sexual network
        imm_pars (dict): parameters of the cross-genotype immunity (see `hpv.CrossImmunity`)
        v2_compat_demographics (bool): annual birth pulses and whole-year ages, as in HPVsim 2
        pars (dict): parameters of the sim or of the modules, by name
        kwargs (dict): as pars, or the modules (`diseases=`, `interventions=`, ...)

    Examples:
        ```python
        sim = hpv.Sim(location='nigeria', genotypes=[16, 18, 'hi5', 'ohr'], start=1980, stop=2030)
        sim.run()
        sim.results.all_hpv.asr_cancer_incidence
        ```
    """

    def __init__(self, location=None, genotypes=None, genotype_pars=None, init_seeding='exclusive', init_hpv_dist=None,
                 n_agents=10_000, start=1990, stop=2060, dt=0.25, total_pop=None, ms_agent_ratio=1, pars=None,
                 v2_compat_demographics=False, end=None, datafolder=None, model_hiv=None, hiv_data=None, hiv_pars=None,
                 nw_pars=None, imm_pars=None, **kwargs):
        if end is not None:
            ss.warn('hpv.Sim: `end` is a deprecated alias for `stop`; use `stop=` instead.')
            stop = end
        if ms_agent_ratio != 1:
            raise NotImplementedError('ms_agent_ratio > 1 (multiscale cancer agents) is not ported to Starsim 4')
        pars = sc.mergedicts(pars, kwargs)
        location = sc.ifelse(location, pars.pop('location', None))
        modules = {key: sc.tolist(pars.pop(key, None)) for key in ['diseases', 'connectors', 'analyzers', 'interventions', 'networks', 'demographics']}
        people = pars.pop('people', None)

        # Population
        country = None
        if location is None:
            ss.warn('hpv.Sim: no location supplied; using uniform ages 0-60, no births/deaths/migration, and pop_scale=1. '
                    'Pass location= or demographics= to model a real population, or call hpv.demo() for a canonical Nigeria sim.')
        else:
            country = load_country(location, year=int(start), datafolder=datafolder)
            if total_pop is None:
                total_pop = country['age_data']['value'].sum()
        age_data = None if country is None else country['age_data']
        if people is None:
            people = ss.People(n_agents, age_data=age_data if age_data is not None else ss.uniform(0, 60))

        # Parameters: of the sim, or of the modules, routed by name
        sim_keys = set(SimPars().keys()) - {'location'}
        sim_pars = {k: pars.pop(k) for k in list(pars) if k in sim_keys}
        hpv_keys = set(get_genotype_pars('hpv16').keys()) | {'cin_fn', 'cancer_fn'}
        hpv_pars = {k: pars.pop(k) for k in list(pars) if k in hpv_keys}
        imm_pars = sc.mergedicts(imm_pars, {k: pars.pop(k) for k in list(pars) if k in CrossImmunity.defaults})
        net_pars = sc.mergedicts(None if country is None else country['network_pars'], nw_pars, {k: pars.pop(k) for k in list(pars) if k in SexualNetwork().pars})
        scoped = {k: pars.pop(k) for k in list(pars) if k in {'hpv16', 'hpv18', 'hi5', 'ohr', 'hiv'}}
        if pars:
            raise ValueError(f'hpv.Sim: unrecognized parameter(s) {sorted(pars)}')

        # Genotypes
        diseases = modules['diseases']
        hpvs = [d for d in diseases if isinstance(d, HPV)]
        others = [d for d in diseases if not isinstance(d, HPV)]
        if hpvs and genotypes is not None:
            raise ValueError('Specify HPV via genotypes= or HPV instances in diseases=, not both.')
        if not hpvs:
            keys = [_normalize_genotype(g) for g in (genotypes if genotypes is not None else [16])]
            if init_hpv_dist is not None and set(init_hpv_dist) != set(keys):
                raise ValueError(f'init_hpv_dist keys {sorted(init_hpv_dist)} do not match the genotypes {sorted(keys)}')
            prevs = init_prevs(keys, init_seeding=init_seeding, init_hpv_dist=init_hpv_dist, age_data=age_data)
            gpars = sc.mergedicts(genotype_pars)
            hpvs = [HPV(k, genotypes=keys, pars=sc.mergedicts(dict(init_prev=prevs[k]), hpv_pars, gpars.get(k), scoped.get(k))) for k in keys]
        elif hpv_pars or genotype_pars:
            raise ValueError('hpv.Sim: give genotype parameters to your HPV instances directly')
        keys = [h.genotype for h in hpvs]

        # HIV
        hiv = next((d for d in others if isinstance(d, HIV)), None)
        auto_interventions = []
        if model_hiv:
            if hiv is not None:
                raise ValueError('model_hiv= is mutually exclusive with supplying your own HIV disease in diseases=')
            mode = 'transmission' if model_hiv is True else model_hiv
            hiv_pars = sc.mergedicts(hiv_pars, scoped.get('hiv'))
            if mode == 'transmission':
                hiv = HIV(pars=hiv_pars)
            elif mode == 'incidence':
                data = hiv_data
                if data is not None and not isinstance(data, dict):
                    from .data.hiv import load_hiv_data
                    data = load_hiv_data(data)
                missing = [k for k in ('incidence', 'art_coverage') if not data or k not in data]
                if missing:
                    raise ValueError(f"model_hiv='incidence' requires hiv_data with {missing}")
                if data.get('init_prev') is not None:
                    hiv_pars = sc.mergedicts(dict(init_prev=data['init_prev']), hiv_pars)
                hiv = HIV_incidence(incidence=data['incidence'], pars=hiv_pars)
                if not any(getattr(iv, 'name', None) == 'art' for iv in modules['interventions']):
                    from .data.hiv import reshape_art_coverage
                    auto_interventions.append(ART(hiv, coverage=_art_by_year(reshape_art_coverage(data['art_coverage']))))
            else:
                raise ValueError(f"model_hiv must be True, 'transmission', or 'incidence'; got {model_hiv!r}")
            others.append(hiv)

        # Connectors: immunity across genotypes, each genotype's own modifiers, HIV's effects
        user_cross = [c for c in modules['connectors'] if isinstance(c, CrossImmunity)]
        connectors = [c for c in modules['connectors'] if not isinstance(c, CrossImmunity)]
        if user_cross and imm_pars:
            raise ValueError('imm_pars= is mutually exclusive with supplying your own CrossImmunity in connectors=')
        cross = user_cross[0] if user_cross else CrossImmunity(pars=imm_pars)
        mods = cross.modifiers(hpvs) + [m for h in hpvs for m in h.modifiers()]
        if hiv is not None:
            mods += hiv.effects(hpvs)
        connectors = mods + connectors

        # Network, demographics, analyzers, interventions
        networks = modules['networks'] or [SexualNetwork(**net_pars)]
        if modules['networks'] and nw_pars:
            raise ValueError('nw_pars= is mutually exclusive with supplying your own networks=')
        demographics = modules['demographics']
        if not demographics and country is not None:
            death_rate = country['death_rate']
            death_rate = death_rate[(death_rate['Time'] >= int(start) - 1) & (death_rate['Time'] <= int(stop) + 1)]
            births = AnnualBirths if v2_compat_demographics else Births
            demographics = [births(birth_rate=country['birth_rate']), ss.Deaths(death_rate=death_rate), AgeMigration(v2_compat=v2_compat_demographics)]
        if demographics:
            demographics = list(demographics) + [_NewAgentImmunity()]
        analyzers = [HPVTotal()] + modules['analyzers']
        interventions = auto_interventions + modules['interventions']
        for product in products_in(interventions): # Products act on the sim's genotypes, unless they were given others
            product.set_genotypes(keys)
        for iv in interventions: # Starsim 3's dynamic parameters as modifiers
            if isinstance(iv, ss.modify):
                interventions.remove(iv)
                connectors.append(iv)

        self.location = None if location is None else location.lower()
        self._v2_compat_demographics = v2_compat_demographics
        super().__init__(
            start=start, stop=stop, dt=dt, unit='year',
            people=people, diseases=hpvs + others, connectors=connectors, networks=networks, demographics=demographics,
            analyzers=analyzers, interventions=interventions, total_pop=total_pop, pars=sim_pars,
        )
        return

    def init(self, **kwargs):
        """ Initialize; with v2_compat_demographics, whole-year initial ages """
        with warnings.catch_warnings(): # TODO(v4-rs): each genotype's treatment transitions are undriven until a product names them (W209)
            warnings.filterwarnings('ignore', message='W209 UndrivenTransition: "(hpv16|hpv18|hi5|ohr)\\.treat_')
            super().init(**kwargs)
        if self._v2_compat_demographics:
            uids = self.people.auids
            self.people.age.raw[uids] = np.floor(self.people.age.raw[uids])
        return self

    def init_dists(self):
        """ TODO(v4-rs): engines' distributions hold the sim before the search, which then finds every distribution again through them (port_requests HPVsim #1) """
        for mod in self.modules:
            for dist in getattr(getattr(mod, 'engine', None), 'dists', {}).values():
                dist.sim = None
        return super().init_dists()

    @property
    def genotypes(self):
        """ The HPV genotype diseases """
        return [d for d in self.diseases.values() if isinstance(d, HPV)]


class _NewAgentImmunity(ss.Demographics):
    """
    TODO(v4-rs): when people are added, the arrays grow past the agents in use, with NaN; a cross-immunity modifier
    reads every slot, and E212 sees the NaN (port_requests HPVsim #11). Unused slots get no immunity (0).
    """
    def __init__(self, **kwargs):
        super().__init__(name='new_agent_immunity', **kwargs)
        return

    def step(self):
        for h in self.sim.diseases.values():
            if isinstance(h, HPV):
                for key in ['nab', 'cell']:
                    raw = h[key].raw
                    raw[np.isnan(raw)] = 0.0
        return


def _art_by_year(art):
    """ ART coverage by year, averaged over ages and sexes, as {year: share} """
    df = art.groupby('year')['coverage'].mean()
    return {float(y): float(v) for y, v in df.items()}
