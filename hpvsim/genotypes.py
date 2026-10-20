"""
HPV natural history: each genotype is a declared Starsim disease, and the genotypes a family (`ss.Strains`)

States of each genotype:

    S    susceptible
    P    infection without a lesion (infectious)
    CIN  cervical intraepithelial neoplasia (infectious; women only)
    CA   invasive cervical cancer (not infectious)
    Z    another genotype's cancer: this genotype's course has stopped (the family's `exclusive='CA'`)

Transitions:

    infection  S -> P          per sex act, over a sexual network
    precin     P -> CIN | S    women: after dur_precin; P(CIN) = logf2 of the time spent in P
    clear_m    P -> S          men: after dur_inf_male
    cin        CIN -> CA | S   after dur_cin; P(cancer) from the integral of logf2 over the time spent in CIN
    death      CA -> (dead)    after dur_cancer
    cure       P, CIN -> S     undriven: moved only by treatments

Clearance by women (naturally or by treatment) draws antibodies (`nab`), kept as a running maximum
and gated by seroconversion; `cross_immunity()` turns them into protection against each genotype.
"""

import numpy as np
import sciris as sc
import starsim as ss

from .utils import logf2, intlogf2

__all__ = ['genotype_pars', 'genotype', 'genotypes', 'cross_immunity', 'cross_matrix', 'GENOTYPES']

GENOTYPES = ('hpv16', 'hpv18', 'hi5', 'ohr')

lognorm = lambda mean, std: ss.lognorm_ex(ss.years(mean), ss.years(std))


def genotype_pars(name):
    """ HPVsim's default natural-history parameters of a genotype """
    shared = dict(
        beta         = 0.25,              # Per sex act, female to male
        transm2f     = 2.0,               # Male to female relative to beta
        dur_precin   = lognorm(2.5, 9),
        dur_cin      = lognorm(4.5, 20),
        dur_cancer   = lognorm(8, 3),
        dur_inf_male = lognorm(1, 1),
        cin_k        = 0.2,               # Growth rate of the logf2 curve of P(CIN) by the time in P (also the cancer curve's)
        x_infl       = 0,
        ttc          = 50,                # The time (years) at which the curves reach 1
        transform_prob = 1.5e-3,
        imm_init     = ss.uniform(0.5, 0.95),
        sero_prob    = 0.6,
    )
    own = dict(
        hpv16 = dict(dur_precin=lognorm(3, 9), dur_cin=lognorm(5, 20), cin_k=0.3, transform_prob=2e-3, sero_prob=0.75),
        hpv18 = dict(beta=0.25*0.75, dur_cin=lognorm(5, 20), cin_k=0.25, transform_prob=2e-3, sero_prob=0.56),
        hi5   = dict(beta=0.25*0.9),
        ohr   = dict(beta=0.25*0.9),
    )
    if name not in own:
        raise ValueError(f'Unknown genotype "{name}"; the genotypes with defaults are {sc.strjoin(own)}')
    return sc.mergedicts(shared, own[name])


def genotype(name='hpv16', init_prev=None, **kwargs):
    """
    One HPV genotype, as a declared disease

    Args:
        name (str): 'hpv16', 'hpv18', 'hi5', or 'ohr', whose defaults are used (see `genotype_pars()`)
        init_prev (float): the share infected at the start, for a genotype on its own (in a family, the family seeds)
        kwargs (dict): parameter values, e.g. `cin_k=0.35`

    Examples:
        ```python
        hpv16 = hpv.genotype('hpv16', beta=0.3)
        ```
    """
    pars = genotype_pars(name)
    unknown = set(kwargs) - set(pars)
    if unknown:
        raise ValueError(f'Unknown parameter(s) of genotype "{name}": {sc.strjoin(sorted(unknown))}; they are {sc.strjoin(pars)}')
    p = sc.objdict({key: ss.par(val, name=key) for key, val in sc.mergedicts(pars, kwargs).items()})

    S, P, CIN, CA = ss.states('S P CIN CA')
    yr = ss.years(1)
    p_cin = logf2(P.time/yr, p.cin_k, p.x_infl, ttc=p.ttc)
    dysp = intlogf2(CIN.time/yr, p.cin_k, p.x_infl, ttc=p.ttc)
    p_cancer = 1 - (1 - p.transform_prob)**(dysp**2)

    precin = ss.Transition(P, [CIN, S], p=[p_cin, None], dur=p.dur_precin, where=ss.ppl.female)
    cin = ss.Transition(CIN, [CA, S], p=[p_cancer, None], dur=p.dur_cin)
    cure = ss.Transition(P + CIN, S, undriven=True)
    return ss.Disease(S, P, CIN, CA,
        Z         = ss.State(people=True),
        infection = ss.transmission('S -> P', beta=ss.peract(m2f=p.beta*p.transm2f, f2m=p.beta), source=P + CIN),
        precin    = precin,
        clear_m   = ss.Transition(P, S, dur=p.dur_inf_male, where=ss.ppl.male),
        cin       = cin,
        death     = CA >> ss.null @ p.dur_cancer,
        cure      = cure,
        nab       = ss.attr(p.imm_init*ss.bernoulli(p.sero_prob.value), # UPSTREAM: a Dist with an ss.par parameter cannot be drawn in an ss.attr (agents.py:775)
                             on=precin.to(S) | cin.to(S) | cure.to(S), combine='max', init=0),
        infected  = P + CIN,
        init_prev = init_prev,
        name      = name,
    )


def genotypes(names=('hpv16', 'hpv18'), init_prev=0.1, split=None, pars=None):
    """
    Several genotypes, as a family: one cancer per woman, joint seeding, and results over all of them (`sim.results.hpv`)

    Args:
        names (list): the genotypes
        init_prev (float/func): the share of people infected at the start, each with one genotype
        split (list): the share of each genotype among those infected at the start (default equal)
        pars (dict): parameter values by genotype, e.g. `dict(hpv16=dict(cin_k=0.35))`

    Cross-immunity is not part of the family: add `hpv.cross_immunity(family)` to the sim's connectors.
    """
    pars = sc.mergedicts(pars)
    members = [genotype(name, **pars.get(name, {})) for name in names]
    return ss.Strains('hpv', members, init_prev=init_prev, split=split, exclusive='CA')


def cross_matrix(names, own=0.9, med=0.3, high=0.5, full_own=('hpv16', 'hpv18'), high_pairs=(('hpv16', 'hpv18'),)):
    """ HPVsim's protection matrix from antibodies, with rows the genotype protected against and columns the genotype that gave them """
    names = list(names)
    m = np.full((len(names), len(names)), med)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                m[i, j] = 1.0 if a in full_own else own
            elif (a, b) in high_pairs or (b, a) in high_pairs:
                m[i, j] = high
    return m


def cross_immunity(family, matrix=None):
    """
    Protection against infection from the antibodies of every genotype: rel_sus of genotype i is multiplied by `1 − min(1, Σ_j m[i, j]·nab_j)`

    Args:
        family (ss.Strains): the genotypes (from `hpv.genotypes()`)
        matrix (array): rows the genotype protected against, columns the genotype whose antibodies protect (default `cross_matrix()`)
    """
    members = list(family.members)
    names = [g.name for g in members]
    m = cross_matrix(names) if matrix is None else np.asarray(matrix, dtype=float)
    # UPSTREAM: ss.immunity(protected=) takes states only, not an attribute such as nab, so the family cannot hold this; one modifier per genotype instead
    return [ss.modify(g.rel_sus, multiply=1 - np.clip(sum(m[i, j]*h.nab for j, h in enumerate(members) if m[i, j]), 0, 1), name=f'nab_{g.name}')
            for i, g in enumerate(members)]
