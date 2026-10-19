"""
HPV products, from HPVsim's product tables (``hpvsim/data/products_*.csv``), as Starsim 4 products over the genotype diseases

  - ``hpv.vx``: prophylactic vaccine (``ss.Vaccine``), efficacy by genotype
  - ``hpv.dx``: diagnostic (``ss.Test``), the probability of each result by state and genotype
  - ``hpv.tx``: treatment (``ss.Treatment``), moving people along each genotype's ``treat_*`` transitions
  - ``hpv.txvx``: therapeutic vaccine (``ss.Treatment``): the lesion- and infection-clearing part
  - ``hpv.radiation``: cancer treatment (``ss.Treatment`` that holds): slower cancer death

Each takes the genotypes it acts on (``genotypes=``); ``hpv.Sim`` fills them in from the sim's
genotypes when not given.
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

from .parameters import GENOTYPE_KEYS

__all__ = ['vx', 'dx', 'tx', 'txvx', 'radiation', 'default_vx', 'default_dx', 'default_tx']

_DATA = Path(__file__).parent / 'data'

# HPVsim's states, as the declared genotype's states (susceptible is people in none of them)
_STATES = dict(latent='L', precin='P', cin='C', cancerous='X')

# The results of each test, mildest first (HPVsim 3 lists them most severe first)
_DX_RESULTS = {
    'via':           ['negative', 'inadequate', 'positive'],
    'lbc':           ['normal', 'inadequate', 'ascus', 'abnormal'],
    'pap':           ['normal', 'inadequate', 'ascus', 'abnormal'],
    'colposcopy':    ['normal', 'ascus', 'lsil', 'hsil', 'cancer'],
    'hpv':           ['negative', 'inadequate', 'positive'],
    'hpv1618':       ['negative', 'inadequate', 'positive'],
    'hpv_type':      ['negative', 'inadequate', 'positive_ohr', 'positive_1618'],
    'txvx_assigner': ['none', 'txvx', 'triage'],
    'tx_assigner':   ['none', 'ablation', 'excision', 'radiation'],
}


@functools.lru_cache(maxsize=None)
def _load(kind):
    df = pd.read_csv(_DATA / f'products_{kind}.csv')
    return df


def _table(kind, name):
    df = _load(kind)
    if name not in set(df['name']):
        raise ValueError(f'Unknown {kind} product {name!r}. Valid names: {sc.strjoin(df["name"].unique())}.')
    return df[df['name'] == name].reset_index(drop=True)


def default_vx(name):
    """ {genotype: rel_imm} for a vaccine in products_vx.csv """
    df = _table('vx', name)
    return dict(zip(df['genotype'], df['rel_imm'].astype(float)))


def default_dx(name):
    """ The rows of a diagnostic in products_dx.csv """
    return _table('dx', name)


def default_tx(name):
    """ The rows of a treatment in products_tx.csv """
    return _table('tx', name)


def _genotypes(genotypes):
    return list(GENOTYPE_KEYS) if genotypes is None else [str(g) for g in genotypes]


class _GenotypeProduct:
    """ A product that is built for the genotypes of the sim: ``hpv.Sim`` calls ``set_genotypes()`` if none were given """
    def set_genotypes(self, genotypes, force=False):
        if self.genotypes is None or force:
            self.genotypes = list(genotypes)
            self.build()
        return self


class vx(_GenotypeProduct, ss.Vaccine):
    """
    Prophylactic HPV vaccine, with efficacy by genotype

    Efficacy against genotype g is ``rel_imm[g]`` from products_vx.csv, times the mean protection of
    HPVsim's all-or-nothing draw (a share ``sterilizing_p`` fully protected, the rest at ``sterilizing_p``),
    as a reduction in susceptibility.

    Args:
        name (str): 'bivalent', 'quadrivalent', or 'nonavalent' (or give rel_imm)
        rel_imm (dict): {genotype: relative immunity}, instead of a named product
        sterilizing_p (float): HPVsim 3's share with sterilizing immunity (default 0.95)
        genotypes (list): the genotype diseases it acts on (default: filled in by hpv.Sim)
        kwargs (dict): passed to `ss.Vaccine` (e.g. `doses=[0.85, 1.0]`, `cost=`, `waning=`)
    """
    def __init__(self, name=None, rel_imm=None, sterilizing_p=0.95, genotypes=None, product_name=None, **kwargs):
        if (name is None) == (rel_imm is None):
            raise ValueError('hpv.vx requires exactly one of `name` or `rel_imm`.')
        self.rel_imm = default_vx(name) if rel_imm is None else dict(rel_imm)
        self.sterilizing_p = sterilizing_p
        self.genotypes = genotypes
        super().__init__(efficacy=None, name=product_name or name or 'vx', **kwargs)
        self._default_name = False
        if genotypes is not None:
            self.build()
        return

    def build(self):
        p = self.sterilizing_p
        mean = p + (1 - p) * p # The mean protection of the all-or-nothing draw, per unit of rel_imm
        self.efficacy = {g: float(np.clip(self.rel_imm.get(g, 0.0) * mean, 0, 1)) for g in self.genotypes}
        return


class dx(_GenotypeProduct, ss.Test):
    """
    HPV diagnostic, with the probability of each result by state and genotype, from products_dx.csv

    Results with zero probability everywhere (e.g. 'inadequate') are dropped. A product whose rows are
    for 'all' genotypes reads each state across genotypes (the highest chance of each result,
    ``combine='max'``); one by genotype detects each genotype independently, and the worst result
    is the person's (``combine='worst'``).

    Args:
        name (str): e.g. 'via', 'hpv', 'hpv1618', 'hpv_type', 'lbc', 'colposcopy', 'tx_assigner'
        df (DataFrame): rows [state, genotype, result, probability] instead of a named product
        results (list): the results, mildest first (default: from the name, or the order in df)
        genotypes (list): the genotype diseases it reads (default: filled in by hpv.Sim)
        kwargs (dict): passed to `ss.Test` (e.g. `delay=`, `cost=`)
    """
    def __init__(self, name=None, df=None, results=None, genotypes=None, product_name=None, **kwargs):
        if name is None and df is None:
            raise ValueError('hpv.dx requires at least one of `name` or `df`.')
        self.df = default_dx(name) if df is None else df
        self.result_order = results or _DX_RESULTS.get(name) or list(self.df['result'].unique())
        self.genotypes = genotypes
        self.by_genotype = not set(self.df['genotype']) == {'all'}
        super().__init__(outcomes=self._outcomes(list(GENOTYPE_KEYS)), combine='worst' if self.by_genotype else 'max', name=product_name or name or 'dx', **kwargs)
        self._default_name = False
        if genotypes is not None:
            self.build()
        return

    def _outcomes(self, genotypes):
        df = self.df
        out = {}
        for k, result in enumerate(self.result_order):
            rows = df[df['result'] == result]
            if k and not rows['probability'].any():
                continue # A result that never happens
            if not k:
                out[result] = None # The first takes the rest
                continue
            table = {}
            for state, short in _STATES.items():
                for g in genotypes:
                    row = rows[(rows['state'] == state) & (rows['genotype'].isin([g, 'all']))]
                    if len(row):
                        table[f'{g}.{short}'] = float(row['probability'].values[0])
            sus = rows[rows['state'] == 'susceptible'] # People in none of the states
            if len(sus) and sus['probability'].max() > 0:
                table[None] = float(sus['probability'].max())
            out[result] = table
        return out

    def build(self):
        self.outcomes = self._outcomes(self.genotypes)
        return


class tx(_GenotypeProduct, ss.Treatment):
    """
    HPV treatment from products_tx.csv: moves people with precin, CIN, or a latent infection to susceptible, with
    efficacy by state and genotype; people with cancer are not cured by it

    Args:
        name (str): 'ablation', 'excision', or a therapeutic vaccine's row ('txvx1', 'txvx2')
        df (DataFrame): rows [state, genotype, efficacy] instead of a named product
        genotypes (list): the genotype diseases it acts on (default: filled in by hpv.Sim)
        kwargs (dict): passed to `ss.Treatment` (e.g. `cost=`)
    """
    _TRANSITIONS = dict(precin='treat_precin', cin='treat_cin', latent='treat_latent')

    def __init__(self, name=None, df=None, genotypes=None, product_name=None, **kwargs):
        if name is None and df is None:
            raise ValueError('hpv.tx requires at least one of `name` or `df`.')
        self.df = default_tx(name) if df is None else df
        self.genotypes = genotypes
        super().__init__(moves=self._moves(list(GENOTYPE_KEYS)), name=product_name or name or 'tx', **kwargs)
        self._default_name = False
        if genotypes is not None:
            self.build()
        return

    def _moves(self, genotypes):
        df = self.df
        out = {}
        for g in genotypes:
            for state, trname in self._TRANSITIONS.items():
                row = df[(df['state'] == state) & (df['genotype'].isin([g, 'all']))]
                if len(row) and row['efficacy'].values[0] > 0:
                    out[f'{g}.{trname}'] = float(row['efficacy'].values[0])
        return out

    def build(self):
        self.moves = self._moves(self.genotypes)
        return


class txvx(tx):
    """
    Therapeutic HPV vaccine: clears infections and lesions with the efficacy of its rows in products_tx.csv ('txvx1', 'txvx2')

    TODO: the severity immunity HPVsim 3 also confers (imm_init/imm_boost, products_txvx.csv) is not ported yet.
    """
    def __init__(self, name='txvx1', **kwargs):
        super().__init__(name=name, **kwargs)
        return


class radiation(_GenotypeProduct, ss.Treatment):
    """
    Radiation for cervical cancer: held, it slows cancer death

    HPVsim 3 adds ``dur`` (normal, mean 18 months) to the time of death. As a rate, that is a
    lower cancer-death hazard: efficacy ``dur/(dur_remaining + dur)``, where ``dur_remaining`` is the
    mean time to death left at treatment (default 7 years: 8 from onset, less 1 to detection).

    Args:
        dur (float/dur): the years of life it adds on average (default 1.5)
        dur_remaining (float): the mean years left at treatment (default 7)
        genotypes (list): the genotype diseases it acts on (default: filled in by hpv.Sim)
    """
    def __init__(self, dur=1.5, dur_remaining=7.0, genotypes=None, product_name='radiation', **kwargs):
        if isinstance(dur, dict): # HPVsim 3's dict(dist='normal', par1=mean, par2=std)
            dur = dur['par1']
        dur = getattr(dur, 'years', dur)
        self.eff = float(dur / (dur_remaining + dur))
        self.genotypes = genotypes
        super().__init__(efficacy={g: dict(cancer_death=self.eff) for g in GENOTYPE_KEYS}, name=product_name, **kwargs)
        self._default_name = False
        if genotypes is not None:
            self.build()
        return

    def build(self):
        self.efficacy = {g: dict(cancer_death=self.eff) for g in self.genotypes}
        return


def products_in(interventions):
    """ The HPV products in a list of deliveries and their follow-ups """
    out = []
    seen = set()
    def walk(d):
        if id(d) in seen or not isinstance(d, ss.Delivery):
            return
        seen.add(id(d))
        if isinstance(d.product, _GenotypeProduct):
            out.append(d.product)
        for f in d.followups:
            walk(f)
    for d in sc.tolist(interventions):
        walk(d)
    return out
