"""
HPV products from HPVsim's tables (`hpvsim/data/products_*.csv`), as Starsim products over the genotypes

    hpv.vaccine('bivalent')     an ss.Vaccine, efficacy by genotype
    hpv.diagnostic('via')       an ss.Diagnostic with HPVsim's results, most severe first
    hpv.treatment('ablation')   an ss.Treatment along each genotype's `cure`

Each takes the genotypes of the sim (`genotypes=`, names), since a product keyed by a disease
that is not in the sim is an error.
"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['vaccine', 'diagnostic', 'treatment', 'dx_results']

datadir = Path(__file__).parent / 'data'

# HPVsim's states, as states of a genotype; 'susceptible' is people in none of them (the key None)
states = dict(susceptible=None, precin='P', cin='CIN', cancerous='CA')

# The results of each test, most severe first: the last is everyone the others do not take
dx_results = dict(
    via        = ['positive', 'inadequate', 'negative'],
    lbc        = ['abnormal', 'ascus', 'inadequate', 'normal'],
    pap        = ['abnormal', 'ascus', 'inadequate', 'normal'],
    colposcopy = ['cancer', 'hsil', 'lsil', 'ascus', 'normal'],
    hpv        = ['positive', 'inadequate', 'negative'],
    hpv1618    = ['positive', 'inadequate', 'negative'],
    hpv_type   = ['positive_1618', 'positive_ohr', 'inadequate', 'negative'],
)

# Vaccine efficacy against a genotype it targets: HPVsim gives 95% of people sterilizing immunity and the rest 95%
vx_efficacy = 0.95 + 0.05*0.95


@functools.cache
def load(kind):
    return pd.read_csv(datadir / f'products_{kind}.csv')


def table(kind, name):
    df = load(kind)
    if name not in set(df['name']):
        raise ValueError(f'Unknown {kind} product "{name}"; they are {sc.strjoin(df["name"].unique())}')
    return df[df['name'] == name]


def vaccine(name='bivalent', genotypes=('hpv16', 'hpv18'), **kwargs):
    """
    A prophylactic vaccine from HPVsim's table: its efficacy against each genotype is its `rel_imm` there times 0.9975

    Args:
        name (str): 'bivalent', 'quadrivalent', or 'nonavalent'
        genotypes (list): the genotypes in the sim
        kwargs (dict): passed to `ss.Vaccine`, e.g. `doses=[0.8, 1.0]`
    """
    df = table('vx', name)
    rel = dict(zip(df['genotype'], df['rel_imm'].astype(float)))
    missing = [g for g in genotypes if g not in rel]
    if missing:
        raise ValueError(f'The vaccine "{name}" has no efficacy for {sc.strjoin(missing)}')
    efficacy = {g: vx_efficacy*rel[g] for g in genotypes}
    return ss.Vaccine(efficacy=efficacy, name=kwargs.pop('name', name), **kwargs)


def diagnostic(name='via', genotypes=('hpv16', 'hpv18'), **kwargs):
    """
    A diagnostic from HPVsim's table: the probability of each result by state, for any genotype or by genotype

    A test whose table is for any genotype ('all') reads, for each result, the highest probability over the
    states a woman is in (`combine='max'`); one by genotype reads each genotype and takes the most severe
    result (`combine='worst'`), as HPVsim does.

    Args:
        name (str): a test in products_dx.csv, e.g. 'hpv', 'via', 'lbc', 'colposcopy'
        genotypes (list): the genotypes in the sim
        kwargs (dict): passed to `ss.Diagnostic`, e.g. `delay=`, `cost=`
    """
    df = table('dx', name)
    df = df[df['state'].isin(states)] # Latency is not modelled
    order = dx_results[name]
    by_genotype = not (df['genotype'] == 'all').all()
    results = {}
    for res in order[:-1]:
        probs = {}
        for _, row in df[(df['result'] == res) & (df['probability'] > 0)].iterrows():
            state = states[row['state']]
            if by_genotype:
                if row['genotype'] in genotypes:
                    if state is None:
                        raise NotImplementedError(f'The test "{name}" is positive for people susceptible to {row["genotype"]}; a test by genotype reads only infected states')
                    probs[f'{row["genotype"]}.{state}'] = float(row['probability'])
            else:
                for g in ([None] if state is None else genotypes):
                    probs[None if g is None else f'{g}.{state}'] = float(row['probability'])
        results[res] = probs
    results[order[-1]] = None
    combine = 'worst' if by_genotype else 'max'
    default = f'{name}_test' if name.startswith('hpv') else name # Not 'hpv', the family's name
    return ss.Diagnostic(results=results, combine=combine, name=kwargs.pop('name', default), **kwargs)


def treatment(name='ablation', genotypes=('hpv16', 'hpv18'), **kwargs):
    """
    A treatment from HPVsim's table: it clears the infection and lesion of each genotype (its `cure`) with the efficacy given

    Args:
        name (str): 'ablation' or 'excision'
        genotypes (list): the genotypes in the sim
        kwargs (dict): passed to `ss.Treatment`
    """
    df = table('tx', name).set_index('state')['efficacy']
    eff = {s: float(df.get(s, 0)) for s in ['precin', 'cin']}
    if eff['precin'] != eff['cin']:
        raise NotImplementedError(f'The treatment "{name}" has efficacies {eff} that differ by state; only one for both is ported')
    if df.get('cancerous', 0) > 0:
        raise NotImplementedError(f'The treatment "{name}" treats cancer, which is not ported')
    return ss.Treatment(moves={f'{g}.cure': eff['cin'] for g in genotypes}, name=kwargs.pop('name', name), **kwargs)
