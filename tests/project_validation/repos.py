"""
Project registry and helper functions for the HPVsim project validation suite.

Defines project configurations, cloning utilities, and sim construction
for reproducing natural-history results from published HPVsim projects.
"""

import os
import subprocess
import numpy as np
import sciris as sc
import hpvsim as hpv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
thisdir = sc.thisdir(__file__)
repos_dir = os.path.join(thisdir, '.repos')

# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------
_ages_16 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]

_default_init_hpv_dist = dict(hpv16=0.4, hpv18=0.25, hi5=0.25, ohr=0.1)

_default_init_hpv_prev = dict(
    age_brackets=np.array([12, 17, 24, 34, 44, 64, 80, 150]),
    m=np.array([0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
    f=np.array([0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
)

_1dose_layer_probs_default = dict(
    m=np.array([
        _ages_16,
        [0, 0, 0.05, 0.25, 0.70, 0.90, 0.95, 0.70, 0.75, 0.65, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40],
        [0, 0, 0.01, 0.01, 0.10, 0.50, 0.60, 0.70, 0.70, 0.70, 0.70, 0.80, 0.70, 0.60, 0.50, 0.60],
    ]),
    c=np.array([
        _ages_16,
        [0, 0, 0.10, 0.70, 0.80, 0.60, 0.60, 0.50, 0.20, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0, 0, 0.05, 0.70, 0.80, 0.60, 0.60, 0.50, 0.50, 0.40, 0.30, 0.10, 0.05, 0.01, 0.01, 0.01],
    ]),
)

_1dose_layer_probs_ethiopia = dict(
    m=np.array([
        _ages_16,
        [x * 0.7 for x in [0, 0, 0.05, 0.25, 0.70, 0.90, 0.95, 0.70, 0.75, 0.65, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40]],
        [0, 0, 0.01, 0.01, 0.10, 0.50, 0.60, 0.70, 0.70, 0.70, 0.70, 0.80, 0.70, 0.60, 0.50, 0.60],
    ]),
    c=np.array([
        _ages_16,
        [0, 0, 0.3, 0.21, 0.3, 0.3, 0.3, 0.6, 0.6, 1.2, 0.3, 0.3, 0.3, 0.03, 0.03, 0.03],
        [0, 0, 0.3, 0.21, 0.3, 0.3, 0.3, 0.6, 1.2, 2.1, 1.5, 0.9, 0.3, 0.03, 0.03, 0.03],
    ]),
)

_1dose_partners = dict(
    m_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2)),
    f_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2)),
)

# ---------------------------------------------------------------------------
# Project registry
# ---------------------------------------------------------------------------
PROJECT_REPOS = dict(

    hpv_faster_kenya=dict(
        url='https://github.com/starsimhub/hpv_faster_kenya',
        location='kenya',
        pars_file='results/kenya_pars.obj',
        genotypes=[16, 18, 'hi5', 'ohr'],
        start=1960,
        end=2020,
        dt=0.25,
        n_agents=10e3,
        debut=dict(
            f=dict(dist='lognormal', par1=17, par2=4),
            m=dict(dist='lognormal', par1=18, par2=4),
        ),
        layer_probs=dict(
            m=np.array([
                _ages_16,
                [0, 0, 0, 0.1, 0.7, 0.8, 0.8, 0.6, 0.65, 0.3, 0.2, 0.2, 0.1, 0.07, 0.035, 0.007],
                [0, 0, 0, 0.1, 0.2, 0.71, 0.9, 0.6, 0.6, 0.45, 0.3, 0.3, 0.1, 0.1, 0.05, 0.01],
            ]),
            c=np.array([
                _ages_16,
                [0, 0, 0.05, 0.2, 0.6, 0.5, 0.4, 0.35, 0.35, 0.3, 0.2, 0.2, 0.10, 0.02, 0.02, 0.02],
                [0, 0, 0.01, 0.2, 0.4, 0.5, 0.5, 0.5, 0.6, 0.5, 0.3, 0.2, 0.02, 0.02, 0.02, 0.02],
            ]),
        ),
        m_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2)),
        f_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2)),
        target_files=['data/kenya_cancer_cases.csv', 'data/kenya_cancer_types.csv'],
    ),

    hpvsim_india=dict(
        url='https://github.com/hpvsim/hpvsim_india',
        location='india',
        pars_file='results/india_pars.obj',
        genotypes=[16, 18, 'hi5', 'ohr'],
        start=1960,
        end=2020,
        dt=0.25,
        n_agents=10e3,
        debut=dict(
            f=dict(dist='lognormal', par1=15, par2=2),
            m=dict(dist='lognormal', par1=20, par2=2),
        ),
        layer_probs=dict(
            m=np.array([
                _ages_16,
                [0, 0, 0.05, 0.25, 0.60, 0.80, 0.95, 0.80, 0.80, 0.65, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40],
                [0, 0, 0.01, 0.05, 0.10, 0.70, 0.90, 0.90, 0.90, 0.90, 0.80, 0.60, 0.60, 0.60, 0.60, 0.60],
            ]),
            c=np.array([
                _ages_16,
                [0, 0, 0.10, 0.50, 0.60, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.01, 0.01],
                [0, 0, 0.10, 0.20, 0.25, 0.35, 0.40, 0.70, 0.90, 0.90, 0.95, 0.95, 0.70, 0.30, 0.10, 0.10],
            ]),
        ),
        m_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.1)),
        f_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='neg_binomial', par1=2, par2=0.025)),
        f_cross_layer=0.025,
        m_cross_layer=0.25,
        target_files=['data/india_hpv_prevalence.csv', 'data/india_cancer_cases.csv'],
    ),

    hpvsim_rwanda=dict(
        url='https://github.com/hpvsim/hpvsim_rwanda',
        location='rwanda',
        pars_file='results/rwanda_pars.obj',
        genotypes=[16, 18, 'hi5', 'ohr'],
        start=1960,
        end=2025,
        dt=0.25,
        n_agents=10e3,
        debut=dict(
            f=dict(dist='lognormal', par1=20.96, par2=3.34),
            m=dict(dist='lognormal', par1=17.91, par2=2.83),
        ),
        init_hpv_dist=_default_init_hpv_dist,
        init_hpv_prev=_default_init_hpv_prev,
        layer_probs=dict(
            m=np.array([
                _ages_16,
                [0, 0, 0.025, 0.0115, 0.1555, 0.313, 0.3875, 0.408, 0.3825, 0.334, 0.275, 0.20, 0.20, 0.20, 0.20, 0.20],
                [0, 0, 0.01, 0.023, 0.311, 0.626, 0.775, 0.816, 0.765, 0.668, 0.70, 0.80, 0.70, 0.60, 0.50, 0.60],
            ]),
            c=np.array([
                _ages_16,
                [0, 0, 0.1, 0.6, 0.3, 0.2, 0.2, 0.2, 0.2, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0, 0, 0.1, 0.3, 0.4, 0.3, 0.3, 0.4, 0.5, 0.50, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ]),
        ),
        m_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2)),
        f_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2)),
        target_files=[],
    ),

    hpvsim_1dose_nigeria=dict(
        url='https://github.com/hpvsim/hpvsim_1dose',
        location='nigeria',
        pars_file='results/nigeria_pars.obj',
        genotypes=[16, 18, 'hi5', 'ohr'],
        start=1960,
        end=2020,
        dt=0.25,
        n_agents=10e3,
        init_hpv_dist=_default_init_hpv_dist,
        init_hpv_prev=_default_init_hpv_prev,
        layer_probs=_1dose_layer_probs_default,
        **_1dose_partners,
        target_files=[],
    ),

    hpvsim_1dose_ethiopia=dict(
        url='https://github.com/hpvsim/hpvsim_1dose',
        location='ethiopia',
        pars_file='results/ethiopia_pars.obj',
        genotypes=[16, 18, 'hi5', 'ohr'],
        start=1960,
        end=2020,
        dt=0.25,
        n_agents=10e3,
        init_hpv_dist=_default_init_hpv_dist,
        init_hpv_prev=_default_init_hpv_prev,
        layer_probs=_1dose_layer_probs_ethiopia,
        **_1dose_partners,
        target_files=[],
    ),

    hpvsim_1dose_cambodia=dict(
        url='https://github.com/hpvsim/hpvsim_1dose',
        location='cambodia',
        pars_file='results/cambodia_pars.obj',
        genotypes=[16, 18, 'hi5', 'ohr'],
        start=1960,
        end=2020,
        dt=0.25,
        n_agents=10e3,
        init_hpv_dist=_default_init_hpv_dist,
        init_hpv_prev=_default_init_hpv_prev,
        layer_probs=_1dose_layer_probs_default,
        **_1dose_partners,
        target_files=[],
    ),

    hpvsim_pxv_younger=dict(
        url='https://github.com/amath-idm/hpvsim_pxv_younger',
        location='nigeria',
        pars_file='results/nigeria_pars.obj',
        genotypes=[16, 18, 'hi5', 'ohr'],
        start=1960,
        end=2020,
        dt=0.25,
        n_agents=10e3,
        debut=dict(
            f=dict(dist='lognormal', par1=16, par2=4),
            m=dict(dist='lognormal', par1=18, par2=4),
        ),
        layer_probs=dict(
            m=np.array([
                _ages_16,
                [0, 0, 0, 0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.3, 0.4, 0.4, 0.2, 0.07, 0.035, 0.007],
                [0, 0, 0, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.4, 0.4, 0.4, 0.2, 0.1, 0.05, 0.01],
            ]),
            c=np.array([
                _ages_16,
                [0, 0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.6, 0.2, 0.10, 0.02, 0.02, 0.02],
                [0, 0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.6, 0.5, 0.2, 0.02, 0.02, 0.02, 0.02],
            ]),
        ),
        m_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2)),
        f_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2)),
        target_files=[],
    ),
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clone_repos(projects=None, force=False):
    """
    Shallow-clone project repos into the ``.repos/`` directory.

    Args:
        projects (list): list of project names to clone (default: all)
        force (bool): if True, re-clone even if the directory already exists
    """
    os.makedirs(repos_dir, exist_ok=True)
    if projects is None:
        projects = list(PROJECT_REPOS.keys())

    # Deduplicate URLs so we only clone each repo once
    url_to_names = {}
    for name in projects:
        cfg = PROJECT_REPOS[name]
        url_to_names.setdefault(cfg['url'], []).append(name)

    for url, names in url_to_names.items():
        # Derive a directory name from the URL (last path component without .git)
        repo_dirname = url.rstrip('/').split('/')[-1].replace('.git', '')
        dest = os.path.join(repos_dir, repo_dirname)
        if os.path.isdir(dest) and not force:
            print(f'  Skipping {repo_dirname} (already cloned)')
            continue
        if os.path.isdir(dest) and force:
            sc.rmpath(dest)
        print(f'  Cloning {url} -> {dest}')
        subprocess.run(
            ['git', 'clone', '--depth', '1', url, dest],
            check=True,
        )

    return repos_dir


def _repo_dir(name):
    """Return the local clone directory for a project."""
    cfg = PROJECT_REPOS[name]
    repo_dirname = cfg['url'].rstrip('/').split('/')[-1].replace('.git', '')
    return os.path.join(repos_dir, repo_dirname)


def load_project_pars(name):
    """
    Load calibrated ``.obj`` parameters from a cloned project repo.

    Calibrated pars were tuned under the old engine where cross-layer
    probabilities were used as per-timestep values.  The current engine
    treats them as annual probabilities, so we convert here.

    Args:
        name (str): project name from ``PROJECT_REPOS``

    Returns:
        dict: calibrated parameter dictionary (with probabilities converted to annual)
    """
    cfg = PROJECT_REPOS[name]
    dt = cfg['dt']
    pars_path = os.path.join(_repo_dir(name), cfg['pars_file'])
    if not os.path.isfile(pars_path):
        raise FileNotFoundError(
            f'Pars file not found for {name}: {pars_path}. '
            f'Have you run clone_repos()?'
        )
    pars = sc.load(pars_path)

    # Strip HIV-specific pars if present (we only validate natural history)
    if isinstance(pars, dict) and 'hiv_pars' in pars:
        del pars['hiv_pars']

    # Convert per-timestep cross-layer probabilities to annual
    for key in ['m_cross_layer', 'f_cross_layer']:
        if key in pars:
            p = np.clip(pars[key], 0, 1 - 1e-10)
            pars[key] = 1 - (1 - p) ** (1 / dt)

    # Convert per-timestep layer_probs if present in calibrated pars
    if 'layer_probs' in pars:
        pars['layer_probs'] = _convert_layer_probs_to_annual(pars['layer_probs'], dt)

    return pars


def get_target_data(name):
    """
    Return a list of absolute paths to calibration target CSVs for a project.

    Args:
        name (str): project name from ``PROJECT_REPOS``

    Returns:
        list: list of file paths (may be empty)
    """
    cfg = PROJECT_REPOS[name]
    repo = _repo_dir(name)
    paths = []
    for rel in cfg.get('target_files', []):
        p = os.path.join(repo, rel)
        if os.path.isfile(p):
            paths.append(p)
        else:
            print(f'  Warning: target file not found: {p}')
    return paths


def _convert_layer_probs_to_annual(layer_probs, dt):
    """
    Convert per-timestep layer_probs to annual probabilities.

    Project repos calibrated layer_probs as per-timestep values under the old
    HPVsim engine. The current branch (fix/dt-invariance) now treats layer_probs
    as annual probabilities and internally scales them by dt. To get the same
    effective rates, we invert the old per-timestep interpretation:

        p_annual = 1 - (1 - p_timestep)^(1/dt)

    Args:
        layer_probs (dict): layer_probs dict with arrays per layer key
        dt (float): the timestep the values were originally calibrated for

    Returns:
        dict: layer_probs with values converted to annual probabilities
    """
    converted = {}
    for lkey, lp in layer_probs.items():
        lp_new = lp.copy()
        for row in [1, 2]:  # Row 0 is age bins, rows 1/2 are female/male probs
            vals = lp_new[row, :]
            # Clamp to [0, 1) before conversion to avoid complex numbers
            vals = np.clip(vals, 0, 1 - 1e-10)
            lp_new[row, :] = 1 - (1 - vals) ** (1 / dt)
        converted[lkey] = lp_new
    return converted


def make_project_sim(name, seed=0):
    """
    Construct a natural-history ``hpv.Sim`` using the project's configuration
    and calibrated parameters.

    No interventions or custom analyzers are included.

    Args:
        name (str): project name from ``PROJECT_REPOS``
        seed (int): random seed

    Returns:
        hpv.Sim: configured (but not yet initialized) simulation
    """
    cfg = PROJECT_REPOS[name]

    # Load calibrated pars
    calib_pars = load_project_pars(name)

    dt = cfg['dt']

    # Build base sim pars
    sim_pars = dict(
        location=cfg['location'],
        genotypes=cfg['genotypes'],
        start=cfg['start'],
        end=cfg['end'],
        dt=dt,
        n_agents=cfg['n_agents'],
        ms_agent_ratio=100,
        rand_seed=seed,
        verbose=0,
        model_hiv=False,
    )

    # Optional demographic / initialization pars
    if 'debut' in cfg:
        sim_pars['debut'] = cfg['debut']

    if 'init_hpv_dist' in cfg:
        sim_pars['init_hpv_dist'] = cfg['init_hpv_dist']

    if 'init_hpv_prev' in cfg:
        sim_pars['init_hpv_prev'] = cfg['init_hpv_prev']

    # Convert per-timestep layer_probs to annual (needed because the current
    # engine treats layer_probs as annual and applies dt-scaling internally)
    sim_pars['layer_probs'] = _convert_layer_probs_to_annual(cfg['layer_probs'], dt)

    # Convert cross-layer probabilities the same way
    if 'f_cross_layer' in cfg:
        p = cfg['f_cross_layer']
        sim_pars['f_cross_layer'] = 1 - (1 - p) ** (1 / dt)
    if 'm_cross_layer' in cfg:
        p = cfg['m_cross_layer']
        sim_pars['m_cross_layer'] = 1 - (1 - p) ** (1 / dt)

    # Partnership pars (not probability-based, no conversion needed)
    sim_pars['m_partners'] = cfg['m_partners']
    sim_pars['f_partners'] = cfg['f_partners']

    # Create the sim
    sim = hpv.Sim(pars=sim_pars)

    # Apply calibrated pars on top
    sim.update_pars(calib_pars)

    return sim
