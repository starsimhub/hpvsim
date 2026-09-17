"""6-panel Rwanda context figure (HIV prevalence / ART / HPV prevalence vs data)

SLATED FOR DELETION IN v3.3 (test cleanup). This is a one-off script from the
v2 -> v3 Rwanda migration, not a test: it is not collected by pytest, it has
no assertions, and several of these run a full Optuna calibration or a
multi-seed sim. They are kept for now because the v3 HIV-HPV parameterization
was derived here and the derivation is worth being able to re-read. Anything
here that should outlive 3.3 -- most likely the CalibProbe-style age-by-HIV
probes, which localizations reimplement -- needs promoting into the package
or into ``tests/`` first.
built with the CALIBRATED params.

Reuses plot_rwanda_calib.py's tested 6-panel layout and data overlays, but
monkeypatches its sim builder to calibrate_rwanda.build_sim with the best-fit
params from results/rwanda_calib/best_pars.json, and redirects the output so
the original (ported-v2) figure is not overwritten.

Run: .venv/Scripts/python.exe tests/regression/plot_rwanda_context_calibrated.py \
         [n_seeds=10] [n_agents=10000]
Saves: results/rwanda_calib/rwanda_calibrated_context.png
       tests/regression/figures/rwanda_calibrated_context.png
"""
import shutil
import sys
from pathlib import Path

import sciris as sc

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tests.regression.plot_rwanda_calib as pc  # noqa: E402
from tests.regression.calibrate_rwanda import build_sim  # noqa: E402


def main(n_seeds=10, n_agents=10000):
    bp = sc.loadjson(_ROOT / 'results' / 'rwanda_calib' / 'best_pars.json')
    params = dict(bp['best_params'])
    params.setdefault('base_beta', 0.12)
    print(f'Context figure on calibrated fit (gof={bp.get("best_gof"):.3f}): '
          f'{params}')

    def calibrated_builder(seed, n_agents, start=1960, stop=2020):
        return build_sim(seed, n_agents, start, stop, **params)

    outdir = _ROOT / 'results' / 'rwanda_calib'
    pc.build_rwanda_sim = calibrated_builder   # swap ported -> calibrated
    pc._FIGDIR = outdir                          # redirect output dir
    pc.main(n_seeds, n_agents)                    # runs + plots the 6 panels

    src = outdir / 'rwanda_calib.png'             # pc.main's hardcoded filename
    dst = outdir / 'rwanda_calibrated_context.png'
    if src.exists():
        src.replace(dst)
    figdst = _ROOT / 'tests' / 'regression' / 'figures' / 'rwanda_calibrated_context.png'
    figdst.parent.mkdir(exist_ok=True)
    shutil.copy(dst, figdst)
    print(f'\nSaved {dst}\n      {figdst}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(n_seeds=int(a[0]) if len(a) > 0 else 10,
         n_agents=int(a[1]) if len(a) > 1 else 10000)