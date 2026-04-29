"""
Tests for hpvsim/parameters.py — vaccine dose parameters and severity computation.
"""

import numpy as np
import sciris as sc
import hpvsim as hpv
from hpvsim import parameters as hppar

hpv.options.set(interactive=False)


def test_get_vaccine_dose_pars():
    """Test get_vaccine_dose_pars for various vaccines."""

    # Default returns all pars
    all_pars = hppar.get_vaccine_dose_pars()
    assert isinstance(all_pars, dict)
    assert 'bivalent' in all_pars

    # Get default vaccine pars
    default_pars = hppar.get_vaccine_dose_pars(default=True)
    assert 'imm_init' in default_pars
    assert 'doses' in default_pars

    # Get specific vaccine
    for vaccine in ['bivalent', 'bivalent_2dose', 'bivalent_3dose', 'quadrivalent', 'nonavalent']:
        vp = hppar.get_vaccine_dose_pars(vaccine=vaccine)
        assert 'imm_init' in vp
        assert 'doses' in vp
        assert isinstance(vp['doses'], int)

    # Multi-dose vaccine should have imm_boost
    pars_3dose = hppar.get_vaccine_dose_pars(vaccine='bivalent_3dose')
    assert pars_3dose['doses'] == 3
    assert pars_3dose['imm_boost'] is not None


def test_compute_inv_severity():
    """Test compute_inv_severity with different functional forms."""

    sev_vals = np.array([0.1, 0.3, 0.5, 0.7])

    # logf2 form (default)
    pars_logf2 = dict(form='logf2', k=0.5, x_infl=10)
    result = hppar.compute_inv_severity(sev_vals, pars=pars_logf2)
    assert len(result) == len(sev_vals)
    assert np.all(np.isfinite(result))

    # logf3 form
    pars_logf3 = dict(form='logf3', k=0.5, x_infl=10, s=1.0)
    result3 = hppar.compute_inv_severity(sev_vals, pars=pars_logf3)
    assert len(result3) == len(sev_vals)
    assert np.all(np.isfinite(result3))

    # With rel_sev scaling
    rel_sev = np.array([1.5, 1.5, 1.5, 1.5])
    result_scaled = hppar.compute_inv_severity(sev_vals, rel_sev=rel_sev, pars=dict(form='logf2', k=0.5, x_infl=10))
    assert np.all(result_scaled < result)  # Higher rel_sev -> shorter time

    # Custom callable form
    custom_form = lambda sev_vals, **kw: sev_vals * 10
    pars_custom = dict(form=custom_form)
    result_custom = hppar.compute_inv_severity(sev_vals, pars=pars_custom)
    assert np.allclose(result_custom, sev_vals * 10)


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()
    test_get_vaccine_dose_pars()
    test_compute_inv_severity()
    sc.toc(T)
    print('Done.')