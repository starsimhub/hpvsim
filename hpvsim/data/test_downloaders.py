"""
Test data downloaders; does not require a full HPVsim install
"""

import downloaders as dl

kwargs = dict(verbose=True)
ad = dl.get_age_data(**kwargs)
as = dl.get_age_sex_data(**kwargs)
bd = dl.get_birth_data(**kwargs)
dd = dl.get_death_data(**kwargs)
ed = dl.get_ex_data(**kwargs)