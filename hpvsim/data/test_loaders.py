"""
Test data loaders; does not require a full HPVsim install
"""

import loaders as ld

kw = dict(location='nigeria')
ad = ld.get_age_distribution(year=1990, **kw)
at = ld.get_age_distribution_over_time(**kw)
tp = ld.get_total_pop(**kw)
dr = ld.get_death_rates(**kw)
br = ld.get_birth_rates(**kw)
