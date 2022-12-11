==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1


Version 0.4.6 (2022-12-12)
--------------------------
- Changes to several default parameters: default genotypes are now 16, 18, and other high-risk; and default hpv control prob is now 0.
 - Results now capture infections by age and type distributions.
- Adds age of cancer to analyzer
- Changes to default plotting styles
- Various bugfixes: prevents immunity values from exceeding 1, ensures people with cancer aren't given second cancers
- *GitHub info*: PRs `458 <https://github.com/amath-idm/hpvsim/pull/458>`__


Version 0.4.5 (2022-12-06)
--------------------------
- Removes default screening products pending review
- *GitHub info*: PRs `464 <https://github.com/amath-idm/hpvsim/pull/464>`__


Version 0.4.4 (2022-12-05)
--------------------------
- Changes to progression to cancer -- no longer based on clinical cutoffs, now stochastically applied by genotype to CIN3 agents
- *GitHub info*: PRs `430 <https://github.com/amath-idm/hpvsim/pull/430>`__


Version 0.4.3 (2022-12-01)
--------------------------
- Fixes bug with population growth function
- *GitHub info*: PRs `459 <https://github.com/amath-idm/hpvsim/pull/459>`__


Version 0.4.2 (2022-11-21)
--------------------------
- Changes to parameterization of immunity
- *GitHub info*: PRs `425 <https://github.com/amath-idm/hpvsim/pull/425>`__


Version 0.4.1 (2022-11-21)
--------------------------
- Fixes age of migration
- Adds scale parameter for vital dynamics
- *GitHub info*: PRs `423 <https://github.com/amath-idm/hpvsim/pull/423>`__


Version 0.4.0 (2022-11-16)
--------------------------
- Adds merge method for scenarios and fixes printing bugs
- *GitHub info*: PRs `422 <https://github.com/amath-idm/hpvsim/pull/422>`__


Version 0.3.9 (2022-11-15)
--------------------------
- Simplifies genotype initialization, adds checks for HIV runs.
- Since the last release, changes were also made to virological clearance rates for people receiving treatment - previously all treated people would clear infection, but now some may control latently instead.
- *GitHub info*: PRs `421 <https://github.com/amath-idm/hpvsim/pull/421>`__ and `420 <https://github.com/amath-idm/hpvsim/pull/420>`__


Version 0.3.8 (2022-11-02)
--------------------------
- Store treatment properties as part of sim.people
- *GitHub info*: PR `413 <https://github.com/amath-idm/hpvsim/pull/413>`__


Version 0.3.7 (2022-11-01)
--------------------------
- Fix to ensure consistent results for the number of txvx doses 
- *GitHub info*: PR `411 <https://github.com/amath-idm/hpvsim/pull/411>`__


Version 0.3.6 (2022-11-01)
--------------------------
- Fix bug related to screening eligibility. NB, this has a sizeable impact on results - screening strategies will be much more effective after this fix. 
- *GitHub info*: PR `396 <https://github.com/amath-idm/hpvsim/pull/396>`__


Version 0.3.5 (2022-10-31)
--------------------------
- Store stocks related to interventions
- *GitHub info*: PR `395 <https://github.com/amath-idm/hpvsim/pull/395>`__


Version 0.3.4 (2022-10-31)
--------------------------
- Bugfixes for therapeutic vaccination
- *GitHub info*: PR `394 <https://github.com/amath-idm/hpvsim/pull/394>`__


Version 0.3.3 (2022-10-30)
--------------------------
- Changes to therapeautic vaccine efficacy assumptions
- *GitHub info*: PR `393 <https://github.com/amath-idm/hpvsim/pull/393>`__


Version 0.3.2 (2022-10-26)
--------------------------
- Additional tutorials and minor release tidying
- *GitHub info*: PR `380 <https://github.com/amath-idm/hpvsim/pull/380>`__


Version 0.3.1 (2022-10-26)
--------------------------
- Fixes bug with screening
- Increases coverage of baseline test
- *GitHub info*: PR `373 <https://github.com/amath-idm/hpvsim/pull/373>`__


Version 0.3.0 (2022-10-26)
--------------------------
- Implements multiscale modeling
- Minor release tidying
- *GitHub info*: PR `365 <https://github.com/amath-idm/hpvsim/pull/365>`__


Version 0.2.11 (2022-10-25)
---------------------------
- Changes the way dates of HPV clearance are assigned to use durations sampled
- *GitHub info*: PR `374 <https://github.com/amath-idm/hpvsim/pull/374>`__


Version 0.2.10 (2022-10-24)
---------------------------
- Fixes bug with treatment
- *GitHub info*: PR `354 <https://github.com/amath-idm/hpvsim/pull/354>`__


Version 0.2.9 (2022-10-18)
--------------------------
- Prevents infectious people from being passed to People.infect()
- Fixes bugs with initialization within scenario runs 
- Remove ununsed prevalence results
- *GitHub info*: PR `338 <https://github.com/amath-idm/hpvsim/pull/345>`__


Version 0.2.8 (2022-10-17)
--------------------------
- Fixes bug with intervention year interpolation
- Changes reactivation probabilities to annual, not per time step
- Refactor prognoses calls
- *GitHub info*: PR `338 <https://github.com/amath-idm/hpvsim/pull/338>`__



Version 0.2.7 (2022-10-14)
--------------------------
- Adds robust relative paths via ``hpv.datadir``
- *GitHub info*: PR `333 <https://github.com/amath-idm/hpvsim/pull/333>`__


Version 0.2.6 (2022-10-12)
--------------------------
- Removes Numba since slower for small sims and only 10% faster for large sims.
- Moves functions from ``utils.py`` into ``people.py``, ``sim.py``, and ``population.py``.
- *GitHub info*: PR `326 <https://github.com/amath-idm/hpvsim/pull/326>`__


Version 0.2.5 (2022-10-07)
--------------------------
- Adds people filtering (NB: not used, and later removed).
- Fixes bug with ``print(sim)`` not working.
- Adds baseline tests.
- *GitHub info*: PR `310 <https://github.com/amath-idm/hpvsim/pull/310>`__


Version 0.2.4 (2022-10-07)
--------------------------
- Changes to dysplasia progression parameterization
- Adds a new implementation of HPV natural history for HIV positive women 
- Note: HIV was added since the previous version
- *GitHub info*: PR `304 <https://github.com/amath-idm/hpvsim/pull/304>`__


Version 0.2.3 (2022-09-01)
--------------------------
- Adds a ``use_migration`` parameter that activates immigration/emigration to ensure population sizes line up with data.
- Adds simple data versioning.
- *GitHub info*: PR `279 <https://github.com/amath-idm/hpvsim/pull/279>`__


Version 0.2.2 (2022-08-22)
--------------------------
- Separates out the ``Calibration`` class into a separate file and to no longer inherit from ``Analyzer``. Functionality is unchanged.
- *GitHub info*: PR `255 <https://github.com/amath-idm/hpvsim/pull/255>`__


Version 0.2.1 (2022-08-19)
--------------------------
- Improves calibration to enable support for MySQL.
- Fixes plotting bug.
- *GitHub info*: PR `253 <https://github.com/amath-idm/hpvsim/pull/253>`__


Version 0.2.0 (2022-08-19)
--------------------------
- Fixed tests and data loading logic.
- *GitHub info*: PR `251 <https://github.com/amath-idm/hpvsim/pull/251>`__


Version 0.1.0 (2022-08-01)
--------------------------
- Updated calibration.
- *GitHub info*: PR `215 <https://github.com/amath-idm/hpvsim/pull/215>`__


Version 0.0.3 (2022-07-18)
--------------------------
- Updated data loading scripts.
- *GitHub info*: PR `156 <https://github.com/amath-idm/hpvsim/pull/156>`__


Version 0.0.2 (2022-06-15)
--------------------------
- Made into a Python module.
- *GitHub info*: PR `64 <https://github.com/amath-idm/hpvsim/pull/64>`__


Version 0.0.1 (2022-04-04)
--------------------------
- Initial version.
