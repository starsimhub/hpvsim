==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1

Version 2.0.0 (2023-11-29)
---------------------------
- Simplifies natural history model by compressing CIN grades
- Changes the way HPV progression is modeled so that there is a probability of developing CIN based upon duration of precin and probability of cancer based upon duration of cancer (based upon Rodriguez et al. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3705579/)
- Adds support for pre-calibration explorations
- Improvements to networks, including clustering functionality, support for different distributions for male and female partners and for differing concurrency rates, and changes to default partnership durations
- Exposes a parameter for specifying the sex ratio of a population
- Fixes plotting issue with tutorial
- Updates filtering for tests that are not genotype-specific
- *Github info* PR `643 <https://github.com/amath-idm/hpvsim/pull/643>`__ 

Version 1.2.7 (2023-09-22)
---------------------------
- Updates ``sim.summary`` to have more useful information
- *Github info* PR `618 <https://github.com/amath-idm/hpvsim/pull/618>`__

Version 1.2.6 (2023-09-22)
---------------------------
- Fixes plotting issue with MultiSims and Jupyter notebooks
- Allows scenarios to be run fully in parallel
- *Github info* PR `614 <https://github.com/amath-idm/hpvsim/pull/614>`__

Version 1.2.5 (2023-09-21)
---------------------------
- Fixes file path when run via Jupyter
- *Github info* PR `610 <https://github.com/amath-idm/hpvsim/pull/610>`__

Version 1.2.4 (2023-09-19)
---------------------------
- Fixes Matplotlib regression in plotting
- *Github info* PR `609 <https://github.com/amath-idm/hpvsim/pull/609>`__

Version 1.2.3 (2023-08-30)
---------------------------
- Updates data loading to be much more efficient
- *Github info* PR `604 <https://github.com/amath-idm/hpvsim/pull/604>`__

Version 1.2.2 (2023-08-11)
---------------------------
- Improved tests and included ``conda`` environment specification
- *Github info* PR `598 <https://github.com/amath-idm/hpvsim/pull/598>`__

Version 1.2.1 (2023-07-09)
---------------------------
- Updated data files being used
- *Github info* PR `586 <https://github.com/amath-idm/hpvsim/pull/586>`__

Version 1.2.0 (2023-05-31)
---------------------------
- Changes to improve run speed, most notably changes to how migration is applied
- Additional tests to ensure consistency between calibration results, age analyzer results, and sim results
- Updates to natural history to prevent people progressing too quickly to cancer
- *Github info* PR `576 <https://github.com/amath-idm/hpvsim/pull/576>`__

Version 1.1.5 (2023-03-23)
---------------------------
- Adds cross-protection functionality to t-cell immunity and adds `sev_imm` attribute to people
- *Github info* PR `564 <https://github.com/amath-idm/hpvsim/pull/564>`__

Version 1.1.4 (2023-03-15)
---------------------------
- Fixes bug that caused location data to be loaded twice
- *Github info* PR `546 <https://github.com/amath-idm/hpvsim/pull/546>`__

Version 1.1.3 (2023-03-14)
---------------------------
- Fixes bug that misses some ways you can specify sex for vaccination
- *Github info* PR `555 <https://github.com/amath-idm/hpvsim/pull/555>`__

Version 1.1.2 (2023-03-13)
---------------------------
- Fixes bug that never computed cancer deaths by age
- *Github info* PR `554 <https://github.com/amath-idm/hpvsim/pull/554>`__

Version 1.1.1 (2023-03-01)
---------------------------
- Sets time to and date of HIV death for those not on ART and who fail on ART
- Moves all HIV attributes, parameters, and results into hivsim class instance
- Merges HIV results with sim.results at conclusion of simulation
- Adds HIV pars as an argument to calibration as well as HIV-specific results to age-results analyzer
- Allows for flexible severity growth functions
- *Github info* PR `542 <https://github.com/amath-idm/hpvsim/pull/542>`__


Version 1.1.0 (2023-02-16)
---------------------------
- Moves all HIV functionality into hiv.py
- Establishes new class HIVsim, which is defined by a set of parameters and methods for updating a people object
- Bug fix for setting people.sev wrong on day of infection
- *Github info* PR `526 <https://github.com/amath-idm/hpvsim/pull/526>`__


Version 1.0.1 (2023-02-09)
---------------------------
- Fixes computation of dur_episomal by adjusting for dt
- *GitHub info*: PR `527 <https://github.com/amath-idm/hpvsim/pull/527>`__


Version 1.0.0 (2023-01-31)
---------------------------
- Official release!
- *GitHub info*: PR `521 <https://github.com/amath-idm/hpvsim/pull/521>`__


Version 0.4.17 (2023-01-31)
---------------------------
- Adds a tutorial on calibration
- Small changes to parameter values
- *GitHub info*: PR `520 <https://github.com/amath-idm/hpvsim/pull/520>`__


Version 0.4.16 (2023-01-30)
---------------------------
- Change to natural history, including computation of transformation based upon time with dysplasia
- Addition of cellular immunity to moderate progression in a secondary infection
- Default parameter changes and some small typo/bug fixes
- *GitHub info*: PR `513 <https://github.com/amath-idm/hpvsim/pull/513>`__


Version 0.4.15 (2023-01-13)
---------------------------
- Fixed bug in intervention and analyzer initialization
- *GitHub info*: PR `511 <https://github.com/amath-idm/hpvsim/pull/511>`__


Version 0.4.14 (2023-01-11)
---------------------------
- Add Sweep class
- *GitHub info*: PR `431 <https://github.com/amath-idm/hpvsim/pull/431>`__


Version 0.4.13 (2023-01-09)
---------------------------
- Dysplasia percentages are now tracked throughout agent lifetimes, and CIN grades are defined as properties based on these percentages
- Removes all genotypes aside from HPV 16, 18 and a composite 'other high risk' genotype from the defaults 
- *GitHub info*: PR `507 <https://github.com/amath-idm/hpvsim/pull/507>`__


Version 0.4.12 (2023-01-02)
---------------------------
- Adds documentation and examples for screening algorithms.
- *GitHub info*: PR `505 <https://github.com/amath-idm/hpvsim/pull/505>`__


Version 0.4.11 (2022-12-21)
---------------------------
- Adds colposcopy and cytology testing options, along with default values for screening sensitivity and specificity.
- Adds a clearance probability for treatment to control the % of treated women who also clear their infection
- Removes use_multiscale parameter and sets ms_agent_ratio to 1 by default
- *GitHub info*: PR `497 <https://github.com/amath-idm/hpvsim/pull/497>`__


Version 0.4.10 (2022-12-19)
---------------------------
- Change the seed used for running simulations to avoid having random processes in the model run sometimes being correlated with population attributes
- Deprecate ``Sim.set_seed()`` - use ``hpu.set_seed()`` instead
- Added ``hpvsim.rootdir`` to provide a convenient absolute path to the
- Added equality operator for `Result` objects
- Exporting simulation results to JSON now includes 2D results (e.g., by genotype)
- ``age_pyramid`` and ``age_results`` analyzer argument changed from ``datafile`` to ``data`` since this input supports both passing in a filename or a dataframe
- *GitHub info*: PR `485 <https://github.com/amath-idm/hpvsim/pull/485>`__


Version 0.4.9 (2022-12-16)
--------------------------
- Added in high- and low-grade lesions to type distribution results
- Changes default duration and rate of dysplasia for hr HPVs
- *GitHub info*: PR `479 <https://github.com/amath-idm/hpvsim/pull/482>`__


Version 0.4.8 (2022-12-14)
--------------------------
- Small bug fix to re-enable plots of cytology outcomes by genotype
- *GitHub info*: PR `484 <https://github.com/amath-idm/hpvsim/pull/484>`__


Version 0.4.7 (2022-12-13)
--------------------------
- Migration is now modeled by finding mismatches between the modeled population size by age and data on population sizes by age (previously, this adjustment was done for the overall population rather than by age bucket).
- *GitHub info*: PR `479 <https://github.com/amath-idm/hpvsim/pull/479>`__


Version 0.4.6 (2022-12-12)
--------------------------
- Changes to several default parameters: default genotypes are now 16, 18, and other high-risk; and default hpv control prob is now 0.
- Results now capture infections by age and type distributions.
- Adds age of cancer to analyzer
- Changes to default plotting styles
- Various bugfixes: prevents immunity values from exceeding 1, ensures people with cancer aren't given second cancers
- *GitHub info*: PR `458 <https://github.com/amath-idm/hpvsim/pull/458>`__


Version 0.4.5 (2022-12-06)
--------------------------
- Removes default screening products pending review
- *GitHub info*: PR `464 <https://github.com/amath-idm/hpvsim/pull/464>`__


Version 0.4.4 (2022-12-05)
--------------------------
- Changes to progression to cancer -- no longer based on clinical cutoffs, now stochastically applied by genotype to CIN3 agents
- *GitHub info*: PR `430 <https://github.com/amath-idm/hpvsim/pull/430>`__


Version 0.4.3 (2022-12-01)
--------------------------
- Fixes bug with population growth function
- *GitHub info*: PR `459 <https://github.com/amath-idm/hpvsim/pull/459>`__


Version 0.4.2 (2022-11-21)
--------------------------
- Changes to parameterization of immunity
- *GitHub info*: PR `425 <https://github.com/amath-idm/hpvsim/pull/425>`__


Version 0.4.1 (2022-11-21)
--------------------------
- Fixes age of migration
- Adds scale parameter for vital dynamics
- *GitHub info*: PR `423 <https://github.com/amath-idm/hpvsim/pull/423>`__


Version 0.4.0 (2022-11-16)
--------------------------
- Adds merge method for scenarios and fixes printing bugs
- *GitHub info*: PR `422 <https://github.com/amath-idm/hpvsim/pull/422>`__


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
