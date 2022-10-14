==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1


Version 0.2.7 (2022-10-14)
--------------------------
- Adds robust relative paths via ``hpv.defaults.datadir``
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
