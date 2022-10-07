==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1


Version 0.3.0 (2022-10-07)
--------------------------
- HIV was added.
- Adds people filtering.
- Fixes bug with ``print(sim)`` not working.
- *GitHub info*: PR `308 <https://github.com/amath-idm/hpvsim/pull/308>`__


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