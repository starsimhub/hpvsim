# Tests

This folder contains the fast unit/integration tests for HPVsim. The whole suite runs in well under a minute; the expensive statistical acceptance gates live in [`../devtests`](../devtests) and are not collected here.

## Installation

To install test dependencies, use `pip install -r requirements.txt`.

## Usage

Recommended usage is `./run_tests` or `./check_coverage`. You can also use `pytest` to run all the tests in the folder.

If you want to test a specific version of HPVsim, you can use the included `conda` environments, e.g.:

    conda env create -f hpvsim_v1.2.2.yml

## Writing tests here

Sim cost is dominated by a fixed per-timestep overhead rather than by the agent count, so the cheapest way to buy statistical signal (cancers, especially) is a *wide, coarse* run — more agents at a larger `dt` — rather than a narrow, fine one. Where a test needs resolved cancer events, raising `ms_agent_ratio` is cheaper still. Note that cancer counts fall sharply as `dt` grows, so when shrinking a sim, assert that whatever the test depends on is actually present rather than guarding it behind an `if`; several tests here do exactly that so a future shrink fails loudly instead of passing vacuously.

If a test genuinely needs many seeds or a real calibration to be meaningful, it belongs in `./devtests`, not here.
