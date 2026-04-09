"""
Tests for hpvsim/people.py — People class methods including make_naive and story.
"""

import numpy as np
import sciris as sc
import hpvsim as hpv

hpv.options.set(interactive=False)


def test_make_naive():
    """Test People.make_naive() resets agent state."""
    sim = hpv.Sim(n_agents=1e3, n_years=10, genotypes=[16, 18], verbose=0)
    sim.run()

    people = sim.people
    # Find some people who have been infected (have non-nan date_infectious)
    infected = np.where(~np.isnan(people.date_infectious[0, :]))[0]
    if len(infected) > 0:
        test_inds = infected[:5]
        people.make_naive(test_inds)

        # After make_naive, these people should be susceptible
        for g in range(sim['n_genotypes']):
            assert np.all(people.susceptible[g, test_inds] == True)
            assert np.all(people.infectious[g, test_inds] == False)

        # Immunity should be reset
        assert np.all(people.peak_imm[:, test_inds] == 0)
        assert np.all(people.nab_imm[:, test_inds] == 0)

        # Dates should be nan
        assert np.all(np.isnan(people.date_infectious[:, test_inds]))


def test_story():
    """Test People.story() prints agent history without errors."""
    sim = hpv.Sim(n_agents=500, n_years=10, verbose=0)
    sim.run()

    # Should not raise for any agent
    sim.people.story(0)

    # Multiple agents via *args
    sim.people.story(0, 1, 2)

    # List input
    sim.people.story([3, 4])


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()
    test_make_naive()
    test_story()
    sc.toc(T)
    print('Done.')