"""
Tests for hpvsim/people.py — People class methods including story.
"""

import sciris as sc
import hpvsim as hpv

hpv.options.set(interactive=False)


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
    test_story()
    sc.toc(T)
    print('Done.')