"""
Create a simulation for running the HPV module.
"""

import starsim as ss 
import pandas as pd
import hpvsim as hpv
import stisim as sti


__all__ = ["Sim"]


class Sim(ss.Sim):
    """Custom simulation class for HMB module, inheriting from starsim.Sim."""
    def __init__(self, hpv_pars=None, n_agents=1e3, start=2020, stop=2040, location=None, verbose=1/12, datafolder=None, **kwargs):

        # Handle location
        if location is None:
            ppl = ss.People(n_agents)
            total_pop = None
            dem = []

        # TODO: Add support for more locations
        elif location in ['kenya', 'india']:
            dflocation = location.replace(" ", "_")
            total_pop = {
                'kenya': {2020: 52.2e6}[start],
                'india': {2020: 1.4e9}[start]
            }[location]
            ppl = ss.People(
                n_agents,
                age_data=pd.read_csv(f"{datafolder}/{dflocation}_age.csv", index_col="age")["value"]
            )
            fertility_data = pd.read_csv(f"{datafolder}/{dflocation}_asfr.csv")
            pregnancy = ss.Pregnancy(unit='month', fertility_rate=fertility_data)
            death_data = pd.read_csv(f"{datafolder}/{dflocation}_deaths.csv")
            death = ss.Deaths(unit='year', death_rate=death_data, rate_units=1)
            dem = [pregnancy, death]

        else:
            raise ValueError(f"Location {location} not supported")

        # Create HPV modules
        if hpv_pars is None:
            hpv_pars = {16: None, 18:None}
        hpv16 = hpv.HPV16(hpv_pars[16])
        hpv18 = hpv.HPV18(hpv_pars[18])
        dis = [hpv16, hpv18]
        hpv_connector = hpv.hpv(genotypes=[hpv16, hpv18])
        connectors = [hpv_connector]

        # Network
        nw = sti.StructuredSexual(
            unit="month",
            dt=3,
        )

        # Time
        time_args = dict(unit='month', dt=1, start=ss.date(start), stop=ss.date(stop), verbose=verbose)

        # Initialize
        super().__init__(
            people=ppl, demographics=dem, total_pop=total_pop,
            diseases=dis, networks=[nw], connectors=connectors,
            **time_args, **kwargs
        )

        return

