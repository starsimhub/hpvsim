import starsim as ss
import scipy.stats as sps

__all__ = ["beta", "beta_mean"]


class beta(ss.Dist):

    def __init__(self, a=0.0, b=1.0, **kwargs):  # Does not accept dtype
        super().__init__(distname="beta", dist=sps.beta, a=a, b=b, **kwargs)
        return


class beta_mean(ss.Dist):
    def __init__(self, par1=0.0, par2=1.0, **kwargs):  # Does not accept dtype
        a = ((1 - par1)/par2 - 1/par1) * par1**2
        b = a * (1 / par1 - 1)
        super().__init__(distname="beta", dist=sps.beta, a=a, b=b, **kwargs)
        return
