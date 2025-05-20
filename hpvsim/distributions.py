import starsim as ss
import scipy.stats as sps

__all__ = ["beta"]


class beta(ss.Dist):

    def __init__(self, a=0.0, b=1.0, **kwargs):  # Does not accept dtype
        super().__init__(distname="beta", dist=sps.beta, a=a, b=b, **kwargs)
        return
