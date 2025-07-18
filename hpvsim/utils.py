import numpy as np
import sciris as sc


__all__ = ["logf2", "compute_cancer_prob"]


def get_asymptotes(k, x_infl, s=1, y_max=1, ttc=25):
    """
    Get upper asymptotes for logistic functions
    """
    term1 = (
        1 + np.exp(k * (x_infl - ttc))
    ) ** s  # Note, this is 1 for most parameter combinations
    term2 = (1 + np.exp(k * x_infl)) ** s
    u_asymp_num = y_max * term1 * (1 - term2)
    u_asymp_denom = term1 - term2
    u_asymp = u_asymp_num / u_asymp_denom
    l_asymp = y_max * term1 / (term1 - term2)
    return l_asymp, u_asymp


def logf3(x, k, x_infl, s=1, y_max=1, ttc=25):
    """
    Logistic function passing through (0,0) and (ttc,y_max).
    This version is derived from the 5-parameter version here: https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
    However, since it's constrained to pass through 2 points, there are 3 free parameters remaining.
    Args:
         k: growth rate, equivalent to b in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         x_infl: a location parameter, equivalent to C in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         s: asymmetry parameter, equivalent to s in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         ttc (time to cancer): x value for which the curve passes through 1. For x values beyond this, the function returns 1
    """
    l_asymp, u_asymp = get_asymptotes(k, x_infl, s=1, y_max=y_max, ttc=ttc)
    return np.minimum(
        1, l_asymp + (u_asymp - l_asymp) / (1 + np.exp(k * (x_infl - x))) ** s
    )


def logf2(x, k, x_infl=0, y_max=1, ttc=25):
    """
    Logistic function constrained to pass through (0,0) and (ttc,y_max).
    This version is derived from the 5-parameter version here: https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
    Since it's constrained to pass through 2 points, there are 3 free parameters remaining, and this verison fixes s=1
    Args:
         k: growth rate, equivalent to b in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         x_infl: point of inflection, equivalent to C in https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
         ttc (time to cancer): x value for which the curve passes through 1. For x values beyond this, the function returns 1
    """
    return logf3(x, k, x_infl, s=1, y_max=y_max, ttc=ttc)


def indef_int_logf2(x, k, x_infl, ttc=25, y_max=1):
    """
    Indefinite integral of logf2; see definition there for arguments
    """
    t1 = 1 + np.exp(k * (x_infl - ttc))
    t2 = 1 + np.exp(k * x_infl)
    integ = np.log(np.exp(k * (x_infl - x)) + 1) / k + x
    result = y_max / (t1 - t2) * (1 - t1 * t2 * integ)
    return result


def intlogf2(upper, k, x_infl=0, ttc=25, y_max=1):
    """
    Integral of logf2 between 0 and the limit given by upper
    """
    # Find the upper limits not including the part past time to cancer
    exceeding_ttc_inds = (upper > ttc).nonzero()
    lims_to_find = np.minimum(ttc, upper)

    # Take the integral
    val_at_0 = indef_int_logf2(0, k, x_infl, ttc)
    val_at_lim = indef_int_logf2(lims_to_find, k, x_infl, ttc)
    integral = val_at_lim - val_at_0

    # Deal with those whose duration of infection exceeds the time to cancer
    # Note, another option would be to set their transformation probability to 1
    excess_integral = upper[exceeding_ttc_inds] - ttc
    integral[exceeding_ttc_inds] += excess_integral

    return integral


def compute_severity_integral(t, rel_sev=None, pars=None):
    """
    Compute the integral of the severity function
    """
    if rel_sev is not None:
        t = rel_sev * t
    output = intlogf2(t, **pars)
    return output


def compute_cancer_prob(t, rel_sev=None, pars=None):
    """
    Args:
        t: array of durations that women have been in their current health state
        rel_sev: array of individual relative severity values
    """

    pars = sc.dcp(pars)

    # Complete these next stages if cancer progression probabilities are being modeled
    # as the cumulative severity-time of dysplasia.
    if pars.get('method') == 'cin_integral':
        del pars['method']
        if pars.get('ld50'):
            ld50 = pars.pop('ld50')
            if pars.get('transform_prob'):
                _ = pars.pop('transform_prob')
            sev_at_ld50 = intlogf2(np.array([ld50]), pars=pars)[0]
            transform_prob = 1 - 0.5**(1/sev_at_ld50**2)
        elif pars.get('transform_prob'):
            transform_prob = pars.pop('transform_prob')
        else:
            errormsg = ('If using calculating cancer probabilities using the integral of the CIN function, must provide'
                        ' an LD50 or transform prob.')
            raise ValueError(errormsg)

        sev = compute_severity_integral(t, rel_sev=rel_sev, pars=pars)
        cancer_probs = 1 - np.power(1 - transform_prob, sev**2)
        return cancer_probs
    else:
        error_msg = f"Unknown method for computing cancer probabilities: {pars.get('method')}"
        raise ValueError(error_msg)
        return
