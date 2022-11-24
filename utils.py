import numpy as np
import numba as nb

def logit(x, lb=0., ub=1.):
    z = (x-lb)/(ub-lb)
    return np.log(z) - np.log(1-z)

def sigmoid(x, lb=0., ub=1.):
    return lb + (ub-lb)/(1 + np.exp(-x))

def hill_estimate(x, k):
    x = np.sort(abs(x))[::-1]
    log_x = np.log(x + 1e-10)[:k]
    avesumlog = np.cumsum(log_x)/np.arange(1, k+1)
    xihat = (avesumlog - log_x)[1:]
    alphahat = 1./xihat
    return alphahat

# https://stackoverflow.com/a/55060589
@nb.njit(fastmath=True,error_model='numpy')
def gammaln(z):
    """Numerical Recipes 6.1"""
    #Don't use global variables.. (They only can be changed if you recompile the function)
    coefs = np.array([
    57.1562356658629235, -59.5979603554754912,
    14.1360979747417471, -0.491913816097620199,
    .339946499848118887e-4, .465236289270485756e-4,
    -.983744753048795646e-4, .158088703224912494e-3,
    -.210264441724104883e-3, .217439618115212643e-3,
    -.164318106536763890e-3, .844182239838527433e-4,
    -.261908384015814087e-4, .368991826595316234e-5])

    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092
    for coef in coefs:
        y = y + 1.
        ser = ser + coef/y

    out = tmp + np.log(2.5066282746310005 * ser / z)
    return out

def VaR(x, p=0.99):
    return np.percentile(np.sort(x), (1-p)*100)


def ecdf(samples, x=None):
    """
    Compute empirical cdf from samples

    Parameters
    ----------
    samples
        Array of samples from the distribution of interest
    x
        Array of points where to evaluate the cdf

    Returns
    -------
    x
        Array of points where to evaluate the cdf
    cdf_eval
        Evaluations of the empirical cdf
    """
    sorted_samples = np.sort(samples)
    if x is None:
        x = sorted_samples

    cdf_eval = np.searchsorted(sorted_samples, x, side='left')
    cdf_eval = cdf_eval/len(sorted_samples)
    cdf_eval[-1] += np.sum(sorted_samples==x[-1])/len(sorted_samples)
    return cdf_eval
