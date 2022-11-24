import numpy as np
import numpy.random as npr
import numba as nb
from etstablernd import etstablernd

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def GGPsumrnd(eta, sigma, c, n):
    if sigma < 1e-8:
        S = np.zeros(n)
        for i in range(n):
            S[i] = npr.gamma(eta, 1./c)
        return S
    else:
        return etstablernd(eta/sigma, sigma, c, n)

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def GBFRYsumrndold(eta, tau, sigma, c, n):
    kappa = tau - sigma
    S = GGPsumrnd(eta*c**kappa/kappa, sigma, c, n)
    for i in range(n):
        K = npr.poisson(eta*c**tau/tau/kappa)
        for _ in range(K):
            S[i] += npr.gamma(1-sigma, 1)/(npr.beta(tau, 1)*c + 1e-10)
    return S

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def GBFRYsumrnd(eta, tau, sigma, c, n):
    S = GGPsumrnd(eta/c**sigma, sigma, c, n)
    for i in range(n):
        K = npr.poisson(eta/tau)
        for _ in range(K):
            S[i] += npr.gamma(1-sigma, 1)/(npr.beta(tau, 1)*c + 1e-20)
    return S
