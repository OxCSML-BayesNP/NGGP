import sys

sys.path.append('../')

import numpy as np
import numpy.random as npr

from scipy.stats import geninvgauss

from scipy.special import kv, kve, gammaln

import particles.distributions as dists

import particles.smc_samplers as smc

tol = 1e-16

class VGamma4IID(smc.StaticModel):
    params_name = {
        'log_eta': 'eta',
        'log_c': 'c'
    }

    params_latex = {
        'log_eta': '\\eta',
        'log_c': 'c'
    }

    params_transform = {
        'log_eta': np.exp,
        'log_c': np.exp
    }

    def logpyt(self, theta, t):
        eta = np.exp(theta['log_eta'])
        c = np.exp(theta['log_c'])

        alpha = np.sqrt(2*c)
        lam = eta

        abs_y = np.maximum(np.abs(self.data[t]), tol)

        # Numerical issues when computing the modified Bessel function
        if kv(lam - 0.5, alpha * abs_y) > 0:
            return 2 * lam * np.log(alpha) + (lam - 0.5) * np.log(abs_y) + \
                   np.log(kv(lam - 0.5, alpha*abs_y)) - 0.5 * np.log(np.pi) - \
                   gammaln(lam) - (lam - 0.5)*np.log(2*alpha)

        return 2 * lam * np.log(alpha) + (lam - 0.5) * np.log(abs_y) + \
               np.log(kve(lam - 0.5, alpha*abs_y)) - abs_y - 0.5 * np.log(np.pi) - \
               gammaln(lam) - (lam - 0.5) * np.log(2 * alpha)

    @staticmethod
    def simulate(log_eta, log_c, T):
        eta = np.exp(log_eta)
        c = np.exp(log_c)

        variances = np.random.gamma(eta, 1./c, size=T)

        return np.random.normal(scale=np.sqrt(variances))


    @staticmethod
    def get_prior():
        prior_dict = {
            'log_eta': dists.LogD(dists.Gamma(a=.1, b=.1)),
            'log_c': dists.LogD(dists.Gamma(a=.1, b=.1))
        }
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = VGamma4IID.get_prior()
        theta0 = prior.rvs()
        theta0['log_eta'] = np.log(1.0) + 0.1 * npr.randn()
        theta0['log_c'] = np.log(1.0) + 0.1 * npr.randn()
        return theta0
