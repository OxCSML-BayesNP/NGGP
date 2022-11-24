import sys

sys.path.append('../')

import numpy as np
import numpy.random as npr

from scipy.stats import geninvgauss

from scipy.special import kv, kve, gammaln

import particles.distributions as dists

import particles.smc_samplers as smc

tol = 1e-16

class VGamma3IID(smc.StaticModel):
    params_name = {
        'log_lambda': 'lambda',
        'log_alpha': 'alpha'
    }

    params_latex = {
        'log_lambda': '\\lambda',
        'log_alpha': '\\alpha'
    }

    params_transform = {
        'log_alpha': np.exp,
        'log_lambda': np.exp
    }

    def logpyt(self, theta, t):
        alpha = np.exp(theta['log_alpha'])
        lam = np.exp(theta['log_lambda'])

        abs_y = np.maximum(np.abs(self.data[t]), tol)

        # Numerical issues when computing the modified Bessel function
        #print("Value of (y, alpha, lambda) = ({}, {}, {})".format(abs_y, alpha, lam))
        #print("Bessel function = {}".format(kv(lam - 0.5, alpha * abs_y)))
        if kv(lam - 0.5, alpha * abs_y) > 0:
            return 2 * lam * np.log(alpha) + (lam - 0.5) * np.log(abs_y) + \
                   np.log(kv(lam - 0.5, alpha*abs_y)) - 0.5 * np.log(np.pi) - \
                   gammaln(lam) - (lam - 0.5)*np.log(2*alpha)

        return 2 * lam * np.log(alpha) + (lam - 0.5) * np.log(abs_y) + \
               np.log(kve(lam - 0.5, alpha*abs_y)) - abs_y - 0.5 * np.log(np.pi) - \
               gammaln(lam) - (lam - 0.5) * np.log(2 * alpha)

    @staticmethod
    def simulate(log_lambda, log_alpha, T):
        lam = np.exp(log_lambda)
        alpha = np.exp(log_alpha)

        variances = np.random.gamma(lam, 2./alpha**2, size=T)

        return np.random.normal(scale=np.sqrt(variances))


    @staticmethod
    def get_prior():
        prior_dict = {
            'log_alpha': dists.LogD(dists.Gamma(a=.1, b=.1)),
            'log_lambda': dists.LogD(dists.Gamma(a=.1, b=.1))
        }
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = VGamma3IID.get_prior()
        theta0 = prior.rvs()
        theta0['log_alpha'] = np.log(2.0) + 0.1 * npr.randn()
        theta0['log_lambda'] = np.log(2.0) + 0.1 * npr.randn()
        return theta0
