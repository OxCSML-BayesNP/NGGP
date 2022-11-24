import sys

sys.path.append('../')

import numpy as np
import numpy.random as npr
import particles.distributions as dists
from scipy.stats import chi2

from iid_incr import IIDIncr


class NormInvChi2(dists.ProbDist):
    def __init__(self, nu, sigma):
        self.nu = nu
        self.sigma = sigma

    def rvs(self, size=1):
        nu = self.nu
        sigma = self.sigma
        return sigma * nu / chi2.rvs(nu, size=size)


class StudentIIDIncr(IIDIncr):
    params_name = {
        'log_half_nu_minus_one': 'nu',
        'log_sigma': 'sigma'
    }

    params_latex = {
        'log_half_nu_minus_one': '\\nu',
        'log_sigma': '\sigma'
    }

    params_transform = {
        'log_half_nu_minus_one': lambda x: 2*np.exp(x) + 2.0,
        'log_sigma': np.exp
    }

    def __init__(self,
                 mu=0.,
                 beta=0.,
                 log_half_nu_minus_one=0.,
                 log_sigma=0.,
                 volumes=1.):
        super(StudentIIDIncr, self).__init__(mu=mu, beta=beta)
        self.log_half_nu_minus_one = log_half_nu_minus_one
        self.log_sigma = log_sigma
        self.volumes = volumes

    def get_volume(self, t):
        if np.size(self.volumes) == 1 or np.size(self.volumes) <= t:
            return np.mean(self.volumes)
        else:
            return self.volumes[t]

    def PX0(self):
        nu = 2*np.exp(self.log_half_nu_minus_one)+2
        sigma = np.exp(self.log_sigma)
        return NormInvChi2(nu, sigma)

    def PX(self, t, xp):
        nu = 2*np.exp(self.log_half_nu_minus_one)+2
        sigma = np.exp(self.log_sigma)
        return NormInvChi2(nu, sigma)

    @staticmethod
    def get_prior():
        prior_dict = {
            'log_half_nu_minus_one': dists.LogD(dists.Gamma(a=1., b=1.)),
            'log_sigma': dists.LogD(dists.Gamma(a=.1, b=.1))
        }
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = StudentIIDIncr.get_prior()
        theta0 = prior.rvs()
        theta0['log_half_nu_minus_one'] = np.log(1.0) + 0.1 * npr.randn()
        theta0['log_sigma'] = np.log(1.0) + 0.1 * npr.randn()
        return theta0
