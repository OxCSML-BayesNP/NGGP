import sys
sys.path.append('../')

import numpy as np
import numpy.random as npr
import particles.distributions as dists
from scipy.stats import geninvgauss

from iid_incr import IIDIncr

class GIGaussian(dists.ProbDist):
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p

    def rvs(self, size=1):
        p = self.p
        a = self.a
        b = self.b

        theta = np.sqrt(a*b)
        scale = np.sqrt(b/a)
        return scale*geninvgauss.rvs(p, theta, size=size)

class GHDIIDIncr(IIDIncr):

    params_name = {
            'lam': 'lambda',
            'log_alpha': 'alpha',
            'log_delta': 'delta'}

    params_latex = {
            'lam': '\lambda',
            "log_alpha": '\\alpha',
            'log_delta': '\\delta'}

    params_transform = {
            'lam': lambda x: x,
            'log_alpha': np.exp,
            'log_delta': np.exp}

    def __init__(self,
                mu=0.,
                beta=0.,
                lam=1.,
                log_alpha=0.,
                log_delta=0.,
                volumes=1.):
        super(GHDIIDIncr, self).__init__(mu=mu, beta=beta)
        self.lam = lam
        self.log_alpha = log_alpha
        self.log_delta = log_delta
        self.volumes = volumes

    def get_volume(self, t):
        if np.size(self.volumes) == 1 or np.size(self.volumes) <= t:
            return np.mean(self.volumes)
        else:
            return self.volumes[t]

    def PX0(self):
        lam = self.lam
        alpha = np.exp(self.log_alpha)
        delta = np.exp(self.log_delta)

        return GIGaussian(alpha**2, delta**2, lam)

    def PX(self, t, xp):
        lam = self.lam
        alpha = np.exp(self.log_alpha)
        delta = np.exp(self.log_delta)

        return GIGaussian(alpha**2, delta**2, lam)

    @staticmethod
    def get_prior():
        prior_dict = {
            'lam': dists.Normal(loc=0., scale=1.),#W
            'log_alpha': dists.LogD(dists.Gamma(a=.1, b=.1)),#W
            'log_delta': dists.LogD(dists.Gamma(a=.1, b=.1)),
        }
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = GHDIIDIncr.get_prior()
        theta0 = prior.rvs()
        theta0['lam'] = npr.randn()
        theta0['log_alpha'] = np.log(1.0) + 0.1*npr.randn()
        theta0['log_delta'] = 0.1*npr.randn()
        return theta0
