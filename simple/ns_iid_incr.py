import sys

sys.path.append('../')

import numpy as np
import numpy.random as npr
import particles.distributions as dists

from gbfry import Gammasumrnd
from utils import logit, sigmoid
from iid_incr import IIDIncr

tol = 1e-20
exp = np.exp
log = np.log
pi = np.pi
sin = np.sin
sqrt = np.sqrt


def sdiv(x, y):
    return x / (y + tol)


def zolotarev(u, sigma):
    expn = min(sdiv(1, 1 - sigma), 50.0)
    x = sdiv(sin(sigma * u) ** sigma * sin((1 - sigma) * u) ** (1 - sigma), sin(u)) ** expn
    return x


def stablernd(alpha, size=1):
    # cf Devroye, 2009, Equation (2)
    U = npr.uniform(low=0.0, high=np.pi, size=size)
    E = npr.exponential(size=size)
    samples = (zolotarev(U, alpha) / E) ** ((1 - alpha) / alpha)
    return samples


class StableSumDist(dists.ProbDist):
    def __init__(self, eta, sigma):
        self.eta = eta
        self.sigma = sigma

    def rvs(self, size=1):
        eta = self.eta
        sigma = self.sigma

        return stablernd(sigma, size=size) * eta ** (1 / sigma)


class NSIIDIncr(IIDIncr):
    params_name = {
        'log_eta': 'eta',
        'logit_sigma': 'sigma'}

    params_latex = {
        'log_eta': '\eta',
        'logit_sigma': '\sigma'}

    params_transform = {
        'log_eta': np.exp,
        'logit_sigma': sigmoid}

    def __init__(self,
                 mu=0.,
                 beta=0.,
                 log_eta=np.log(0.1),
                 logit_sigma=logit(0.2),
                 volumes=1.):
        super(NSIIDIncr, self).__init__(mu=mu, beta=beta)
        self.log_eta = log_eta
        self.logit_sigma = logit_sigma
        self.volumes = volumes

    def get_volume(self, t):
        if np.size(self.volumes) == 1 or np.size(self.volumes) <= t:
            return np.mean(self.volumes)
        else:
            return self.volumes[t]

    def PX0(self):
        vol = self.get_volume(0)
        eta = np.exp(self.log_eta) * vol
        sigma = sigmoid(self.logit_sigma)
        return StableSumDist(eta, sigma)

    def PX(self, t, xp):
        vol = self.get_volume(t)
        eta = np.exp(self.log_eta) * vol
        sigma = sigmoid(self.logit_sigma)
        return StableSumDist(eta, sigma)

    @staticmethod
    def get_prior():
        prior_dict = {'log_eta': dists.LogD(dists.Gamma(a=.1, b=.1)),
                      'logit_sigma': dists.LogitD(dists.Beta(a=1., b=1.))}
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = NSIIDIncr.get_prior()
        theta0 = prior.rvs()
        theta0['log_eta'] = np.log(1.0) + 0.1 * npr.randn()
        theta0['logit_sigma'] = sigmoid(logit(0.5) + 0.1 * npr.randn())
        return theta0
