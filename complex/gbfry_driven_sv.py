import sys
sys.path.append('../')

import argparse
import os
import pickle
import numpy as np
import numpy.random as npr
import numba as nb
import particles.distributions as dists
import matplotlib.pyplot as plt

from gbfry import GBFRYsumrnd
from utils import logit, sigmoid
from levy_driven_sv import LevyDrivenSV
from utils import logit, sigmoid, hill_estimate

import particles
from particles import state_space_models as ssm

@nb.njit(nb.f8[:,:](nb.f8, nb.f8, nb.f8, nb.f8, nb.f8[:]), fastmath=True)
def volatility_transition(eta, tau, c, lam, vp):
    N = len(vp)
    x = np.zeros((N, 2))
    for i in range(N):
        K = npr.poisson(abs(eta*lam) + 1e-5)
        x[i,1] = np.exp(-lam)*vp[i]
        dz = 0
        for j in range(K):
            w = npr.exponential(1./c)/(npr.beta(tau, 1) + 1e-20)
            x[i,1] += w*np.exp(-lam*npr.rand())
            dz += w
        x[i,0] = (vp[i] - x[i,1] + dz)/lam
    return x

class PX0(dists.ProbDist):
    def __init__(self, eta, tau, c, lam):
        self.eta = eta
        self.tau = tau
        self.c = c
        self.lam = lam

    def rvs(self, size=1, n_warmup=1):
        #v0 = GBFRYsumrnd(self.eta/self.c**self.tau, self.tau, 0.0, self.c, size)
        v0 = GBFRYsumrnd(self.eta, self.tau, 0.0, self.c, size)
        for _ in range(n_warmup-1):
            v0 = volatility_transition(self.eta, self.tau, self.c, self.lam, v0)
            v0 = v0[...,1]
        return volatility_transition(self.eta, self.tau, self.c, self.lam, v0)

class PX(dists.ProbDist):
    def __init__(self, eta, tau, c, lam, xp):
        self.eta = eta
        self.tau = tau
        self.c = c
        self.lam = lam
        self.vp = xp[...,1]

    def rvs(self, size=1):
        return volatility_transition(self.eta, self.tau,
                self.c, self.lam, self.vp)

class GBFRYDrivenSV(LevyDrivenSV):

    params_name = {
            'log_eta':'eta',
            'log_tau_minus_one':'tau',
            'log_c':'c',
            'log_lam':'lambda'}

    params_latex = {
            'log_eta':'\eta',
            'log_tau_minus_one':'\\tau',
            'log_c':'c',
            'log_lam':'\lambda'}

    params_transform = {
            'log_eta':np.exp,
            'log_tau_minus_one':lambda x: np.exp(x) + 1.0,
            'log_c':np.exp,
            'log_lam':np.exp}

    def __init__(self,
            mu=0.0,
            beta=0.0,
            log_eta=np.log(4.0),
            log_tau_minus_one=np.log(2.0-1.0),
            log_c = np.log(1.0),
            log_lam = np.log(0.01)):
        super(GBFRYDrivenSV, self).__init__(
                mu=mu,
                beta=beta,
                log_lam=log_lam)
        self.log_eta = log_eta
        self.log_tau_minus_one = log_tau_minus_one
        self.log_c = log_c

    def PX0(self):
        eta = np.exp(self.log_eta)
        tau = np.exp(self.log_tau_minus_one) + 1
        c = np.exp(self.log_c)
        lam = np.exp(self.log_lam)
        return PX0(eta, tau, c, lam)

    def PX(self, t, xp):
        eta = np.exp(self.log_eta)
        tau = np.exp(self.log_tau_minus_one) + 1
        c = np.exp(self.log_c)
        lam = np.exp(self.log_lam)
        return PX(eta, tau, c, lam, xp)

    @staticmethod
    def get_prior():
        prior_dict = {
                'log_eta':dists.LogD(dists.Gamma(a=0.1, b=0.1)),
                'log_tau_minus_one':dists.LogD(dists.Gamma(a=1.0, b=1.0)),
                'log_c':dists.LogD(dists.Gamma(a=0.1, b=0.1)),
                'log_lam':dists.LogD(dists.Gamma(a=0.1, b=0.1))}
        #prior_dict = {
        #        'log_eta':dists.LogD(dists.Gamma(a=1.0, b=1.0)),
        #        'log_tau_minus_one':dists.LogD(dists.Gamma(a=1.0, b=1.0)),
        #        'log_c':dists.LogD(dists.Gamma(a=1.0, b=1.0)),
        #        'log_lam':dists.LogD(dists.Gamma(a=1.0, b=1.0))}
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = GBFRYDrivenSV.get_prior()
        theta0 = prior.rvs()
        halpha = hill_estimate(y, 50)[15:].mean()
        theta0['log_tau_minus_one'] = np.log(max(0.5*halpha-1, 1e-2)) + 0.1*npr.randn()
        theta0['log_c'] = np.log(1.0) + 0.1*npr.randn()
        theta0['log_eta'] = np.log(1.0) + 0.1*npr.randn()
        theta0['log_lam'] = np.log(0.01) + 0.1*npr.randn()
        return theta0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--lam', type=float, default=1e-2)
    parser.add_argument('--eta', type=float, default=5.0)
    parser.add_argument('--tau', type=float, default=1.5)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    data = {}

    true_params = {
            'mu':args.mu,
            'beta':args.beta,
            'log_eta':np.log(args.eta),
            'log_tau_minus_one':np.log(args.tau-1),
            'log_c':np.log(args.c),
            'log_lam':np.log(args.lam)}
    model = GBFRYDrivenSV(**true_params)
    x, y = model.simulate(args.T)
    y = np.array(y)
    x = np.array(x)[:,0,1]

    data['true_params'] = true_params
    data['x'] = x
    data['y'] = np.array(y).squeeze()

    if not os.path.isdir('../data'):
        os.makedirs('../data')

    if args.filename is None:
        filename = ('../data/gbfry_driven_sv_'
                'T_{}_'
                'eta_{}_'
                'tau_{}_'
                'c_{}'
                '.pkl'
                .format(args.T, args.eta, args.tau, args.c))
    else:
        filename = args.filename

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    plt.figure('x')
    plt.plot(x)
    plt.figure('y')
    plt.plot(y)
    plt.show()
