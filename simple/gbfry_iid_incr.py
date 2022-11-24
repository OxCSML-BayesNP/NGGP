import sys
sys.path.append('../')

import numpy as np
import numpy.random as npr
import particles.distributions as dists

from gbfry import GBFRYsumrnd
from utils import logit, sigmoid, hill_estimate
from iid_incr import IIDIncr

import argparse
import os
import pickle

class GBFRYSumDist(dists.ProbDist):
    def __init__(self, eta, tau, sigma, c=1.0):
        self.eta = eta
        self.tau = tau
        self.sigma = sigma
        self.c = c

    def rvs(self, size=1):
        return GBFRYsumrnd(self.eta, self.tau, self.sigma, self.c, size)

class GBFRYIIDIncr(IIDIncr):

    params_name = {
            'log_eta':'eta',
            'log_c':'c',
            'log_tau_minus_one':'tau',
            'logit_sigma':'sigma'}

    params_latex = {
            'log_eta':'\eta',
            'log_c':'c',
            'log_tau_minus_one':'\\tau',
            'logit_sigma':'\sigma'}

    params_transform = {
            'log_eta':np.exp,
            'log_c':np.exp,
            'log_tau_minus_one': lambda x: np.exp(x) + 1.0,
            'logit_sigma':sigmoid}

    def __init__(self,
                mu=0.,
                beta=0.,
                log_eta=np.log(0.1),
                log_tau_minus_one=np.log(3.0-1.),
                logit_sigma=logit(0.2),
                log_c=np.log(1.),
                volumes=1.):
        super(GBFRYIIDIncr, self).__init__(mu=mu, beta=beta)
        self.log_eta = log_eta
        self.log_tau_minus_one = log_tau_minus_one
        self.logit_sigma = logit_sigma
        self.log_c = log_c
        self.volumes = volumes

    def get_volume(self, t):
        if np.size(self.volumes) == 1 or np.size(self.volumes) <= t:
            return np.mean(self.volumes)
        else:
            return self.volumes[t]

    def PX0(self):
        vol = self.get_volume(0)
        eta = np.exp(self.log_eta)*vol
        tau = np.exp(self.log_tau_minus_one) + 1.0
        sigma = sigmoid(self.logit_sigma)
        c = np.exp(self.log_c)
        return GBFRYSumDist(eta, tau, sigma, c)

    def PX(self, t, xp):
        vol = self.get_volume(t)
        eta = np.exp(self.log_eta)*vol
        tau = np.exp(self.log_tau_minus_one) + 1.0
        sigma = sigmoid(self.logit_sigma)
        c = np.exp(self.log_c)
        return GBFRYSumDist(eta, tau, sigma, c)

    @staticmethod
    def get_prior():
        prior_dict = {
            'log_eta': dists.LogD(dists.Gamma(a=.1, b=.1)), #W
            'log_tau_minus_one': dists.LogD(dists.Gamma(a=1., b=1.)),#W
            'logit_sigma': dists.LogitD(dists.Beta(a=1., b=1.)),
            'log_c': dists.LogD(dists.Gamma(a=.1, b=.1))#W
        }
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = GBFRYIIDIncr.get_prior()
        theta0 = prior.rvs()
        halpha = hill_estimate(y, 50)[15:].mean()
        theta0['log_c'] = np.log(1.0) + 0.1*npr.randn()
        theta0['log_eta'] = np.log(1.0) + 0.1*npr.randn()
        theta0['logit_sigma'] = sigmoid(logit(0.5) + 0.1*npr.randn())
        theta0['log_tau_minus_one'] = np.log(0.5*halpha)
        return theta0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--eta', type=float, default=.6)
    parser.add_argument('--sigma', type=float, default=.7)
    parser.add_argument('--tau', type=float, default=1.8)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    data = {}

    true_params = {
            'mu':args.mu,
            'beta':args.beta,
            'logit_sigma': logit(args.sigma),
            'log_eta':np.log(args.eta),
            'log_tau_minus_one':np.log(args.tau-1),
            'log_c':np.log(args.c)}
    model = GBFRYIIDIncr(**true_params)
    x, y = model.simulate(args.T)

    data['true_params'] = true_params
    data['x'] = x
    data['y'] = np.array(y).squeeze()

    if not os.path.isdir('../data'):
        os.makedirs('../data')

    if args.filename is None:
        filename = ('../data/gbfry_iid_'
                'T_{}_'
                'eta_{}_'
                'tau_{}_'
                'sigma_{}_'
                'c_{}'
                '.pkl'
                .format(args.T, args.eta, args.tau, args.sigma, args.c))
    else:
        filename = args.filename

    with open(filename, 'wb') as f:
        pickle.dump(data, f)
