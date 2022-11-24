import argparse
import os
import pickle
import numpy as np
import numpy.random as npr
import numba as nb
import particles.distributions as dists
from levy_driven_sv import LevyDrivenSV
import matplotlib.pyplot as plt

# x: 2 dim vector, x[0] = intergrated volatility bar V, x[1] = volatility V
@nb.njit(nb.f8[:,:](nb.f8, nb.f8, nb.f8, nb.f8[:]), fastmath=True)
def volatility_transition(eta, c, lam, vp):
    N = len(vp)
    x = np.zeros((N, 2))
    for i in range(N):
        K = npr.poisson(eta*lam)
        x[i,1] = np.exp(-lam)*vp[i]
        dz = 0
        for j in range(K):
            w = npr.exponential(1./c)
            x[i,1] += w*np.exp(-lam*npr.rand())
            dz += w
        x[i,0] = (vp[i] - x[i,1] + dz)/lam
    return x

class PX0(dists.ProbDist):
    def __init__(self, eta, c, lam):
        self.eta = eta
        self.c = c
        self.lam = lam

    def rvs(self, size=1, n_warmup=1):
        v0 = npr.gamma(self.eta, 1./self.c, size=size)
        for _ in range(n_warmup-1):
            v0 = volatility_transition(self.eta, self.c, self.lam, v0)
            v0 = v0[...,1]
        return volatility_transition(self.eta, self.c, self.lam, v0)

class PX(dists.ProbDist):
    def __init__(self, eta, c, lam, xp):
        self.eta = eta
        self.c = c
        self.lam = lam
        self.vp = xp[...,1]

    def rvs(self, size=1):
        return volatility_transition(self.eta, self.c, self.lam, self.vp)

class GammaDrivenSV(LevyDrivenSV):

    params_name = {
            'log_eta':'eta',
            'log_c':'c',
            'log_lam':'lambda'}

    params_latex = {
            'log_eta':'\eta',
            'log_c':'c',
            'log_lam':'\lambda'}

    params_transform = {
            'log_eta':np.exp,
            'log_c':np.exp,
            'log_lam':np.exp}

    def __init__(self, mu=0.0, beta=0.0,
            log_eta=np.log(4.0), log_c=np.log(8.0), log_lam=np.log(0.01)):
        super(LevyDrivenSV, self).__init__(mu=mu, beta=beta)
        self.mu = mu
        self.beta = beta
        self.log_eta = log_eta
        self.log_c = log_c
        self.log_lam =log_lam

    def PX0(self):
        eta = np.exp(self.log_eta)
        c = np.exp(self.log_c)
        lam = np.exp(self.log_lam)
        return PX0(eta, c, lam)

    def PX(self, t, xp):
        eta = np.exp(self.log_eta)
        c = np.exp(self.log_c)
        lam = np.exp(self.log_lam)
        return PX(eta, c, lam, xp)

    @staticmethod
    def get_prior():
        prior_dict = {
                'log_eta':dists.LogD(dists.Gamma(a=0.1, b=0.1)),
                'log_c':dists.LogD(dists.Gamma(a=0.1, b=0.1)),
                'log_lam':dists.LogD(dists.Gamma(a=0.1, b=0.1))}
        return dists.StructDist(prior_dict)

    @staticmethod
    def get_theta0(y):
        prior = GammaDrivenSV.get_prior()
        theta0 = prior.rvs()
        theta0['log_eta'] = np.log(1.0) + 0.1*npr.randn()
        theta0['log_c'] = np.log(1.0) + 0.1*npr.randn()
        theta0['log_lam'] = np.log(0.01) + 0.1*npr.randn()
        return theta0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--eta', type=float, default=4.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    data = {}

    true_params = {'mu':0.0,
            'beta':0.0,
            'log_lam':np.log(0.01),
            'log_eta':np.log(args.eta),
            'log_c':np.log(args.c)}
    model = GammaDrivenSV(**true_params)
    x, y = model.simulate(args.T)
    y = np.array(y)
    x = np.array(x)[:,0,1]

    data['true_params'] = true_params
    data['x'] = x
    data['y'] = y

    if not os.path.isdir('../data'):
        os.makedirs('../data')

    if args.filename is None:
        filename = ('../data/gamma_driven_sv_'
                'T_{}_'
                'eta_{}_'
                'c_{}'
                '.pkl'
                .format(args.T, args.eta, args.c))
    else:
        filename = args.filename

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    plt.figure('x')
    plt.plot(x)
    plt.figure('y')
    plt.plot(y)
    plt.show()
