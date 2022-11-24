import numpy as np
import numpy.random as npr

import particles
from particles import state_space_models as ssm
import particles.distributions as dists

class LevyDrivenSV(ssm.StateSpaceModel):
    def __init__(self, mu=0.0, beta=0.0, log_lam=np.log(0.01)):
        super(LevyDrivenSV, self).__init__()
        self.mu = mu
        self.beta = beta
        self.log_lam = log_lam

    def PY(self, t, xp, x):
        v = x[...,0]
        return dists.Normal(loc=self.mu+self.beta*v, scale=np.sqrt(v)+1e-5)

    @staticmethod
    def get_prior():
        raise NotImplementedError
