import numpy as np
import particles
from particles import state_space_models as ssm
import particles.distributions as dists

tol = 1e-16

class IIDIncr(ssm.StateSpaceModel):
    def __init__(self, mu=0.0, beta=0.0):
        super(IIDIncr, self).__init__()
        self.mu = mu
        self.beta = beta

    def PY(self, t, xp, x):
        x = np.maximum(x, tol)
        return dists.Normal(loc=self.mu+self.beta*x, scale=np.sqrt(x))

    @staticmethod
    def get_prior():
        raise NotImplementedError
