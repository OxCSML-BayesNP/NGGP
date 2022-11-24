import sys
sys.path.append('../')

import numpy as np
import numpy.random as npr
import numba as nb
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
import json
import os
import pickle

from mcmc import PMMH
import particles
import particles.distributions as dists

from gamma_driven_sv import GammaDrivenSV
from gbfry_driven_sv import GBFRYDrivenSV

parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--model', type=str, default='gbfry',
        choices=['gbfry', 'gamma'])
parser.add_argument('--norm', action='store_false', default=True)

# for PMMH
parser.add_argument('--save_states', action='store_false')
parser.add_argument('--Nx', type=int, default=1000)
parser.add_argument('--burnin', type=int, default=None)
parser.add_argument('--niter', type=int, default=5000)
parser.add_argument('--verbose', type=int, default=100)

# for saving
parser.add_argument('--run_name', type=str, default='trial')

args = parser.parse_args()

if args.filename is None:
    raise ValueError('You must specify data')
else:
    data = os.path.splitext(os.path.basename(args.filename))[0]

save_dir = os.path.join('results', args.model, data, args.run_name)

with open(os.path.join(args.filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')

if args.model == 'gamma':
    ssm_cls = GammaDrivenSV
elif args.model == 'gbfry':
    ssm_cls = GBFRYDrivenSV
else:
    raise NotImplementedError
prior = ssm_cls.get_prior()

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.norm:
    y = datafile['y'] / np.std(datafile['y'])
else:
    print('here')
    y = datafile['y']

pmmh = PMMH(ssm_cls=ssm_cls, data=y,
        prior=prior, theta0=ssm_cls.get_theta0(y),
        Nx=args.Nx, niter=args.niter, keep_states=args.save_states,
        verbose=args.niter/args.verbose)
pmmh.run()

burnin = args.burnin or args.niter // 2

if args.save_states:
    x = np.stack(pmmh.states[burnin:], 0)[:, :, 0]
    with open(os.path.join(save_dir, 'states.pkl'), 'wb') as f:
        pickle.dump(x, f)

with open(os.path.join(save_dir, 'chain.pkl'), 'wb') as f:
    pickle.dump(pmmh.chain[burnin:], f)
