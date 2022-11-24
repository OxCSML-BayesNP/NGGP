import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import argparse
import json
import os
import pickle

from mcmc import PMMH
from particles.mcmc import BasicRWHM

from nig_iid import NIGIID
from gamma_iid_incr import GammaIIDIncr
from ns_iid_incr import NSIIDIncr
from gbfry_iid_incr import GBFRYIIDIncr
from ghyperbolic_iid import GHDIIDIncr
from student_iid import StudentIIDIncr
from vgamma3_iid import VGamma3IID
from vgamma4_iid import VGamma4IID

parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--model', type=str, default='gbfry', choices=['gbfry', 'gamma', 'nig', 'ns',
                                                                   'ghd', 'student', 'vgamma3', 'vgamma4'])

# for PMMH
parser.add_argument('--save_states', action='store_true')
parser.add_argument('--Nx', type=int, default=500)
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

# Set to false for the models using particles SMC
mh_flag = False

# Remove the hour slots where the volume is abnormal
datafile = datafile[datafile.Volume >= 200]

y = datafile['y']
y = y / np.std(y) #W

ssm_options = {}

if args.model == 'gamma':
    ssm_cls = GammaIIDIncr
elif args.model == 'gbfry':
    ssm_cls = GBFRYIIDIncr
elif args.model == 'ns':
    ssm_cls = NSIIDIncr
elif args.model == 'vgamma3':
    mh_flag = True
    ssm_cls = VGamma3IID
elif args.model == 'vgamma4':
    mh_flag = True
    ssm_cls = VGamma4IID
elif args.model == 'nig':
    ssm_cls = NIGIID
elif args.model == 'student':
    ssm_cls = StudentIIDIncr
elif args.model == 'ghd':
    ssm_cls = GHDIIDIncr
else:
    raise NotImplementedError

prior = ssm_cls.get_prior()
theta0 = ssm_cls.get_theta0(y)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if mh_flag:
    model = ssm_cls(data=y, prior=prior)
    pmmh = BasicRWHM(niter=args.niter, verbose=args.niter/args.verbose,
                     theta0=theta0, model=model)
else:
    pmmh = PMMH(ssm_cls=ssm_cls, data=y,
            prior=prior, theta0=ssm_cls.get_theta0(y),
            Nx=args.Nx, niter=args.niter, keep_states=args.save_states,
            ssm_options=ssm_options, verbose=args.niter/args.verbose)
pmmh.run()

burnin = args.burnin or args.niter // 2

if args.save_states:
    x = np.stack(pmmh.states[burnin:], 0)
    print("Saving states")
    with open(os.path.join(save_dir, 'states.pkl'), 'wb') as f:
        pickle.dump(x, f)

print("Saving chains")
with open(os.path.join(save_dir, 'chain.pkl'), 'wb') as f:
    pickle.dump(pmmh.chain[burnin:], f)
