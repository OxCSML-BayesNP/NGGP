import numpy as np
import pandas as pd
import os
import pickle

from gbfry_iid_incr import GBFRYIIDIncr

from utils import logit

# ------------------------------------------------------------------
# This script simulates data from the GBFRY IID model.
# ------------------------------------------------------------------

# Parameters of the simulation

# Number of observations
T = 5000
T_test = 50000
# Parameters of the IID GBFRY model
params = {
    "eta": 1.,
    "c": 1.,
    "tau": 3.,
    "sigma": .6
}
# Directory where to save the generated data
save_dir = "../data/simulated/"

# Simulate y from model
ssm_cls = GBFRYIIDIncr
theta = {
            'log_eta':np.log(params['eta']),
            'log_c':np.log(params['c']),
            'log_tau_minus_one':np.log(params['tau']-1.),
            'logit_sigma':logit(params['sigma'])
}

model = GBFRYIIDIncr(**theta)

# Generate train
x, y = model.simulate(T)

data = pd.DataFrame()
data['x'] = x
data['y'] = np.array(y).squeeze()
data['Volume'] = 500*np.ones_like(y)

# Generate test
x_test, y_test = model.simulate(T_test)

data_test= pd.DataFrame()
data_test['x'] = x_test
data_test['y'] = np.array(y_test).squeeze()
data_test['Volume'] = 500*np.ones_like(y_test)

# Save data
filename = ('gbfry_'
            'T_{}_'
            'eta_{}_'
            'tau_{}_'
            'sigma_{}_'
            'c_{}'
            '_train.pkl'
            .format(T, params['eta'], params['tau'], params['sigma'], params['c']))

filename_test = ('gbfry_'
            'T_{}_'
            'eta_{}_'
            'tau_{}_'
            'sigma_{}_'
            'c_{}_test'
            '.pkl'
            .format(T, params['eta'], params['tau'], params['sigma'], params['c']))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

with open(save_dir+filename, 'wb') as f:
    pickle.dump(data, f)

with open(save_dir+filename_test, 'wb') as f:
    pickle.dump(data_test, f)