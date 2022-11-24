import numpy as np
import pandas as pd
import os
import pickle

from ghyperbolic_iid import GHDIIDIncr

from utils import logit

# ------------------------------------------------------------------
# This script simulates data from the Generalized Hyperbolic IID model.
# ------------------------------------------------------------------

# Parameters of the simulation

# Number of observations
T = 1500
T_test = 50000
# Parameters of the IID Generalized hyperbolic distribution model
params = {
    "lam": -1.,
    "alpha": 0.5,
    "delta": 0.75
}

# Directory where to save the generated data
save_dir = "../data/simulated/"

# Simulate y from model
theta = {
            'lam': params['lam'],
            'log_alpha': np.log(params['alpha']),
            'log_delta': np.log(params['alpha'])
}

model = GHDIIDIncr(**theta)

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
filename = ('ghd_'
            'T_{}_'
            'lam_{}_'
            'alpha_{}_'
            'delta_{}_'
            '_train.pkl'
            .format(T, params['lam'], params['alpha'], params['delta']))

filename_test = ('ghd_'
            'T_{}_'
            'lam_{}_'
            'alpha_{}_'
            'delta_{}_'
            '_test.pkl'
            .format(T, params['lam'], params['alpha'], params['delta']))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

with open(save_dir+filename, 'wb') as f:
    pickle.dump(data, f)

with open(save_dir+filename_test, 'wb') as f:
    pickle.dump(data_test, f)