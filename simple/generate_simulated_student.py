import numpy as np
import pandas as pd
import os
import pickle

from student_iid import StudentIIDIncr

from utils import logit

# ------------------------------------------------------------------
# This script simulates data from the Student IID model.
# ------------------------------------------------------------------

# Parameters of the simulation

# Number of observations
T = 1500
T_test = 50000
# Parameters of the IID student model
params = {
    "nu": 5.,
}

# Directory where to save the generated data
save_dir = "../data/simulated/"

# Simulate y from model
theta = {
            'log_nu':np.log(params['nu'])
}

model = StudentIIDIncr(**theta)

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
filename = ('student_'
            'T_{}_'
            'nu_{}_'
            '_train.pkl'
            .format(T, params['nu']))

filename_test = ('student_'
            'T_{}_'
            'nu_{}_'
            '_test.pkl'
            .format(T, params['nu']))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

with open(save_dir+filename, 'wb') as f:
    pickle.dump(data, f)

with open(save_dir+filename_test, 'wb') as f:
    pickle.dump(data_test, f)
