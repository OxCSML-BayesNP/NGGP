import pandas as pd
import numpy as np
import pickle
import os

if not os.path.isdir('oxford'):
    os.makedirs('oxford')

csv = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
keys = csv.Symbol.unique()
print(keys)

T = 1000
lag = 2000

for key in keys:
    with open(os.path.join('oxford', '{}.pkl'.format(key[1:])), 'wb') as f:
        rows = csv.loc[csv.Symbol==key]
        data = {}
        data['y'] = np.log(np.array(rows.open_to_close)[-(T+lag):-lag] + 1)
        data['x'] = np.array(rows.rv5_ss)[-(T+lag):-lag]
        print(len(data['x']))
        pickle.dump(data, f)
