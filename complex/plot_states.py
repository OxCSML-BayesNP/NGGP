import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
import os
import pickle
from scipy.stats.mstats import mquantiles
from metrics import KS, bayes_L1_loss, bayes_L2_loss, uniform_KS

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--run_names', type=str, nargs='+', default=['chain1', 'chain2', 'chain3'])
parser.add_argument('--show', action='store_true')
parser.add_argument('--norm', action='store_false')

args = parser.parse_args()

if args.filename is None:
    raise ValueError('You must specify data')
else:
    data = os.path.splitext(os.path.basename(args.filename))[0]

with open(os.path.join(args.filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')
if datafile.get('x') is not None:
    if args.norm:
        std = np.std(datafile['y'])
        true_x = datafile['x'] / np.std(datafile['y'])**2
    else:
        true_x = datafile['x']
else:
    raise ValueError('True states must be given')

sb.set()
sb.set_style("whitegrid", {'axes.grid':False})
T = true_x.shape[-1]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(T), true_x, '--g', label='True', alpha=0.5)
save_dir = os.path.join('results', 'gamma', data)
states = []
for run_name in args.run_names:
    with open(os.path.join(save_dir, run_name, 'states.pkl'), 'rb') as f:
        states.append(pickle.load(f))
        x = np.concatenate(states, 0)

plt.plot(range(T), x.mean(0), label='Estimate', color='b')
qnts = mquantiles(x, [0.025, 0.975], axis=0, alphap=0.5, betap=0.5)
plt.fill_between(range(T), qnts[0], qnts[1],
        color='b', alpha=0.3, edgecolor=None)

plt.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Volatility', fontsize=20)

plt.subplot(1, 2, 2)
plt.plot(range(T), true_x, '--g', label='True', alpha=0.5)
save_dir = os.path.join('results', 'gbfry', data)
states = []
for run_name in args.run_names:
    with open(os.path.join(save_dir, run_name, 'states.pkl'), 'rb') as f:
        states.append(pickle.load(f))
        x = np.concatenate(states, 0)

plt.plot(range(T), x.mean(0), label='Estimate', color='r')
qnts = mquantiles(x, [0.025, 0.975], axis=0, alphap=0.5, betap=0.5)
plt.fill_between(range(T), qnts[0], qnts[1],
        color='r', alpha=0.3, edgecolor=None)

plt.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Volatility', fontsize=20)

plt.tight_layout()
plt.savefig('plots/{}_states.png'.format(data), bbox_inches='tight')
plt.show()
