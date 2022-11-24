import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
import os
import pickle
from scipy.stats.mstats import mquantiles

from gbfry_driven_sv import GBFRYDrivenSV
from gamma_driven_sv import GammaDrivenSV

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--model', type=str, default='gbfry', choices=['gamma', 'gbfry'])
parser.add_argument('--run_names', type=str, nargs='+', default=['chain1', 'chain2', 'chain3'])
parser.add_argument('--show', action='store_true', default=False)
parser.add_argument('--norm', action='store_false', default=True)

args = parser.parse_args()

if args.model == 'gamma':
    ssm_cls = GammaDrivenSV
elif args.model == 'gbfry':
    ssm_cls = GBFRYDrivenSV
else:
    raise NotImplementedError
prior = ssm_cls.get_prior()

if args.filename is None:
    raise ValueError('You must specify data')
else:
    data = os.path.splitext(os.path.basename(args.filename))[0]

save_dir = os.path.join('results', args.model, data)
prefix = '{}_{}'.format(data, args.model)

fig_dir = os.path.join('plots', data, args.model)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

keys = prior.laws.keys()
chains = []
sb.set()
sb.set_style("whitegrid", {'axes.grid':False})

for run_name in args.run_names:
    with open(os.path.join(save_dir, run_name, 'chain.pkl'), 'rb') as f:
        chains.append(pickle.load(f))

with open(os.path.join(args.filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')

for i, key in enumerate(keys):
    gathered = []
    param_latex = r"${}$".format(ssm_cls.params_latex[key])
    if datafile.get('true_params') is not None:
        true_val = ssm_cls.params_transform[key](datafile['true_params'][key])
    else:
        true_val = None
    plt.figure(figsize=(10,5))
    plt.subplot(122)
    for j, chain in enumerate(chains):
        val = ssm_cls.params_transform[key](chain.theta[key])
        gathered.append(np.array(val))
        plt.plot(range(2500, 5000), val, linewidth=0.7, label='Chain {}'.format(j+1))
        plt.xlim([2500, 5000])
    if true_val is not None:
        plt.axhline(true_val, color='k', linestyle='--', linewidth=1.2)
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel(param_latex, fontsize=15)
    plt.legend(fontsize=15)

    plt.subplot(121)
    val = np.concatenate(gathered)
    sb.distplot(val, 20, color='r', hist=True)
    if true_val is not None:
        plt.axvline(true_val, color='k', linestyle='--', linewidth=1.2)
    plt.xlabel(param_latex, fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '{}_samples.png'.format(ssm_cls.params_name[key])),
            bbox_inches='tight')

    qnts = mquantiles(val, [0.025, 0.975])
    print(f'{key} \qnts{{{val.mean():.2f}}}{{{qnts[0]:.2f}}}{{{qnts[1]:.2f}}}')

states = []
for run_name in args.run_names:
    with open(os.path.join(save_dir, run_name, 'states.pkl'), 'rb') as f:
        states.append(pickle.load(f))
x = np.concatenate(states, 0)

if datafile.get('x') is not None:
    if args.norm:
        std = np.std(datafile['y'])
        true_x = datafile['x'] / np.std(datafile['y'])**2
    else:
        true_x = datafile['x']
else:
    true_x = None

plt.figure('{}_states'.format(prefix))
T = x.shape[-1]
qnts = mquantiles(x, [0.025, 0.975], axis=0, alphap=0.5, betap=0.5)
plt.plot(range(T), x.mean(0), '-r', label='Estimate')
plt.fill_between(range(T), qnts[0], qnts[1], alpha=0.3, color='r')
if true_x is not None:
    plt.plot(range(T), true_x, '--g', label='True')
plt.xlabel('Time', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Volatility', fontsize=20)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.savefig(os.path.join(fig_dir, 'states.png'), bbox_inches='tight')

if args.show:
    plt.show()
