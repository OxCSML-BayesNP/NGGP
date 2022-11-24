import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
import os
import pickle
from metrics import KS, bayes_L1_loss, bayes_L2_loss, uniform_KS

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--run_names', type=str, nargs='+', default=['chain1', 'chain2', 'chain3'])
parser.add_argument('--show', action='store_true')

args = parser.parse_args()

if args.filename is None:
    raise ValueError('You must specify data')
else:
    data = os.path.splitext(os.path.basename(args.filename))[0]

with open(os.path.join(args.filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')
if datafile.get('x') is not None:
    std = np.std(datafile['y'])
    true_x = datafile['x'] / np.std(datafile['y'])**2
else:
    raise ValueError('True states must be given')

if not os.path.isdir('metrics'):
    os.makedirs('metrics')

alpha_list = [.5, .6, .7, .8, .9, .95, .99]

with open('metrics/{}.txt'.format(data), 'w') as logf:
    for model in ['gbfry', 'gamma']:
        save_dir = os.path.join('results', model, data)
        states = []
        for run_name in args.run_names:
            with open(os.path.join(save_dir, run_name, 'states.pkl'), 'rb') as f:
                states.append(pickle.load(f))
                x = np.concatenate(states, 0)

        print(model)
        logf.write(model + '\n')

        line = 'Bayes L2 loss: {:.4f}'.format(bayes_L2_loss(x, true_x))
        print(line)
        logf.write(line + '\n')

        line = 'KS(pi, U(0, 1): {:.4f}'.format(uniform_KS(x, true_x))
        print(line)
        logf.write(line + '\n')

        line = 'reweighted KS(pi, U(0, 1)): {:.4f}'.format(uniform_KS(x, true_x, reweighted=True))
        print(line)
        logf.write(line + '\n')

        losses = []
        for alpha in alpha_list:
            loss = bayes_L1_loss(x, true_x, alpha=alpha)
            losses.append(loss)
            line = 'Bayes L1 loss (alpha={:.2f}): {:.4f}'.format(alpha, loss)
            print(line)
            logf.write(line + '\n')

        plt.plot(alpha_list, losses, label=model)

        print()
        logf.write('\n')

    plt.legend()
    plt.savefig('metrics/{}_L1.pdf'.format(data), bbox_inches='tight')
    plt.show()
