import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
import os
import pickle
from scipy.stats.mstats import mquantiles

from gamma_iid_incr import GammaIIDIncr
from gbfry_iid_incr import GBFRYIIDIncr
from nig_iid import NIGIID
from ns_iid_incr import NSIIDIncr
from ghyperbolic_iid import GHDIIDIncr
from student_iid import StudentIIDIncr
from vgamma3_iid import VGamma3IID
from vgamma4_iid import VGamma4IID

from utils import logit

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--model', type=str, default='gbfry')
parser.add_argument('--run_names', type=str, nargs='+', default=['chain1', 'chain2', 'chain3'])
parser.add_argument('--show', action='store_true')
parser.add_argument('--no_states', action='store_true')

args = parser.parse_args()

if args.model == 'gamma':
    ssm_cls = GammaIIDIncr
elif args.model == 'gbfry':
    ssm_cls = GBFRYIIDIncr
elif args.model == 'ns':
    ssm_cls = NSIIDIncr
elif args.model == 'vgamma3':
    ssm_cls = VGamma3IID
elif args.model == 'vgamma4':
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

# Set font sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

for run_name in args.run_names:
    with open(os.path.join(save_dir, run_name, 'chain.pkl'), 'rb') as f:
        chains.append(pickle.load(f))

gathered = {}
for key in keys:
    params = r"${}$".format(ssm_cls.params_latex[key])
    gathered[params] = []
    plt.figure('{}_{}_trace'.format(prefix, ssm_cls.params_name[key]))
    plt.ylabel(params)
    plt.xlabel("Iteration")
    #plt.title(params)
    c_id = 1
    for chain in chains:
        val = ssm_cls.params_transform[key](chain.theta[key])
        gathered[params].append(np.array(val))
        plt.plot(val, linewidth=0.7, label='Chain {}'.format(c_id))
        c_id += 1
    plt.legend()
    plt.savefig(os.path.join(fig_dir, '{}_trace_plot.png'.format(ssm_cls.params_name[key])),
            bbox_inches='tight')


    plt.figure('{}_{}_hist'.format(prefix, ssm_cls.params_name[key]))
    plt.ylabel("Frequency")
    plt.xlabel(params)
    #plt.title(params)
    val = np.concatenate(gathered[params])
    sb.distplot(val, 20, color='r')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, '{}_histogram.png'.format(ssm_cls.params_name[key])),
            bbox_inches='tight')


plt.figure('{}_posteriors'.format(prefix))
for i, params in enumerate(gathered.keys()):
    plt.subplot(2, np.ceil(.5*len(keys)), i+1)
    plt.title(params)
    val = np.concatenate(gathered[params])
    sb.distplot(val, 35, color='r')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'posteriors.png'),
        bbox_inches='tight')

if not args.no_states:
    states = []
    for run_name in args.run_names:
        with open(os.path.join(save_dir, run_name, 'states.pkl'), 'rb') as f:
            states.append(pickle.load(f))
    x = np.concatenate(states, 0)

    with open(os.path.join(args.filename), 'rb') as f:
        datafile = pickle.load(f, encoding='latin1')
    if datafile.get('x') is not None:
        std = np.std(datafile['y'])
        true_x = datafile['x'] / np.std(datafile['y'])**2
    else:
        true_x = None

    plt.figure('{}_states'.format(prefix))
    T = x.shape[-1]
    qnts = mquantiles(x, [0.025, 0.975], axis=0, alphap=0.5, betap=0.5)
    plt.plot(range(T), x.mean(0), '-r', label='estimated')
    #plt.ylim(0., 20.) # Delete
    plt.fill_between(range(T), qnts[0], qnts[1], alpha=0.3, color='r')
    if true_x is not None:
        plt.plot(range(T), true_x, '--g', label='true', linewidth=1.)
    plt.legend()
    plt.savefig(os.path.join(fig_dir, 'states.png'), bbox_inches='tight')

if args.show:
    plt.show()

# Delete
for k in keys:
    params = r"${}$".format(ssm_cls.params_latex[k])
    val_ = gathered[params]
    qnts = np.quantile(val_, [0.025, 0.975])
    m = np.mean(val_)
    print("{} mean = {}, 95\% conf interval = ({}, {})".format(params,
                                                                m,
                                                                qnts[0],
                                                                qnts[1]))
