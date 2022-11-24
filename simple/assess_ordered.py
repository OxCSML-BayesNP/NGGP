import sys



sys.path.append('../')
from metrics import KS


import argparse
import os
import pickle
import json

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

from gbfry_iid_incr import GBFRYIIDIncr
from gamma_iid_incr import GammaIIDIncr
from nig_iid import NIGIID
from ns_iid_incr import NSIIDIncr
from ghyperbolic_iid import GHDIIDIncr
from student_iid import StudentIIDIncr
from vgamma3_iid import VGamma3IID
from vgamma4_iid import VGamma4IID

from utils import VaR, ecdf



parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--run_names', type=str, nargs='+', default=['chain1', 'chain2', 'chain3'])
parser.add_argument('--model', type=str, default='gbfry')
parser.add_argument('--eta_name', type=str, default='log_eta')
parser.add_argument('--burnin', type=int, default=0)
parser.add_argument('--thin', type=int, default=10)
parser.add_argument('--niter', type=int, default=2500)
parser.add_argument('--credible_mass', type=int, default=95)

args = parser.parse_args()

# Set to false for the models using particles SMC
mh_flag = False

# Set font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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

if args.filename is None:
    raise ValueError('You must specify data')
else:
    data = os.path.splitext(os.path.basename(args.filename))[0]

save_dir = os.path.join('results', args.model, data)
fig_dir = os.path.join('plots', data, args.model)
print(fig_dir)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

prefix = '{}_{}'.format(args.model, data)

keys = prior.laws.keys()
chains = []


with open(os.path.join(args.filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')
    datafile = datafile[datafile.Volume >= 200]
    std_true = np.std(datafile['y'])
    train_y = datafile['y'].values
    train_y = train_y / std_true #W

test_filename = args.filename[:args.filename.rfind('train.pkl')] + 'test.pkl'
#test_filename = args.filename[:args.filename.rfind('.pkl')] + '_test.pkl' #W
with open(os.path.join(test_filename), 'rb') as f:
    datafile = pickle.load(f, encoding='latin1')
    datafile = datafile[datafile.Volume >= 200]
    ssm_options = {}

true_y = datafile['y'].values
true_y = true_y / std_true #W

true_var = VaR(true_y)

T = len(true_y)

print('VaR of observed data: {}'.format(true_var))

for run_name in args.run_names:
    with open(os.path.join(save_dir, run_name, 'chain.pkl'), 'rb') as f:
        chains.append(pickle.load(f))


n_samples = len(chains)*(int((args.niter-args.burnin)/args.thin))
pred_y_matrix = np.zeros((n_samples, T))

n_var = 30
var_values = np.geomspace(2e-5, .5, n_var)
pred_var = np.zeros((n_samples, n_var))

n_var_lin = 500
lin_var_values = np.linspace(2e-5, 1.-2e-5, n_var_lin, endpoint=True)
lin_pred_var = np.zeros((n_samples, n_var_lin))

n_bins = 40
s = 0
for chain in chains:
    for j in tqdm(range(args.burnin, args.niter, args.thin)):
        theta = {}
        for key in keys:
            theta[key] = chain.theta[key][j]
        if mh_flag:
            y_pred = ssm_cls.simulate(**theta, T=T)
            pred_y_matrix[s, :] = y_pred
        else:
            x_pred, y_pred = ssm_cls(**theta, **ssm_options).simulate(T)
            pred_y_matrix[s, :] = [y_pred_i[0] for y_pred_i in y_pred]
        pred_var[s, :] = VaR(y_pred, var_values)
        lin_pred_var[s, :] = VaR(y_pred, lin_var_values)
        s += 1

assert s==n_samples


with open(os.path.join(fig_dir, "ks_metrics.txt"), 'w') as logf:
    # Print KS
    line = 'Test KS(true_y, pred_y): {:.4f}'.format(KS(pred_y_matrix.flatten(), true_y))
    print(line)
    logf.write(line + '\n')


    line = 'Train KS(true_y, pred_y): {:.4f}'.format(KS(pred_y_matrix.flatten(), train_y))
    print(line)
    logf.write(line + '\n')

# VaR log
low_per = (100-args.credible_mass)/2
high_per = 100-low_per
var_true = VaR(true_y, var_values)
plt.figure('Predicted VaR')
#plt.title("Predictive VaR")
plt.xlabel("1-$\\alpha$")
plt.ylabel("Value at risk")
plt.semilogx(var_values, var_true, linewidth=1.5)
plt.fill_between(var_values, var_cred[0, :], var_cred[1, :], color='b', alpha=.2,
                 label="{}% credible region".format(args.credible_mass))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Predicted_VaR_test.png"), bbox_inches='tight')
plt.close('all')

# Plot ordered y^2
low_per = (100-args.credible_mass)/2
high_per = 100-low_per
plt_range = int(95/100*T)
y_2_matrix = np.sort(pred_y_matrix**2, axis=1)[:, ::-1]
cred_region = np.percentile(y_2_matrix, [low_per, high_per], axis=0)
plt.figure('Ordered y^2')
#plt.title("Posterior predictive of ordered y^2")
plt.xlabel("Rank")
plt.ylabel("y^2")
plt.loglog(range(plt_range), np.sort(true_y**2)[::-1][:plt_range], linewidth=1.5)
plt.fill_between(range(plt_range), cred_region[0, :plt_range], cred_region[1, :plt_range], color='b', alpha=.2,
                 label="{}% credible region".format(args.credible_mass))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Posterior_predictive_ordered_y_square_test.png"), bbox_inches='tight')
plt.close('all')


# Plot log F(x) no zeros with skip of Y^2
low_per = (100-args.credible_mass)/2
high_per = 100-low_per
skip = 500
x_array = np.sort(np.square(true_y[true_y != 0]))
p_0 = np.sum(true_y == 0)/len(true_y)
len_x = len(x_array)
prob_matrix = -np.log(np.array([ecdf(np.square(abs_y), x_array) for abs_y in abs_y_matrix]))
cred_region = np.percentile(prob_matrix, [low_per, high_per], axis=0)
plt.figure('cdf_tail')
#plt.title("Posterior predictive of ordered abs(y)")
plt.ylabel("-log(P(Y^2 < x))")
plt.xlabel("x")
plt.ylim(-np.log(1-1/len_x), np.minimum(-np.log(p_0/10), 5))
empirical_cdf = -np.log(p_0 + (1-p_0)*np.arange(1, len_x+1)/len_x)
plt.loglog(x_array[skip:], empirical_cdf[skip:], linewidth=1.5, label="Empirical cdf")
plt.fill_between(x_array[skip:], cred_region[0, skip:],  cred_region[1, skip:], color='b', alpha=.2,
                 label="{}% credible region".format(args.credible_mass))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Posterior_cdf_square_y_no_zero_skip.png"), bbox_inches='tight')
plt.close('all')

# Plot log F(x) no zeros no skip of Y^2
low_per = (100-args.credible_mass)/2
high_per = 100-low_per
skip = 0
x_array = np.sort(np.square(true_y[true_y != 0]))
p_0 = np.sum(true_y == 0)/len(true_y)
len_x = len(x_array)
prob_matrix = -np.log(np.array([ecdf(np.square(abs_y), x_array) for abs_y in abs_y_matrix]))
cred_region = np.percentile(prob_matrix, [low_per, high_per], axis=0)
plt.figure('cdf_tail')
#plt.title("Posterior predictive of ordered abs(y)")
plt.ylabel("-log(P(Y^2 < x))")
plt.xlabel("x")
plt.ylim(-np.log(1-1/len_x), np.minimum(-np.log(p_0/10), 5))
empirical_cdf = -np.log(p_0 + (1-p_0)*np.arange(1, len_x+1)/len_x)
plt.loglog(x_array[skip:], empirical_cdf[skip:], linewidth=1.5, label="Empirical cdf")
plt.fill_between(x_array[skip:], cred_region[0, skip:],  cred_region[1, skip:], color='b', alpha=.2,
                 label="{}% credible region".format(args.credible_mass))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Posterior_cdf_square_y_no_zero_no_skip.png"), bbox_inches='tight')
plt.close('all')


# Plot abs(y) log log
n_bins = 100
plt.figure('Histogram of posterior predictive of abs(y) (log-log scale)')
bins_range = (0.08, 40.)
logbins = np.logspace(np.log(bins_range[0]),np.log(bins_range[1]), n_bins)
plt.xscale('log')
plt.hist(np.abs(pred_y_matrix.flatten()), bins=logbins, log=True, density=1, alpha=0.5, label="Posterior predictive")
plt.hist(np.abs(true_y), bins=logbins, log=True, density=1, alpha=0.5, label="True observations")
plt.title("Posterior predictive of abs(y)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Histogram_posterior_predictive_abs_y.png"), bbox_inches='tight')
