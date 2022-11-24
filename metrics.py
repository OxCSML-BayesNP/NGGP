import numpy as np

def KS(x1, x2, reweighted=False):
    x1, x2 = np.sort(np.asarray(x1)), np.sort(np.asarray(x2))
    n1, n2 = len(x1), len(x2)
    x_all = np.concatenate([x1, x2])
    cdf1 = np.searchsorted(x1, x_all, side='right') / n1
    cdf2 = np.searchsorted(x2, x_all, side='right') / n2
    D = np.absolute(cdf1 - cdf2)
    if reweighted:
        indices = np.logical_and(cdf2 > 1e-4, cdf2 < 1 - 1e-4)
        cdf1 = cdf1[indices]
        cdf2 = cdf2[indices]
        D = D[indices]
        norm = np.sqrt(cdf2*(1 - cdf2))
        D = D / norm
    return np.max(D)

# x: num_samples * num_timesteps
# x_true: num_timesteps
def bayes_risk_loss(x, x_true, alpha=0.95):

    def L(x, y):
        loss = abs(x - y)
        loss[x < y] = loss[x < y] * ((1 - alpha)/alpha)
        return loss

    # compute x_hat = alpha-quantile
    x_hat = np.quantile(x, alpha, axis=0)
    loss = L(x_true, x_hat)
    return loss.mean()

def uniform_KS(x, x_true, reweighted=False):
    pi = (x >= x_true[None]).mean(0)
    quantiles, counts = np.unique(pi, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / len(pi)

    # CDF of uniform in (0, 1): identity
    uniform_cdf = quantiles

    D = abs(cumprob - uniform_cdf)
    if reweighted:
        indices = np.logical_and(cumprob > 1e-4, cumprob < 1 - 1e-4)
        D = D[indices]
        cumprob = cumprob[indices]
        uniform_cdf = uniform_cdf[indices]
        norm = np.sqrt(cumprob*(1 - cumprob))
        D = D / norm
    return np.max(D)

def l2_loss(x, x_true):
    # compute x_hat = mean(x)
    x_hat = np.mean(x, axis=0)
    loss = (x_true-x_hat)**2
    return np.mean(loss)