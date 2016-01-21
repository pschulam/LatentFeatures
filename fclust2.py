'''Fit clusters to sparse functional data.

'''

import logging
import numpy as np
import scipy.linalg as la

from collections import namedtuple

from common import log_normalize


# Constants used to initialize parameters.
_IPROB_ALPHA = 10.0
_ICOEF_SCALE = 0.1


FunctionalMixture = namedtuple(
    'FunctionalMixture',
    [
        'obs_dim',
        'hid_dim',
        'num_clust',
        'log_prob',
        'center',
        'loading',
        'coef',
        'ranef_cov',
        'noise_var'
    ]
)






def fit(model, dataset, maxiter=500, tol=1e-5):
    '''Fit the functional cluster model using a dataset.

    '''
    total_iter = 0
    convergence = float('inf')

    while convergence > tol:
        logging.info('Starting iteration %d', total_iter + 1)
        total_iter += 1
        if total_iter >= maxiter:
            break


# Functions for estimation from the dataset posteriors.

def estimate_log_prob(memb_posteriors):
    '''Compute new estimates of cluster log probabilities.

    '''
    totals = sum(expected_membership(p) for p in memb_posteriors)
    return np.log(np.asarray(totals) / np.sum(totals))


def estimate_ranef_covar(memb_posteriors, ranef_posteriors):
    '''Compute a new estimate of the random effects covariance.

    '''
    scatter = 0.0
    for memb, ranef in zip(memb_posteriors, ranef_posteriors):
        for p_clust, p_ranef in zip(memb, ranef):
            scatter += p_clust * expected_ranef_outer(p_ranef)

    return scatter / len(posteriors)


# Functions for computing expectations from posteriors.

def expected_membership(posterior):
    '''Compute the expectation of belonging to each cluster.

    Parameters
    ----------
    posterior : 1D array
    Log probabilities of cluster membership.

    '''
    return np.exp(posterior)


def expected_ranef_outer(posterior):
    '''Compute the expectation of the random effect's outer product.

    Parameters
    ----------
    posterior : 2-tuple
    Mean and covar. of a mult. normal.

    '''
    mean, covar = posterior
    return covar + np.outer(mean, mean)


def num_clusters(model):
    '''Get the number of clusters in the mixture.

    '''
    return len(model.prob)


def cluster_marg_mean(model, clust, bases):
    '''Compute the marginal mean for a curve using a cluster.

    '''
    return bases @ (model.center + model.loading @ model.coef[clust])


def cluster_marg_covar(model, clust, bases):
    '''Compute the marginal covariance for a curve using a cluster.

    '''
    return bases @ model.ranef_covar @ bases.transpose()


def membership_posterior(model, curve):
    '''Compute the posterior over cluster membership.

    '''
    num_clust = num_clusters(model)
    scores = np.zeros(num_clust)
    for clust in range(num_clust):
        resid = curve.values - cluster_marg_mean(model, clust, curve)
        covar = cluster_marg_covar(model, clust, curve)
        scores[clust] -= model.log_prob[clust]
        scores[clust] -= 0.5 * resid @ la.solve(covar, resid)

    return log_normalize(scores)


def ranef_posteriors(model, curve):
    '''Compute the posterior over random effects for all clusters.

    '''
    clusters = range(num_clusters(model))
    return [cluster_ranef_posterior(model, c, curve) for c in clusters]


def cluster_ranef_posterior(model, clust, curve):
    '''Compute the Gaussian posterior over random effects.

    '''
    resid = curve.values - cluster_marg_mean(model, clust, curve.bases)
    prec = la.inv(model.ranef_covar)
    prec += curve.bases.transpose() @ curve.bases / model.noise_var
    mean = la.solve(prec, curve.bases.transpose() @ resid)
    return mean, la.inv(prec)


# Model creation, initialization, and storage.

def new_model(obs_dim, hid_dim, num_clust):
    '''Allocate a new set of model parameters.

    '''
    return FunctionalMixture(
        obs_dim=obs_dim,
        hid_dim=hid_dim,
        num_clust=num_clust,
        log_prob=np.zeros(num_clust),
        center=np.zeros(obs_dim),
        loading=np.zeros((obs_dim, hid_dim)),
        coef=np.zeros((hid_dim, num_clust)),
        ranef_cov=_ICOEF_SCALE * np.eye(obs_dim),
        noise_var=_ICOEF_SCALE)


def init_model_randomly(model, seed=0):
    '''Randomly initialize the model parameters.

    '''
    rand = np.random.RandomState(seed)
    obs_dim = model.obs_dim
    hid_dim = model.hid_dim
    num_clust = model.num_clust
    model.log_prob = rand.dirichlet(num_clusters(model) * [_IPROB_ALPHA])
    model.center = rand.normal(scale=_ICOEF_SCALE, size=obs_dim)
    model.loading = rand.normal(scale=_ICOEF_SCALE, size=(obs_dim, hid_dim))
    model.coef = rand.normal(scale=_ICOEF_SCALE, size=(hid_dim, num_clust))

    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    fit(None, None)
