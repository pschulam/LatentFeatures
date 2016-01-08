import numpy as np
import scipy.linalg as la

from scipy.stats import multivariate_normal as mvn


class ProbabilisticPCA:
    def __init__(self, obs_dim, hid_dim, seed=0):
        rand = np.random.RandomState(seed)
        self.loading = rand.normal(scale=0.1, size=(obs_dim, hid_dim))
        self.variance = 1.0

    @property
    def obs_dim(self):
        return self.loading.shape[0]

    @property
    def hid_dim(self):
        return self.loading.shape[1]

    def marginal_param(self):
        d = self.obs_dim
        m = np.zeros(d)
        S = np.dot(self.loading, self.loading.T) + self.variance * np.eye(d)
        return m, S

    def log_likelihood(self, x):
        return mvn.logpdf(x, *self.marginal_param())

    def posterior(self, x):
        M = np.dot(self.loading.T, self.loading) + self.variance * np.eye(self.hid_dim)
        m = la.solve(M, np.dot(self.loading.T, x))        
        S = self.variance * la.inv(M)
        return m, S


def learn_ppca(X, K, maxiter=100, tol=1e-4):
    '''Learn the parameters of probabilistic PCA.

    '''
    m = np.mean(X, axis=0)
    X_centered = X - m

    ppca = ProbabilisticPCA(X_centered.shape[1], K)

    logl = sum(ppca.log_likelihood(x) for x in X_centered)

    for iteration in range(maxiter):
        ppca.loading, ppca.variance = em_step(X_centered, ppca)

        ll_old = logl
        logl = sum(ppca.log_likelihood(x) for x in X_centered)
        delta = (logl - ll_old) / np.abs(ll_old)
        
        print('iter={:04d}, LL={:8f}, dLL={:.8f}'.format(iteration, logl, delta))
        
        if delta < tol:
            break

    return m, ppca


def em_step(X, ppca):
    ss1, ss2, ss3 = expectation_step(X, ppca)
    loading = la.solve(ss2.T, ss1.T).T
    variance = ss3 / len(X) / ppca.obs_dim
    return loading, variance


def expectation_step(X, ppca):
    W = ppca.loading
    WtW = np.dot(W.T, W)
    
    ss1 = 0.0
    ss2 = 0.0
    ss3 = 0.0

    for x in X:
        m, S = ppca.posterior(x)
        ss1 += np.outer(x, m)
        ss2 += np.outer(m, m) + S
        ss3 += la.norm(x - np.dot(W, m))**2 + np.diag(S * WtW).sum()

    return ss1, ss2, ss3


def simulate_data(num_obs, obs_dim, hid_dim, seed=0):
    '''Generate a dataset to use as an example.

    '''
    rand = np.random.RandomState(seed)

    subspace = rand.multivariate_normal(np.zeros(hid_dim), np.eye(hid_dim), size=obs_dim)
    colvecs, singvals = la.svd(subspace, full_matrices=False)[:2]
    loading = colvecs * singvals
    
    latent = rand.multivariate_normal(np.zeros(hid_dim), np.eye(hid_dim), size=num_obs)
    observed = np.dot(latent, loading.T) + rand.normal(size=(num_obs, obs_dim))
    
    return loading, latent, observed


if __name__ == '__main__':
    num_samples = 1000
    obs_dim, hid_dim = 4, 2
    W, Z, X = simulate_data(num_samples, obs_dim, hid_dim)
    center, ppca = learn_ppca(X, hid_dim, tol=1e-8)

    print()
    print('True subspace:')
    print('--------------')
    print(la.svd(W, full_matrices=False)[0])
    print()
    print('Learned subspace:')
    print('-----------------')
    print(la.svd(ppca.loading, full_matrices=False)[0])
