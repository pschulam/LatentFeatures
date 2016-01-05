import numpy as np
import scipy.linalg as la

from scipy.stats import multivariate_normal as mvn


_LOG_2PI = np.log(2 * np.pi)


def learn_ppca(data, hid_dim):
    '''Learn the parameters of probabilistic PCA.

    returns : mean mu, loading matrix W, and variance sigsq.

    '''
    dataset = Dataset(data)    
    m = dataset.empirical_mean()
    S = dataset.scatter_matrix(center=m)
    
    U, s = la.svd(S, full_matrices=False)[:2]
    var = np.mean(s[hid_dim:])
    loading = U[:, :hid_dim] * np.sqrt(s[:hid_dim] - var)

    ppca = ProbabilisticPCA(dataset.obs_dim, hid_dim)    
    ppca.update_mean(m)
    ppca.update_loading(loading)
    ppca.update_var(var)

    return ppca


def observed_loglik(dataset, ppca):
    '''The observed-data log likelihood of the data.

    '''
    N = len(dataset)
    D = ppca.obs_dim
    C = ppca.marginal_cov()
    P = ppca.marginal_prec()
    S = dataset.scatter_matrix(center=ppca.marginal_mean())
    return -N / 2 * (D * _LOG_2PI + np.log(la.det(C)) + trace(P * S))


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


class ProbabilisticPCA:
    def __init__(self, obs_dim, hid_dim):
        self.mean    = np.zeros(obs_dim)
        self.loading = np.zeros((obs_dim, hid_dim))
        self.var     = 1.0
        self.obs_dim, self.hid_dim = self.loading.shape

    def marginal_mean(self):
        '''The mean of the marginal distribution.

        '''
        return self.mean.copy()

    def marginal_cov(self):
        '''The covariance matrix of the marginal distribution.

        '''
        WWt = np.dot(self.loading, self.loading.T)
        return WWt + self.var * np.eye(len(WWt))

    def marginal_prec(self):
        '''The precision matrix of the marginal distribution.

        '''
        prec = np.eye(self.obs_dim) / np.sqrt(self.var)

        W = self.loading
        M = np.dot(W.T, W) + self.var * np.eye(self.hid_dim)
        prec += np.dot(W, la.solve(M, W.T)) / self.var

        return prec

    def update_mean(self, mean):
        self.mean[:] = mean

    def update_loading(self, loading):
        self.loading[:, :] = loading

    def update_var(self, var):
        self.var = var


class Dataset:
    def __init__(self, data):
        self.data = np.array(data)
        self.num_obs, self.obs_dim = self.data.shape
        self.mean, self.cov = Dataset._empirical_stats(self.data)

    def __len__(self):
        return self.num_obs

    def empirical_mean(self):
        return self.mean

    def empirical_cov(self):
        return self.cov

    def scatter_matrix(self, center):
        centered = self.data - center
        return np.dot(centered.T, centered) / len(self.data)

    @staticmethod
    def _empirical_stats(data):
        mean = data.mean(axis=0)
        centered_data = data - mean
        cov = np.dot(centered_data.T, centered_data) / len(data)
        return mean, cov


def trace(A):
    return np.diag(A).sum()


if __name__ == '__main__':
    num_obs = 10000
    obs_dim = 6
    hid_dim = 3
    
    W, Z, X = simulate_data(num_obs, obs_dim, hid_dim)
    dataset = Dataset(X)
    model = ProbabilisticPCA(obs_dim, hid_dim)
    model.update_mean(dataset.empirical_mean())

    print('LL={:.8f}'.format(observed_loglik(dataset, model)))

    learned = learn_ppca(X, hid_dim)

    print('LL={:.8f}'.format(observed_loglik(dataset, learned)))
    print(learned.mean)
    print(learned.loading)
    print(learned.var)
    print(W)
