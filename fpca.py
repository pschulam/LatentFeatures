import numpy as np
import scipy.linalg as la
import logging

from scipy.stats import multivariate_normal as mvn


class FunctionalPCA:

    def __init__(self, obs_dim, hid_dim, seed=0):
        rand = np.random.RandomState(seed)
        self._coef_center = np.zeros(obs_dim)
        self._principal_coef = random_orth_mat(obs_dim, hid_dim, rand)
        self._pc_var = np.ones(hid_dim)
        self._noise_var = 1.0

    @property
    def obs_dim(self):
        return self._principal_coef.shape[0]

    @property
    def hid_dim(self):
        return self._principal_coef.shape[1]

    def param_copy(self):
        mu = np.array(self._coef_center)
        U = np.array(self._principal_coef)
        D = np.array(self._pc_var)
        v = float(self._noise_var)
        return mu, U, D, v

    def log_likelihood(self, y, X):
        m = np.dot(X, self._coef_center)
        S = np.dot(X, np.dot(self._ranef_cov, X.T))
        S += self._noise_var * np.eye(len(y))
        return mvn.logpdf(y, m, S)

    def posterior(self, y, X):
        Z = np.dot(X, self._principal_coef)
        r = y - np.dot(X, self._coef_center)
        P = np.diag(1.0 / self._pc_var) + np.dot(Z.T, Z) / self._noise_var
        S = la.inv(P)
        m = la.solve(P, np.dot(Z.T, r) / self._noise_var)
        return m, S

    @property
    def _ranef_cov(self):
        U = self._principal_coef
        return np.dot(U * self._pc_var, U.T)
        
    
def learn_fpca(dataset, hid_dim, maxiter=500, tol=1e-5):
    objective = lambda fpca: sum(fpca.log_likelihood(*d) for d in dataset)

    num_bases = dataset[0][1].shape[1]
    fpca = FunctionalPCA(num_bases, hid_dim)

    logl = objective(fpca)

    for iteration in range(maxiter):
        old_param = fpca.param_copy()
        
        b, U, D, v = em_step(dataset, fpca)        
        fpca._coef_center = b
        fpca._principal_coef = U
        fpca._pc_var = D
        fpca._noise_var = v

        logl_old = logl
        logl = objective(fpca)

        delta = (logl - logl_old) / np.abs(logl_old)

        msg = 'Iteration={:05d} LL={:20.8f}, dLL={:20.8f}'
        logging.info(msg.format(iteration, logl, delta))

        if delta < tol:
            break

    U, D, _ = la.svd(U * D, full_matrices=False)
    fpca._principal_coef = U
    fpca._pc_var = D

    return fpca


def em_step(dataset, fpca):
    ss1 = 0.0
    ss2 = 0.0
    ss3 = 0.0
    ss4 = 0.0
    ss5 = 0.0

    for y, X in dataset:
        m_i, S_i = fpca.posterior(y, X)

        a = np.r_[1.0, m_i][:, None]
        a_outer = np.bmat(
            [[ np.atleast_2d(1.0), m_i[None, :]             ],
             [ m_i[:, None]      , S_i + np.outer(m_i, m_i) ]])

        ss1 += np.kron(a_outer, np.dot(X.T, X))
        ss2 += np.dot(np.kron(a.T, X).T, y)

        ss3 += S_i + np.outer(m_i, m_i)

        ss4 += len(y)

    coef = la.solve(ss1, ss2)
    b = coef[:fpca.obs_dim]
    U = np.reshape(coef[fpca.obs_dim:], (fpca.hid_dim, fpca.obs_dim)).T
    D = np.diag(ss3 / len(dataset))

    for y, X in dataset:
        m_i, S_i = fpca.posterior(y, X)
        Z = np.dot(X, U)
        ss5 += np.sum((y - np.dot(X, b + np.dot(U, m_i)))**2)
        ss5 += np.diag(np.dot(Z.T, Z) * S_i).sum()

    v = ss5 / ss4

    return b, U, D, v


def random_orth_mat(n, m, rand=None):
    rand = np.random if rand is None else rand
    A = rand.normal(size=(n, m))
    return la.svd(A, full_matrices=False)[0]


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from bsplines import BSplineBasis

    logging.basicConfig(level=logging.INFO)

    bone = pd.read_csv('datasets/bone_density.csv')
    subjects = bone.groupby('idnum')
    trajectories = [(d['age'].values, d['spnbmd'].values) for _, d in subjects]

    low, high = 9, 26
    num_bases = 4
    basis = BSplineBasis.uniform(low, high, num_bases, degree=2)
    dataset = [(y, basis(x)) for x, y in trajectories]

    model = learn_fpca(dataset, hid_dim=2)

    xgrid = np.linspace(low, high, 200)
    Xgrid = basis(xgrid)
    for x, y in trajectories:
        plt.plot(x, y, 'kx')

    b, U, D, v = model.param_copy()

    plt.plot(xgrid, np.dot(Xgrid, b), label='Mean')

    princomp = (U * np.sqrt(D)).T
    for k, u in enumerate(princomp[:2]):
        plt.plot(xgrid, np.dot(Xgrid, u), label='PC {}'.format(k + 1))

    plt.legend()
        
    plt.show()
