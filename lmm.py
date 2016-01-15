'''Linear Mixed Model (LMM)'''

import logging
import numpy as np
import scipy.linalg as la

from scipy.stats import multivariate_normal as mvn


class LinearMixedModel:

    def __init__(self, p1, p2):
        self._coef = np.zeros(p1)
        self._ranef_cov = np.eye(p2)
        self._noise_var = 1.0

    def param_copy(self):
        beta = np.array(self._coef)
        Sigma = np.array(self._ranef_cov)
        v = float(self._noise_var)
        return beta, Sigma, v

    def log_likelihood(self, y, X, Z):
        m = np.dot(X, self._coef)
        S = np.dot(Z, np.dot(self._ranef_cov, Z.T))
        S += self._noise_var * np.eye(len(y))
        return mvn.logpdf(y, m, S)

    def posterior(self, y, X, Z):
        resid = y - np.dot(X, self._coef)
        P = la.inv(self._ranef_cov) + np.dot(Z.T, Z) / self._noise_var
        S = la.inv(P)
        m = la.solve(P, np.dot(Z.T, resid) / self._noise_var)
        return m, S


def learn_lmm(dataset, maxiter=500, tol=1e-5):
    objective = lambda lmm: sum(lmm.log_likelihood(*d) for d in dataset)
    
    p1 = dataset[0][1].shape[1]
    p2 = dataset[0][2].shape[1]
    lmm = LinearMixedModel(p1, p2)

    logl = objective(lmm)

    for iteration in range(maxiter):
        lmm._coef, lmm._ranef_cov, lmm._noise_var = em_step(dataset, lmm)

        logl_old = logl
        logl = objective(lmm)

        delta = (logl - logl_old) / np.abs(logl_old)

        msg = 'Iteration={:05d} LL={:20.8f}, dLL={:20.8f}'
        logging.info(msg.format(iteration, logl, delta))

        if delta < tol:
            break

    return lmm


def em_step(dataset, lmm):
    beta, Sigma, _ = lmm.param_copy()

    ss1 = 0.0
    ss2 = 0.0
    ss3 = 0.0
    ss4 = 0.0
    ss5 = 0.0

    for y, X, Z in dataset:
        m, S = lmm.posterior(y, X, Z)
        exp_res = np.sum((y - np.dot(X, beta) - np.dot(Z, m))**2)
        exp_res += np.diag(np.dot(Z.T, Z) * Sigma).sum()

        ss1 += np.dot(X.T, X)
        ss2 += np.dot(X.T, y - np.dot(Z, m))

        ss3 += S + np.outer(m, m)

        ss4 += len(y)
        ss5 += exp_res

    beta = la.solve(ss1, ss2)
    Sigma = ss3 / len(dataset)
    v = ss5 / ss4

    return beta, Sigma, v


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from bsplines import BSplineBasis

    logging.basicConfig(level=logging.INFO)

    bone = pd.read_csv('datasets/bone_density.csv')
    subjects = bone.groupby('idnum')
    trajectories = [(d['age'].values, d['spnbmd'].values) for _, d in subjects]

    low, high = 9, 26
    basis = BSplineBasis.uniform(low, high, num_bases=4, degree=2)
    dataset = [(y, basis(x), basis(x)) for x, y in trajectories]

    model = learn_lmm(dataset, maxiter=500, tol=1e-4)
    
    xgrid = np.linspace(low, high, 200)
    Xgrid = basis(xgrid)
    for x, y in trajectories:
        plt.plot(x, y, 'kx')

    b, S, v = model.param_copy()
    
    plt.plot(xgrid, np.dot(Xgrid, b), label='Mean')

    U, S, _ = la.svd(S, full_matrices=False)
    princomp = (U * np.sqrt(S)).T
    for k, u in enumerate(princomp[:2]):
        plt.plot(xgrid, np.dot(Xgrid, u), label='PC {}'.format(k + 1))

    plt.legend()

    plt.show()
