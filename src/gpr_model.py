import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

class TunedGPR(BaseEstimator, RegressorMixin):
    """ARD-RBF Gaussian process. Kernel hyperparameters are tuned by marginal likelihood on a
    random subset (n_tune) of the training data, then the GP is refitted on ALL training data with
    the kernel fixed. Prediction is batched so large candidate sets fit in memory."""
    def __init__(self, n_tune=500, max_iter=60, random_state=42, batch=4000):
        self.n_tune = n_tune; self.max_iter = max_iter
        self.random_state = random_state; self.batch = batch

    def _opt(self, obj_func, initial_theta, bounds):
        theta, fval, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=self.max_iter)
        return theta, fval

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.scaler_ = StandardScaler().fit(X); Xs = self.scaler_.transform(X)
        self.ymu_, self.ysd_ = y.mean(), y.std(); ys = (y - self.ymu_) / self.ysd_
        rng = np.random.RandomState(self.random_state)
        sub = rng.choice(len(Xs), min(self.n_tune, len(Xs)), replace=False)
        kernel = (C(1.0, (1e-2, 1e2)) * RBF(np.ones(Xs.shape[1]), (1e-2, 1e3))
                  + WhiteKernel(1e-3, (1e-8, 1e-1)))
        tuner = GaussianProcessRegressor(kernel, optimizer=self._opt, random_state=self.random_state)
        tuner.fit(Xs[sub], ys[sub])
        self.kernel_ = tuner.kernel_
        self.gp_ = GaussianProcessRegressor(self.kernel_, optimizer=None).fit(Xs, ys)
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_.transform(np.asarray(X, float))
        mus, sds = [], []
        for i in range(0, len(Xs), self.batch):
            m, s = self.gp_.predict(Xs[i:i + self.batch], return_std=True)
            mus.append(m); sds.append(s)
        mu = np.concatenate(mus) * self.ysd_ + self.ymu_
        sd = np.concatenate(sds) * self.ysd_
        return (mu, sd) if return_std else mu
