import numpy as np


class Normal:
    def __init__(self, D=1, mean=None, cov=None):
        self.D = D
        self.mean = mean
        self.cov = cov

    def sample(self, N, mean=None, cov=None):
        if mean is None:
            mean = self.mean

        if cov is None:
            cov = self.cov

        x = np.random.multivariate_normal(mean, cov, N)

        return x

    def log_prob(self, x, mean=None, cov=None):
        if mean is None:
            mean = self.mean

        if cov is None:
            cov = self.cov

        if self.D == 1:
            lp = -1 / 2 * (x - mean) ** 2
        else:
            lp = -1 / 2 * ((x - mean) @ np.linalg.inv(cov) @ (x - mean).T)

        return lp
