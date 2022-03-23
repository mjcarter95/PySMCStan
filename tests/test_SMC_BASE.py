import numpy as np
import sys
sys.path.append('../..')  # noqa
from SMC_BASE import SMC
from SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from scipy.stats import multivariate_normal as Normal_PDF

"""
Testing for SMC_BASE

P.L.Green
"""

np.random.seed(42)


class Target(Target_Base):
    """ Define target """

    def __init__(self):
        self.mean = np.array([3.0, 2.0])
        self.cov = np.eye(2)
        self.pdf = Normal_PDF(self.mean, self.cov)

    def logpdf(self, x):
        return self.pdf.logpdf(x)


class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.zeros(2), cov=np.eye(2))

    def logpdf(self, x):
        return self.pdf.logpdf(x)

    def rvs(self, size):
        return self.pdf.rvs(size)


class Q(Q_Base):
    """ Define general proposal """

    def pdf(self, x, x_cond):
        return (2 * np.pi)**-0.5 * np.exp(-0.5 * (x - x_cond).T @ (x - x_cond))

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ (x - x_cond)

    def rvs(self, x_cond):
        return x_cond + np.random.randn(2)


# No. samples and iterations
N = 100
K = 20

# Define problem
p = Target()
q0 = Q0()
q = Q()

def test_gauss_optL():
    """ Test the predictions made by an SMC sampler with Gaussian
        approximation of the optimal L-kernel.

    """

    # SMC sampler
    smc = SMC(N, 2, p, q0, K, proposal=q, optL='gauss', rc_scheme='ESS_Recycling')
    smc.generate_samples()

    # Check estimates
    assert np.allclose(smc.mean_estimate_rc[-1], p.mean, atol=0.1)
    assert np.allclose(smc.var_estimate_rc[-1][0][0], p.cov[0][0],
                       atol=0.2)
    assert np.allclose(smc.var_estimate[-1][1][1], p.cov[1][1],
                       atol=0.2)
    assert np.allclose(smc.var_estimate[-1][0][1], p.cov[0][1],
                       atol=0.2)

    # Check that the sampler will run without a recycling scheme
    smc = SMC(N, 2, p, q0, K, proposal=q, optL='gauss')
    smc.generate_samples()


def test_monte_carlo_optL():
    """ Test the predictions made by an SMC sampler with Monte-Carlo
        approximation of the optimal L-kernel.

    """

    # SMC sampler
    smc = SMC(N, 2, p, q0, K, proposal=q, optL='monte-carlo', rc_scheme='ESS_Recycling')
    smc.generate_samples()

    # Check estimates
    assert np.allclose(smc.mean_estimate_rc[-1], p.mean, atol=0.1)
    assert np.allclose(smc.var_estimate_rc[-1][0][0], p.cov[0][0],
                       atol=0.2)
    assert np.allclose(smc.var_estimate[-1][1][1], p.cov[1][1],
                       atol=0.2)
    assert np.allclose(smc.var_estimate[-1][0][1], p.cov[0][1],
                       atol=0.2)
