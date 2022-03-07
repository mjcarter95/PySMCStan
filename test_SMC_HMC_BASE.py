import numpy as np
import sys
sys.path.append('..')  # noqa
from SMC_HMC_BASE import SMC_HMC
from SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from scipy.stats import multivariate_normal as Normal_PDF
from autograd.scipy import stats as AutoStats

"""
Testing for HMC_SMC_BASE

P.L.Green
"""

np.random.seed(42)


class Target(Target_Base):
    """ Define target """

    # For the autodiff to work we need to box the target into one function
    def logpdf(self, x):
        return AutoStats.multivariate_normal.logpdf(x, mean=np.repeat(2, D), cov=np.eye(D))


class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.zeros(2), cov=np.eye(2))

    def logpdf(self, x):
        return self.pdf.logpdf(x)

    def rvs(self, size):
        return self.pdf.rvs(size)


# No. samples and iterations
N = 100
K = 50

# Dimension of problem
D = 2

# Step-size (h), number of Leapfrog steps (k), and Covariance of the initial velocity distribution
# at the start of a Hamiltonina trajectory (Cov)
h=0.2
k= 3
Cov = 1

# Define problem
p = Target()
q0 = Q0()

def test_gauss_optL():
    """ Test the predictions made by an SMC sampler with Gaussian
        approximation of the optimal L-kernel.

    """

    # SMC sampler
    smc = SMC_HMC(N, D, p, q0, K, h, k, Cov, optL='gauss',
                        rc_scheme='ESS_Recycling')
    smc.generate_samples()

    # Check estimates
    assert np.allclose(smc.mean_estimate_rc[-1], 2, atol=0.1)
    assert np.allclose(smc.var_estimate_rc[-1][0][0], 1,
                        atol=0.2)
    assert np.allclose(smc.var_estimate_rc[-1][1][1], 1,
                        atol=0.2)
    assert np.allclose(smc.var_estimate_rc[-1][0][1], 0,
                        atol=0.2)

    # Check that the sampler will run without a recycling scheme
    smc = SMC_HMC(N, D, p, q0, K, h, k, Cov, optL='gauss',
                  rc_scheme=None)
    smc.generate_samples()
    

def test_monte_carlo_optL():
    """ Test the predictions made by an SMC sampler with Monte-Carlo
        approximation of the optimal L-kernel.

    """

    # SMC sampler
    smc = SMC_HMC(N, D, p, q0, K, h, k, Cov, optL='monte-carlo',
                        rc_scheme='ESS_Recycling')
    smc.generate_samples()

    # Check estimates
    assert np.allclose(smc.mean_estimate_rc[-1], 2, atol=0.1)
    assert np.allclose(smc.var_estimate_rc[-1][0][0], 1,
                        atol=0.2)
    assert np.allclose(smc.var_estimate_rc[-1][1][1], 1,
                        atol=0.2)
    assert np.allclose(smc.var_estimate_rc[-1][0][1], 0,
                        atol=0.2)