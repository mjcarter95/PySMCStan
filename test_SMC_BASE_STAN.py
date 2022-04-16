import numpy as np
import sys
sys.path.append('..')  # noqa
from SMC_BASE import SMC
from SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from scipy.stats import multivariate_normal as Normal_PDF
from STAN_MODEL import StanModel, read_data, read_model

"""
Testing for SMC_BASE

P.L.Green, M.J.Carter
"""

np.random.seed(42)

# Load Stan model
model_name = "gaussian"
gaussian_mean = np.array([-4., -2., 0., 2., 4.])
gaussian_var = np.array([1, 1, 1, 1, 1])
model_data = read_data(model_name)
model_code = read_model(model_name)
sm = StanModel(
    model_name,
    model_code,
    model_data
)

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
q0 = Q0()
q = Q()

def test_gauss_optL():
    """ Test the predictions made by an SMC sampler with Gaussian
        approximation of the optimal L-kernel.

    """

    # SMC sampler
    smc = SMC(N, sm.D, sm, q0, K, proposal=q, optL='gauss', rc_scheme='ESS_Recycling')
    smc.generate_samples()

    # Check estimates
    assert np.allclose(smc.constrained_mean_estimate_rc[-1], gaussian_mean,
                       atol=0.2)
    assert np.allclose(smc.constrained_var_estimate_rc[-1], gaussian_var,
                       atol=0.2)
    assert np.allclose(smc.constrained_mean_estimate[-1], gaussian_mean,
                       atol=0.2)
    assert np.allclose(smc.constrained_var_estimate[-1], gaussian_var,
                       atol=0.2)

    # Check that the sampler will run without a recycling scheme
    smc = SMC(N, sm.D, sm, q0, K, proposal=q, optL='gauss')
    smc.generate_samples()


def test_monte_carlo_optL():
    """ Test the predictions made by an SMC sampler with Monte-Carlo
        approximation of the optimal L-kernel.

    """

    # SMC sampler
    smc = SMC(N, sm.D, sm, q0, K, proposal=q, optL='monte-carlo', rc_scheme='ESS_Recycling')
    smc.generate_samples()

    # Check estimates
    assert np.allclose(smc.constrained_mean_estimate_rc[-1], gaussian_mean,
                       atol=0.2)
    assert np.allclose(smc.constrained_var_estimate_rc[-1], gaussian_var,
                       atol=0.2)
    assert np.allclose(smc.constrained_mean_estimate[-1], gaussian_mean,
                       atol=0.2)
    assert np.allclose(smc.constrained_var_estimate[-1], gaussian_var,
                       atol=0.2)

if __name__ == "__main__":
    test_gauss_optL()
    test_monte_carlo_optL()