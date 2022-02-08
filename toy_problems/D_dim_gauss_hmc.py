import autograd.numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from autograd.scipy import stats as AutoStats
from scipy.stats import multivariate_normal
from SMC_TEMPLATES import Target_Base, Q0_Base
from SMC_HMC_BASE import SMC_HMC

"""
Evaluating optimal L-kernel approaches when targeting a
D-dimensional Gaussian using a fixed step Hamiltonian proposal.

L.J. Devlin
"""

# Dimension of problem
D = 10

class Target(Target_Base):
    """ Define target """

    # For the autodiff to work we need to box the target into one function
    def logpdf(self, x):
        return AutoStats.multivariate_normal.logpdf(x, mean=np.repeat(2, D), cov=np.eye(D))


class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.pdf = multivariate_normal(mean=np.zeros(D), cov=np.eye(D))

    def logpdf(self, x):
        return self.pdf.logpdf(x)

    def rvs(self, size):
        return self.pdf.rvs(size)



p = Target()
q0 = Q0()

# No. samples and iterations
N = 100
K = 100


# Step-size (h), number of Leapfrog steps (k), and Covariance of the initial velocity distribution
#  at the start of a Hamiltonina trajectory (Cov)
h=0.2
k= 3
Cov = 1

# OptL SMC sampler with Monte-Carlo approximation
smc_fp = SMC_HMC(N, D, p, q0, K, h, k, Cov, optL='forwards-proposal')
smc_fp.generate_samples()

# OptL SMC sampler with Gaussian approximation
smc_gauss = SMC_HMC(N, D, p, q0, K, h, k, Cov, optL='gauss')
smc_gauss.generate_samples()

# OptL SMC sampler with Monte-Carlo approximation
smc_mc = SMC_HMC(N, D, p, q0, K, h, k, Cov, optL='monte-carlo')
smc_mc.generate_samples()

# Plots of estimated mean
fig, ax = plt.subplots(ncols=3)
for i in range(3):
    for d in range(D):
        if i == 0:
            ax[i].plot(smc_gauss.mean_estimate_EES[:, d], 'k',
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_mc.mean_estimate_EES[:, d], 'r',
                       alpha=0.5)
        if i == 2:
            ax[i].plot(smc_fp.mean_estimate_EES[:, d], 'b',
                       alpha=0.5)
    ax[i].plot(np.repeat(2, K), 'lime', linewidth=3.0,
               linestyle='--')
    ax[i].set_ylim([-2, 5])
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('E[$x$]')
    if i == 0:
        ax[i].set_title('Gaussian')
    if i == 1:
        ax[i].set_title('Monte-Carlo')
    if i == 2:
        ax[i].set_title('forwards-proposal')
plt.tight_layout()


# Plots of estimated diagonal elements of covariance matrix
fig, ax = plt.subplots(ncols=3)
for i in range(3):
    for d in range(D):
        if i == 0:
            ax[i].plot(smc_gauss.var_estimate_EES[:, d, d], 'k',
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_mc.var_estimate_EES[:, d, d], 'r',
                       alpha=0.5)
        if i == 2:
            ax[i].plot(smc_fp.var_estimate_EES[:, d, d], 'b',
                       alpha=0.5)
    ax[i].plot(np.repeat(1, K), 'lime', linewidth=3.0,
               linestyle='--')
    ax[i].set_ylim([0, 2])
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('Var[$x$]')
    if i == 0:
        ax[i].set_title('Gaussian')
    if i == 1:
        ax[i].set_title('Monte-Carlo')
    if i == 2:
        ax[i].set_title('forwards-proposal')
plt.tight_layout()

# Plot of effective sample size
fig, ax = plt.subplots()
ax.plot(smc_gauss.Neff / smc_gauss.N, 'k',
        label='Gaussian')
ax.plot(smc_mc.Neff / smc_mc.N, 'r',
        label='Monte-Carlo')
ax.plot(smc_fp.Neff / smc_fp.N, 'b',
        label='forwards-proposal')
ax.set_xlabel('Iteration')
ax.set_ylabel('$N_{eff} / N$')
ax.set_ylim([0, 1.1])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()


plt.show()
