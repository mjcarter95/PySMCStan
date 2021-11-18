from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from autograd.scipy import stats as AutoStats
from scipy.stats import multivariate_normal

from SMC_TEMPLATES import Target_Base, Q0_Base
from SMC_HMC_BASE_NEW import SMC_HMC

"""
Evaluating optimal L-kernel approaches when targeting a
D-dimensional Gaussian using a fixed step Hamiltonian proposal.

L.J. Devlin
"""

# Dimension of problem
D = 1


class Target(Target_Base):
    """ Define target """

    def __init__(self):
        self.pdf = multivariate_normal(mean=np.repeat(2, D), cov=np.eye(D))

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
N = 50
K = 100


# OptL SMC sampler with Gaussian approximation
smc_gauss = SMC_HMC(N, D, p, q0, K, proposal='hmc', optL='gauss')
#smc_gauss.generate_samples()

'''
# OptL SMC sampler with Monte-Carlo approximation
smc_mc = SMC_HMC(N, D, p, q0, K, proposal='hmc', optL='monte-carlo')
smc_mc.generate_samples()

# Plots of estimated mean
fig, ax = plt.subplots(ncols=2)
for i in range(2):
    for d in range(D):
        if i == 0:
            ax[i].plot(smc_gauss.mean_estimate_EES[:, d], 'k',
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_mc.mean_estimate_EES[:, d], 'r',
                       alpha=0.5)
    ax[i].plot(np.repeat(2, K), 'lime', linewidth=3.0,
               linestyle='--')
    ax[i].set_ylim([-2, 5])
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('E[$x$]')
    if i == 0:
        ax[i].set_title('(a)')
    if i == 1:
        ax[i].set_title('(b)')
plt.tight_layout()

# Plots of estimated diagonal elements of covariance matrix
fig, ax = plt.subplots(ncols=2)
for i in range(2):
    for d in range(D):
        if i == 0:
            ax[i].plot(smc_gauss.var_estimate_EES[:, d, d], 'k',
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_mc.var_estimate_EES[:, d, d], 'r',
                       alpha=0.5)
    ax[i].plot(np.repeat(1, K), 'lime', linewidth=3.0,
               linestyle='--')
    ax[i].set_ylim([0, 1.5])
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('Var[$x$]')
    if i == 0:
        ax[i].set_title('(a)')
    if i == 1:
        ax[i].set_title('(b)')
plt.tight_layout()

# Plot of effective sample size
fig, ax = plt.subplots()
ax.plot(smc_gauss.Neff / smc_gauss.N, 'k',
        label='Optimal L-kernel (Gaussian)')
ax.plot(smc_mc.Neff / smc_mc.N, 'r',
        label='Optimal L-kernel (Monte-Carlo)')
ax.set_xlabel('Iteration')
ax.set_ylabel('$N_{eff} / N$')
ax.set_ylim([0, 1.1])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.show()

'''