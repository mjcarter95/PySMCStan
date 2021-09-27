import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from SMC_BASE import Target_Base, Q0_Base, Q_Base
from SMC_OPT import *
from SMC_OPT_MC import *

"""
Estimating the optimum L-kernel for a D-dimensional toy problem using the
single_step proposal approach.

P.L.Green
"""

# Dimension of problem
D = 10

class Target(Target_Base):
    """ Define target """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.repeat(2, D), cov=np.eye(D))

    def logpdf(self, x):
        return self.pdf.logpdf(x)


class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.zeros(D), cov=np.eye(D))


    def logpdf(self, x):
        return self.pdf.logpdf(x)

    def rvs(self, size):
        return self.pdf.rvs(size)


class Q(Q_Base):
    """ Define general proposal """

    def pdf(self, x, x_cond):

        dx = np.vstack(x - x_cond)
        p = (2*np.pi)**(-D/2) * np.exp(-0.5 * dx.T @ dx)

        return p[0]

    def logpdf(self, x, x_cond):
        dx = np.vstack(x - x_cond)
        logp = -D/2 * np.log(2*np.pi) - 0.5 * dx.T @ dx
        return logp

    def rvs(self, x_cond):
        return x_cond + np.random.randn(D)

p = Target()
q0 = Q0()
q = Q()

# No. samples and iterations
N = 100
K = 100

# OptL SMC sampler with Gaussian approximation
smc_gauss = SMC_OPT(N, D, p, q0, K, q)
smc_gauss.generate_samples()

# OptL SMC sampler with Monte-Carlo approximation
smc_mc = SMC_OPT_MC(N, D, p, q0, K, q)
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
