import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from SMC_BASE import SMC
from SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base

"""
Evaluating optimal L-kernel approaches when targeting a
D-dimensional Gaussian mixture.

P.L.Green
"""

# Dimension of problem
D = 2


class Target(Target_Base):
    """ Define target """

    def __init__(self):

        # Means
        mu1 = np.vstack(np.repeat(2, D))
        mu2 = np.vstack(np.repeat(-2, D))

        # Covariance matrices
        cov1 = np.eye(D)
        cov2 = np.eye(D)

        # Components of the Gaussian mixture
        self.pdf1 = Normal_PDF(mean=mu1[:, 0], cov=cov1)
        self.pdf2 = Normal_PDF(mean=mu2[:, 0], cov=cov2)

        # Mixture contributions
        self.pi1 = 0.5
        self.pi2 = 0.5

        # True mean of the mixture
        self.mu = self.pi1 * mu1 + self.pi2 * mu2

        # True covariance matrix of the mixture
        self.cov = (self.pi1 * (cov1 + mu1 @ mu1.T) +
                    self.pi2 * (cov2 + mu2 @ mu2.T))

    def logpdf(self, x):

        logp = np.log(self.pi1 * self.pdf1.pdf(x) +
                      self.pi2 * self.pdf2.pdf(x))

        return logp


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
smc_gauss = SMC(N, D, p, q0, K, proposal=q, optL='gauss')
smc_gauss.generate_samples()

# OptL SMC sampler with Monte-Carlo approximation
smc_mc = SMC(N, D, p, q0, K, proposal=q, optL='monte-carlo')
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
    ax[i].plot(np.repeat(p.mu[i], K), 'lime', linewidth=3.0, linestyle='--')
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
    ax[i].plot(np.repeat(p.cov[i, i], K), 'lime', linewidth=3.0,
               linestyle='--')
    ax[i].set_ylim([0, 7])
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
