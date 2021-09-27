import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from SMC_BASE import SMC, Target_Base, Q0_Base, Q_Base, L_Base
from SMC_OPT import *
from SMC_OPT_MC import *

"""
Estimating the optimum L-kernel for a 2D toy problem.

P.L.Green
"""


class Target(Target_Base):
    """ Define target """

    def __init__(self):
        self.pdf = Normal_PDF(mean=np.array([3.0, 2.0]), cov=np.eye(2))

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
        return (2 * np.pi)**-1 * np.exp(-0.5 * (x - x_cond).T @ (x - x_cond))

    def logpdf(self, x, x_cond):
        return  -0.5 * (x - x_cond).T @ (x - x_cond)

    def rvs(self, x_cond):
        return x_cond + np.random.randn(2)


class L(L_Base):
    """ Define L-kernel """

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ (x - x_cond)


p = Target()
q0 = Q0()
q = Q()
l = L()

# No. samples, iterations, and times we'll run the experiment
N = 100
K = 20
N_MC = 100

# Initialise figures
fig_mean, ax_mean = plt.subplots(nrows=2, ncols=1)
fig_cov, ax_cov = plt.subplots(nrows=3, ncols=1)

# Loop over experiments
for n_mc in range(N_MC):
    print('Iteration ', n_mc)

    # SMC sampler with user-defined L-kernel
    smc = SMC(N, 2, p, q0, K, q, l)
    smc.generate_samples()

    # SMC sampler with Gaussian L-kernel
    smc_optL = SMC_OPT(N, 2, p, q0, K, q)
    smc_optL.generate_samples()

    # SMC sampler with Monte-Carlo estimate L-kernel
    smc_mc = SMC_OPT_MC(N, 2, p, q0, K, q)
    smc_mc.generate_samples()

    # Plots of estimated mean
    if n_mc == 0:
        ax_mean[0].plot(np.repeat(3, K), 'lime', linewidth=3.0, label='True value')
    ax_mean[0].plot(smc.mean_estimate_EES[:, 0], 'k', label='Forward proposal L-kernel', alpha=0.5)
    ax_mean[0].plot(smc_optL.mean_estimate_EES[:, 0], 'r', label='L-kernel (Gaussian)', alpha=0.5)
    ax_mean[0].plot(smc_mc.mean_estimate_EES[:, 0], 'b', label='L-kernel (Monte-Carlo)', alpha=0.5)
    if n_mc == 0:
        ax_mean[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_mean[0].set_title('(a)')
        ax_mean[0].set_xlabel('Iteration')
        ax_mean[0].set_ylabel('E[$x_1$]')
        ax_mean[1].plot(np.repeat(2, K), 'lime', linewidth=3.0)
    ax_mean[1].plot(smc.mean_estimate_EES[:, 1], 'k', alpha=0.5)
    ax_mean[1].plot(smc_optL.mean_estimate_EES[:, 1], 'r', alpha=0.5)
    ax_mean[1].plot(smc_mc.mean_estimate_EES[:, 1], 'b', alpha=0.5)
    if n_mc == 0:
        ax_mean[1].set_title('(b)')
        ax_mean[1].set_xlabel('Iteration')
        ax_mean[1].set_ylabel('E[$x_2$]')

    plt.tight_layout()

    # Plots of estimated elements of covariance matrix
    if n_mc == 0:
        ax_cov[0].plot(np.repeat(1, K), 'lime', linewidth=3.0, label='True value')
    ax_cov[0].plot(smc.var_estimate_EES[:, 0, 0], 'k',
               label='Forward proposal L-kernel', alpha=0.5)
    ax_cov[0].plot(smc_optL.var_estimate_EES[:, 0, 0], 'r', label='L-kernel (Gaussian)', alpha=0.5)
    ax_cov[0].plot(smc_mc.var_estimate_EES[:, 0, 0], 'b', label='L-kernel (Monte-Carlo)', alpha=0.5)
    if n_mc == 0:
        ax_cov[0].set_xlabel('Iteration')
        ax_cov[0].set_ylabel('Cov$[x_1, x_1]$')
        ax_cov[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_cov[1].plot(np.repeat(0, K), 'lime', linewidth=3.0)
    ax_cov[1].plot(smc.var_estimate_EES[:, 0, 1], 'k', alpha=0.5)
    ax_cov[1].plot(smc_optL.var_estimate_EES[:, 0, 1], 'r', alpha=0.5)
    ax_cov[1].plot(smc_mc.var_estimate_EES[:, 0, 1], 'b', alpha=0.5)
    if n_mc == 0:
        ax_cov[1].set_xlabel('Iteration')
        ax_cov[1].set_ylabel('Cov$[x_1, x_2]$')
        ax_cov[2].plot(np.repeat(1, K), 'lime', linewidth=3.0)
    ax_cov[2].plot(smc.var_estimate_EES[:, 1, 1], 'k', alpha=0.5)
    ax_cov[2].plot(smc_optL.var_estimate_EES[:, 1, 1], 'r', alpha=0.5)
    ax_cov[2].plot(smc_mc.var_estimate_EES[:, 1, 1], 'b', alpha=0.5)
    if n_mc == 0:
        ax_cov[2].set_xlabel('Iteration')
        ax_cov[2].set_ylabel('Cov$[x_2, x_2]$')

    plt.tight_layout()


plt.show()
