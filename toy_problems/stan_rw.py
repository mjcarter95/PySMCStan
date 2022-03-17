import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from scipy.stats import multivariate_normal as Normal_PDF
from SMC_BASE import SMC
from SMC_TEMPLATES import Target_Base, Q0_Base
from STAN_MODEL import StanModel, read_data, read_model


"""
Evaluating optimal L-kernel approaches when targeting a
D-dimensional Gaussian.

P.L.Green
"""

# Load Stan model
model_name = "eight_schools"
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
        self.pdf = Normal_PDF(mean=np.zeros(sm.D), cov=np.eye(sm.D))

    def logpdf(self, x):
        return self.pdf.logpdf(x)

    def rvs(self, size):
        return self.pdf.rvs(size)

q0 = Q0()

# No. samples and iterations
N = 200
K = 100

# OptL SMC sampler with Gaussian approximation
smc_gauss = SMC(N, sm.D, sm, q0, K, proposal='rw', optL='gauss',
                rc_scheme='ESS_Recycling')
smc_gauss.generate_samples()

# OptL SMC sampler with Monte-Carlo approximation
smc_mc = SMC(N, sm.D, sm, q0, K, proposal='rw', optL='monte-carlo', 
             rc_scheme='ESS_Recycling')
smc_mc.generate_samples()

# Plots of estimated mean
fig, ax = plt.subplots(ncols=2)
for i in range(2):
    for d in range(sm.constrained_D):
        if i == 0:
            ax[i].plot(smc_gauss.constrained_mean_estimate_rc[:, d], 'k',
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_mc.constrained_mean_estimate_rc[:, d], 'r',
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
    for d in range(sm.D):
        if i == 0:
            ax[i].plot(smc_gauss.var_estimate_rc[:, d, d], 'k',
                       alpha=0.5)
        if i == 1:
            ax[i].plot(smc_mc.var_estimate_rc[:, d, d], 'r',
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

