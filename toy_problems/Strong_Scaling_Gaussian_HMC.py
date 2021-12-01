from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # noqa
from autograd.scipy import stats as AutoStats
from scipy.stats import multivariate_normal
from SMC_TEMPLATES import Target_Base, Q0_Base
from SMC_HMC_BASE import SMC_HMC
import pandas as pd
import time

"""
Evaluating optimal L-kernel approaches when targeting a
D-dimensional Gaussian using a fixed step Hamiltonian proposal.

L.J. Devlin
"""

# Dimension of problem
D = 10


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

Iterations = [20, 40, 50, 80, 100, 200, 400]
Samples    = [1000, 500, 400, 250, 200, 100, 50]

Gauss = []
MC = []
FP = []

Gauss_time = []
MC_time = []
FP_time=[]

for not_k in range(0,7):

    # No. samples and iterations
    N = Samples[not_k]
    K = Iterations[not_k]

    # Step-size and number of Leapfrog steps
    h=0.8
    k=1

    # OptL SMC sampler with Monte-Carlo approximation
    smc_fp = SMC_HMC(N, D, p, q0, K, h, k, proposal='hmc', optL='forwards-proposal')
    start = time.time()
    smc_fp.generate_samples()
    FP_time.append(time.time()-start)

    # OptL SMC sampler with Gaussian approximation
    smc_gauss = SMC_HMC(N, D, p, q0, K, h, k, proposal='hmc', optL='gauss')
    start = time.time()
    smc_gauss.generate_samples()
    Gauss_time.append(time.time()-start)

    # OptL SMC sampler with Monte-Carlo approximation
    smc_mc = SMC_HMC(N, D, p, q0, K, h, k, proposal='hmc', optL='monte-carlo')
    start = time.time()
    smc_mc.generate_samples()
    MC_time.append(time.time()-start)

    Gauss_mean = smc_gauss.mean_estimate_EES[-1, :]
    Gauss_error = Gauss_mean - 2.0
    gauss_error = np.linalg.norm(Gauss_error)
    Gauss.append(gauss_error)

    MC_mean = smc_mc.mean_estimate_EES[-1, :]
    MC_error = MC_mean - 2.0
    mc_error = np.linalg.norm(MC_error)
    MC.append(mc_error)

    fp_mean = smc_fp.mean_estimate_EES[-1, :]
    FP_error = fp_mean - 2.0
    fp_error = np.linalg.norm(FP_error)
    FP.append(fp_error)
    
    
    print(not_k)

Final = pd.DataFrame({
    'Iterations':Iterations,
    'Samples': Samples,
    'Gauss_error': Gauss,
    'MC_error': MC,
    'FP_error': FP,
    'Gauss_time': Gauss_time,
    'MC_time': MC_time,
    'FP_time':FP_time
})

Final.to_csv('Output.csv')

