import numpy as np
from SMC_BASE import SMC, L_Base
from scipy.stats import multivariate_normal as Normal_PDF

class L(L_Base):

    def logpdf(self, x, x_cond):
        pass


class SMC_OPT_MC(SMC):
    """
    A class of SMC sampler that builds on the SMC base class by allowing a
    Monte-Carlo approximation of the optimal L-kernel.

    P.L.Green
    """

    def __init__(self, N, D, p, q0, K, q):
        """ Initialiser class method

        """

        # Initiate standard SMC sampler but with no L-kernel defined
        super().__init__(N, D, p, q0, K, q, L())

    def update_weights(self, x, x_new, logw, p_logpdf_x,
                       p_logpdf_x_new, d=None):
        """ Overwrites the method in the base-class

        """

        # Initialise arrays
        logw_new = np.vstack(np.zeros(self.N))

        # Find new weights
        for i in range(self.N):

            # Initialise what will be the denominator of our
            # weight-update equation
            den = np.zeros(1)

            # Realise Monte-Carlo estimate of denominator
            for j in range(self.N):
                den += self.q.pdf(x_new[i], x[j])
            den /= self.N
            
            # Calculate new log-weight
            logw_new[i] = p_logpdf_x_new[i] - np.log(den)

        return logw_new
