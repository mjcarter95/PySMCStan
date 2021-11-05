from SMC_TEMPLATES import Q_Base
import numpy as np


class random_walk_proposal(Q_Base):

    """
        Description
        -----------
        Generic Random Walk proposal distribution with unit variance.

    """

    def __init__(self, D):
        self.D = D

    def pdf(self, x, x_cond):
        """
        Description
        -----------
        Returns pdf from a normal distribution (with unit variance and
        mean x_cond) for parameter x.
        """

        dx = np.vstack(x - x_cond)
        p = (2*np.pi)**(-self.D/2) * np.exp(-0.5 * dx.T @ dx)

        return p[0]

    def logpdf(self, x, x_cond):
        """
        Description
        -----------
        Returns logpdf from a normal distribution (with unit variance and
        mean x_cond) for parameter x.
        """

        dx = np.vstack(x - x_cond)
        logp = -self.D/2 * np.log(2*np.pi) - 0.5 * dx.T @ dx
        return logp

    def rvs(self, x_cond):
        """
        Description
        -----------
        Returns a new sample state based on a standard normal Gaussian
        random walk.
        """
        
        return x_cond + np.random.randn(self.D)

    def kernel_parameters(self, x, x_new):
        """
        Description
        -----------
        Returns values required to calculate both the forward proposal distribution for a random walk proposal and the parameters required
        to calculate the L-kernel. For a random walk proposal this equates to x_new and x for the proposal and x, and x_new for the L-kernel
        """
        return x_new, x, x, x_new
