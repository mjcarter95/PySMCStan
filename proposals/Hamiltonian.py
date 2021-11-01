from SMC_BASE import Q_Base
import autograd.numpy as np
from autograd import elementwise_grad, grad 
from autograd.scipy.stats import multivariate_normal


class HMC_proposal(Q_Base):

    """
        Description
        -----------
        Fixed step Hamiltonian Monte Carlo proposal distribution.

    """

    def __init__(self, D, p):
        self.D = D
        self.logprob = p.logpdf

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
        x_new = self.generate_HMC_samples(x_cond)
        return x_new 


    def generate_HMC_samples(self, x):
        
        """
        Description
        -----------
        Handles the fixed step HMC proposal
        """

        # Sample an initial velocity vector
        v = np.array(np.random.normal(0,1,self.D))
        v_pdf = multivariate_normal.logpdf(x, 0, 1)

        # Too lazy to think of a better way to do this at the moment
        # set the initial velocity as a member variable
        self.vi_pdf = v_pdf

        # Calculate the initial gradient
        grad_x = self.gradient_finite_differece(x)

        # Main leapfrog loop, fixed to 5 s steps
        for k in range(0,5):
            x, v, grad_x = self.Leapfrog(x, v, grad_x)

        self.vf = v

        
        return x

    def gradient_finite_differece(self, x):
        
        """
        Description
        -----------
        Returns gradient at x using finte difference method.
        """
        
        eps=1e-7
        
        return (self.logpdf(x+eps) - self.logpdf(x))/eps

    # Supporting functions - Will Likely move when introducing NUTS

    def Leapfrog(self, x, v, grad_x):
        
        """
        Description
        -----------
        Performs a single Leapfrog step returning the final position, velocity and gradient.
        """
        
        # We will hard-code the step-size for the moment
        h=0.1
        
        v = np.add(v, (h/2)*grad_x)
        x = np.add(x, h*v)
        grad_x = self.gradient_finite_differece(x)
        v = np.add(v, (h/2)*grad_x)
        
        return x, v, grad_x

    def propose_step_size(x):
        pass
