from SMC_BASE import Q_Base
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from scipy.stats import multivariate_normal

class HMC_proposal(Q_Base):

    """
        Description
        -----------
        Fixed step Hamiltonian Monte Carlo proposal distribution.
        
        Parameters
        ----------
        

        p : target distribution

        h : Step size used by Leapfrog

        D: Dimension of target

        Steps : no. of steps made by Leapfrog

        v_dist: Distribuion of the velocity 

        Cov: Scalar term which increases the variance of the velocity distribution

        grad_x: gradient of the target w.r.t. x

    
    """

    def __init__(self, D, p, h, steps, Cov):
        self.D = D
        self.target = p
        self.h=h
        self.steps=steps
        self.grad=egrad(self.target.logpdf)
        self.v_dist = multivariate_normal(mean=np.zeros(D), cov=Cov*np.eye(D))

        
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
        x = x_cond[0,:] 
        v = x_cond[1,:]
        grad_x=x_cond[2, :]

        x_new, v_new = self.generate_HMC_samples(x, v, grad_x)
        return x_new, v_new


    def generate_HMC_samples(self, x, v, grad_x):
        
        """
        Description
        -----------
        Handles the fixed step HMC proposal
        """

        # Main leapfrog loop, fixed to 5 s steps
        for k in range(0,self.steps):

            x, v, grad_x = self.Leapfrog(x, v, grad_x)
        
        return x, v


    # Supporting functions - Will Likely move when introducing NUTS

    def Leapfrog(self, x, v, grad_x):
        
        """
        Description
        -----------
        Performs a single Leapfrog step returning the final position, velocity and gradient.
        """
        
        v = np.add(v, (self.h/2)*grad_x)
        x = np.add(x, self.h*v)
        grad_x = self.grad(x)
        v = np.add(v, (self.h/2)*grad_x)
        
        return x, v, grad_x

    def v_rvs(self, size):

        """
        Description
        -----------
        Draw a number of samples equal to size from the velocity/momentum distribution
        """
        
        return self.v_dist.rvs(size)

    def v_pdf(self, x):
        
        """
        Description
        -----------
        Calculate pdf of velocity/momentum distribution
        """
         
        return self.v_dist.pdf(x)

    def v_logpdf(self, x):
        
        """
        Description
        -----------
        Calculate logpdf of velocity/momentum distribution
        """

        return self.v_dist.logpdf(x)
