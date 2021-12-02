from SMC_BASE import Q_Base
import autograd.numpy as np
from autograd import elementwise_grad as egrad


class HMC_proposal(Q_Base):

    """
        Description
        -----------
        Fixed step Hamiltonian Monte Carlo proposal distribution.

    """

    def __init__(self, D, p,h,steps):
        self.D = D
        self.target = p
        self.h=h
        self.steps=steps
        self.grad=egrad(self.target.logpdf)

        
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

        # Calculate the initial gradient
        #grad_x =  self.grad(x)

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
