import autograd.numpy as np
import importance_sampling as IS
from abc import abstractmethod, ABC
from SMC_BASE import SMC 
from autograd.scipy.stats import multivariate_normal
import time


class Target_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the target distribution.

    """

    @abstractmethod
    def logpdf(self, x):
        """
        Description
        -----------
        Returns log pdf of the target distribution, evaluated at x.

        """
        pass


class Q0_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the initial proposal distribution.

    """

    @abstractmethod
    def logpdf(self, x):
        """
        Description
        -----------
        Returns log pdf of the initial proposal, evaluated at x.
        """
        pass

    @abstractmethod
    def rvs(self, size):
        """
        Description
        -----------
        Returns samples from the initial proposal.

        Parameters
        ----------
        size : size of the sample being returned
        """
        pass

class Q_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the general proposal distribution.
    """

    @abstractmethod
    def pdf(self, x, x_cond):
        """
        Description
        -----------
        Returns q(x | x_cond)
        """
        pass

    @abstractmethod
    def logpdf(self, x, x_cond):
        """
        Description
        -----------
        Returns log q(x | x_cond)
        """
        pass

    @abstractmethod
    def rvs(self, x_cond):
        """
        Description
        -----------
        Returns a single sample from the proposal, q(x | x_cond).
        """

        pass


class SMC_HMC(SMC):

    """
    Description
    -----------
    A base class for an SMC sampler.

    Parameters
    ----------
    N : no. of samples generated at each iteration

    D : dimension of the target distribution

    p : target distribution instance

    q0 : initial proposal instance

    K : no. iterations to run

    q : general proposal distribution instance

    optL : approximation method for the optimal L-kernel. Can be either
        'gauss' or 'monte-carlo' (representing a Gaussian approximation
        or a Monte-Carlo approximation respectively).

    Methods
    -------

    estimate : realise importance sampling estimates of mean and
        covariance matrix of the target.

    generate_samples : runs the SMC sampler to generate weighted
        samples from the target.

    update_weights : updates the log weight associated with each sample
        i.e. evaluates the incremental weights.

    Author
    ------
    P.L.Green
    """

    def __init__(self, N, D, p, q0, K, proposal, optL, verbose=False):

        # Assign variables to self
        self.N = N
        self.D = D
        self.p = p
        self.q0 = q0
        self.K = K
        self.optL = optL
        self.verbose = verbose

        if(isinstance(proposal, Q_Base)):
            self.q = proposal
            self.proposal='user'
        elif(proposal == 'rw'):
            from proposals.random_walk import random_walk_proposal
            self.q = random_walk_proposal(self.D)
            self.proposal = 'rw'
        elif(proposal == 'hmc'):
            from proposals.Hamiltonian import HMC_proposal
            self.q = HMC_proposal(self.D, p)
            self.v_new = np.zeros([self.N, self.D])
            self.v_ini = np.zeros([self.N, self.D])
            self.q_ini = np.zeros([self.N])
            self.proposal = 'hmc'

    def generate_samples(self):

        """
        Description
        -----------
        Run SMC sampler to generate weighted samples from the target.

        """

        # Initialise arrays
        x_new = np.zeros([self.N, self.D])
        lr = np.array([])

        # Initilise estimates of target mean and covariance matrix
        self.mean_estimate = np.zeros([self.K, self.D])
        self.mean_estimate_EES = np.zeros([self.K, self.D])
        if self.D == 1:
            self.var_estimate = np.zeros([self.K, self.D])
            self.var_estimate_EES = np.zeros([self.K, self.D])
        else:
            self.var_estimate = np.zeros([self.K, self.D, self.D])
            self.var_estimate_EES = np.zeros([self.K, self.D, self.D])

        # Used to record the effective sample size and the points
        # where resampling occurred.
        self.Neff = np.zeros(self.K)
        self.resampling_points = np.array([])

        # Sample from prior and find initial evaluations of the
        # target and the prior. Note that, be default, we keep
        # the log weights vertically stacked.
        x = np.vstack(self.q0.rvs(size=self.N))
        p_logpdf_x = np.vstack(self.p.logpdf(x))
        p_q0_x = np.vstack(self.q0.logpdf(x))

        # Find weights of prior samples
        logw = p_logpdf_x - p_q0_x

        # Main sampling loop
        for self.k in range(self.K):

            if self.verbose:
                print('\nIteration :', self.k)

            # Find normalised weights and realise estimates
            wn = IS.normalise_weights(logw)
            (self.mean_estimate[self.k],
             self.var_estimate[self.k]) = IS.estimate(x, wn, self.D, self.N)

            # EES recycling scheme
            lr = np.append(lr, np.sum(wn)**2 / np.sum(wn**2))
            lmbda = np.array([])
            for k_dash in range(self.k + 1):
                lmbda = np.append(lmbda, lr[k_dash] / np.sum(lr))
                self.mean_estimate_EES[self.k] += (lmbda[k_dash] *
                                                   self.mean_estimate[k_dash])
                self.var_estimate_EES[self.k] += (lmbda[k_dash] *
                                                  self.var_estimate[k_dash])

            # Record effective sample size at kth iteration
            self.Neff[self.k] = 1 / np.sum(np.square(wn))

            # Resample if effective sample size is below threshold
            if self.Neff[self.k] < self.N/2:

                self.resampling_points = np.append(self.resampling_points,
                                                   self.k)
                x, p_logpdf_x, wn = IS.resample(x, p_logpdf_x, wn, self.N)
                logw = np.log(wn)

            # This is horrible, need to find a better way
            # Propose new samples
            if(self.proposal=='hmc'):
                for i in range(self.N):
                    x_new[i] = self.q.rvs(x_cond=x[i])
                    self.v_new[i]=self.q.vf
                    self.v_ini[i]=self.q.vi
                    self.q_ini[i]=self.q.v_pdf

            else:
                for i in range(self.N):
                    x_new[i] = self.q.rvs(x_cond=x[i])


            # Make sure evaluations of likelihood are vectorised
            p_logpdf_x_new = self.p.logpdf(x_new)

            # Update log weights
            logw_new = self.update_weights(-1*self.v_new, x_new, logw, p_logpdf_x,
                                           p_logpdf_x_new, self.v_ini, x)

            # Make sure that, if p.logpdf(x_new) is -inf, then logw_new
            # will also be -inf. Otherwise it is returned as NaN.
            for i in range(self.N):
                if p_logpdf_x_new[i] == -np.inf:
                    logw_new[i] = -np.inf
                elif logw[i] == -np.inf:
                    logw_new[i] = -np.inf

            # Update samples, log weights, and posterior evaluations
            x = np.copy(x_new)
            logw = np.copy(logw_new)
            p_logpdf_x = np.copy(p_logpdf_x_new)

        # Final quantities to be returned
        self.x = x
        self.logw = logw

    def update_weights(self, x, x_new, logw, p_logpdf_x,
                       p_logpdf_x_new, v_old, x_old):
        """
        Description
        -----------
        Used to update the log weights of a new set of samples, using the
            weights of the samples from the previous iteration. This is
            either done using a Gaussian approximation or a Monte-Carlo
            approximation of the opimal L-kernel.

        Parameters
        ----------
        x : samples from the previous iteration

        x_new : samples from the current iteration

        logw : low importance weights associated with x

        p_logpdf_x : log target evaluations associated with x

        p_logpdf_x_new : log target evaluations associated with x_new

        Returns
        -------
        logw_new : log weights associated with x_new

        """

        # Initialise
        logw_new = np.vstack(np.zeros(self.N))

        # Use Gaussian approximation of the optimal L-kernel
        if self.optL == 'gauss':

            # Collect x and x_new together into X
            X = np.hstack([x, x_new])

            # Directly estimate the mean and covariance matrix of X
            mu_X = np.mean(X, axis=0)
            cov_X = np.cov(np.transpose(X))

            # Find mean of the joint distribution (p(x, x_new))
            mu_x, mu_xnew = mu_X[0:self.D], mu_X[self.D:2 * self.D]

            # Find covariance matrix of joint distribution (p(x, x_new))
            (cov_x_x,
             cov_x_xnew,
             cov_xnew_x,
             cov_xnew_xnew) = (cov_X[0:self.D, 0:self.D],
                               cov_X[0:self.D, self.D:2 * self.D],
                               cov_X[self.D:2 * self.D, 0:self.D],
                               cov_X[self.D:2 * self.D, self.D:2 * self.D])

            # Define new L-kernel
            def L_logpdf(x, x_cond):

                # Mean of approximately optimal L-kernel
                mu = (mu_x + cov_x_xnew @ np.linalg.inv(cov_xnew_xnew) @
                      (x_cond - mu_xnew))

                # Variance of approximately optimal L-kernel
                cov = (cov_x_x - cov_x_xnew @
                       np.linalg.inv(cov_xnew_xnew) @ cov_xnew_x)

                # Add ridge to avoid singularities
                cov += np.eye(self.D) * 1e-6

                # Log det covariance matrix
                sign, logdet = np.linalg.slogdet(cov)
                log_det_cov = sign * logdet

                # Inverse covariance matrix
                inv_cov = np.linalg.inv(cov)

                # Find log pdf
                logpdf = (-0.5 * log_det_cov -
                          0.5 * (x - mu).T @ inv_cov @ (x - mu))

                return logpdf

            # Find new weights
            for i in range(self.N):
                logw_new[i] = (logw[i] +
                               p_logpdf_x_new[i] -
                               p_logpdf_x[i] +
                                L_logpdf(x[i], x_new[i]) -
                               multivariate_normal.logpdf(v_old[i], np.zeros(self.D), np.eye(self.D)))

        
        # Use Monte-Carlo approximation of the optimal L-kernel
        if self.optL == 'monte-carlo':

            for i in range(self.N):

                # Initialise what will be the denominator of our
                # weight-update equation
                den = np.zeros(1)

                # Realise Monte-Carlo estimate of denominator
                for j in range(self.N):
                    
                    H= 2*(-p_logpdf_x_new[i]+p_logpdf_x[j]+0.5*np.dot(x[i],x[i]))
                    if(H < 0):
                        continue
                    else:
                        v_other = np.sqrt(H)
                        den+=multivariate_normal.pdf(v_other, np.zeros(1), np.eye(1))
                                
                den /= self.N

                # Calculate new log-weight
                logw_new[i] = p_logpdf_x_new[i] - np.log(den)

        return logw_new


        # H=Potential+
