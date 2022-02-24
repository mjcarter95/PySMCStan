import autograd.numpy as np
import importance_sampling as IS
from proposals.Hamiltonian import HMC_proposal
from autograd import elementwise_grad as egrad
from RECYCLING import ESS_Recycling

class SMC_HMC():

    """
    Description
    -----------
    A base class for an SMC sampler with a fixed length Hamiltonian proposal.
    Estimates of the mean and variance / covariance matrix associated with a 
    specific iteration are reported in mean_estimate and var_estimate respectively. 
    Recycled estimates of the mean and variance / covariance matrix are reported in 
    mean_estimate_rc and var_estimate_rc respectively (when recycling is active). 

    Parameters
    ----------
    N : no. of samples generated at each iteration

    D : dimension of the target distribution

    p : target distribution instance

    q0 : initial proposal instance

    K : no. iterations to run

    h: Step-size of the Leapfrog method

    steps: Total number of steps before stopping

    Cov: Scale of the diagonal matrix to generate samples for the initial momentum distribution

    optL : approximation method for the optimal L-kernel. Can be either
        'gauss' or 'monte-carlo' (representing a Gaussian approximation
        or a Monte-Carlo approximation respectively).

    rc_scheme : option to have various recycling schemes (or none) Can 
        currently be 'ESS_Recycling' (which aims to maximise the effective
        sample size) or 'None' (which is currently the default). 

    verbose : option to print various things while the sampler is running

    Methods
    -------

    generate_samples : runs the SMC sampler to generate weighted
        samples from the target.

    update_weights : updates the log weight associated with each sample
        i.e. evaluates the incremental weights.

    Author
    ------
    L.J. Devlin and P.L.Green
    """

    def __init__(self, N, D, p, q0, K, h, steps, Cov, optL, 
                 rc_scheme=None, verbose=False):

        # Assign variables to self
        self.N = N
        self.D = D
        self.p = p
        self.q0 = q0
        self.K = K
        self.optL = optL
        self.verbose = verbose
        self.T=h*steps  # 'Time' that leapfrog will simulate over
        self.q = HMC_proposal(self.D, p, h, steps, Cov)
        
        # Initialise recycling scheme. For now we just have one,
        # but we might add some more later!
        if rc_scheme == 'ESS_Recycling':
            self.rc = ESS_Recycling(self.D)
        elif rc_scheme == None:
            self.rc = None

    def generate_samples(self):

        """
        Description
        -----------
        Run SMC sampler to generate weighted samples from the target.

        """

        
        # Initialise arrays to store new position, velocity and the gradient 
        # at the initial position 
        x_new = np.zeros([self.N, self.D])
        v_new = np.zeros([self.N, self.D])
        grad_x = np.zeros([self.N, self.D]) 

        # Initilise estimates of target mean and covariance matrix,
        # where 'rc' represents the overall estimate (i.e. after
        # recyling).
        self.mean_estimate = np.zeros([self.K, self.D])
        self.mean_estimate_rc = np.zeros([self.K, self.D])
        if self.D == 1:
            self.var_estimate = np.zeros([self.K, self.D])
            if self.rc:
                self.var_estimate_rc = np.zeros([self.K, self.D])
        else:
            self.var_estimate = np.zeros([self.K, self.D, self.D])
            if self.rc:
                self.var_estimate_rc = np.zeros([self.K, self.D, self.D])

        # Used to record the effective sample size and the points
        # where resampling occurred.
        self.Neff = np.zeros(self.K)
        self.resampling_points = np.array([])

        # Sample x from the prior and find initial evaluations of the
        # target and the prior. Note that, by default, we keep the log
        # weights vertically stacked         
        x = np.vstack(self.q0.rvs(size=self.N))
        
        p_logpdf_x = np.vstack(self.p.logpdf(x))
        p_log_q0_x = np.vstack(self.q0.logpdf(x))

        # Find weights of prior samples
        logw = p_logpdf_x - p_log_q0_x

        # Main sampling loop
        for self.k in range(self.K):

            if self.verbose:
                print('\nIteration :', self.k)

            # Find normalised weights and realise estimates
            wn = IS.normalise_weights(logw)
            (self.mean_estimate[self.k],
             self.var_estimate[self.k]) = IS.estimate(x, wn, self.D, self.N)

            # Recycling scheme
            if self.rc:
                (self.mean_estimate_rc[self.k], 
                 self.var_estimate_rc[self.k]) = self.rc.update_estimate(wn, self.k, 
                                                                         self.mean_estimate, 
                                                                         self.var_estimate)

            # Record effective sample size at kth iteration
            self.Neff[self.k] = 1 / np.sum(np.square(wn))

            # Resample if effective sample size is below threshold
            if self.Neff[self.k] < self.N/2:

                self.resampling_points = np.append(self.resampling_points,
                                                   self.k)
                x, p_logpdf_x, wn = IS.resample(x, p_logpdf_x, wn, self.N)
                logw = np.log(wn)


            #  Sample v from prior to start trajectories. The velocity is sampled
            #  from a normal distribution with zero mean and covariance of 'covar'.
            #  We do this here so we have easy access to them for later in weight updates for
            #  the forwards proposal
            v = np.vstack(self.q.v_rvs(size=self.N))
        
            # Importance sampling step, calculate gradient to start Leapfrog. Notice:
            # to pass parameters to the HMC proposal we package the positions, velocity
            # and intial gradient evaluations together so we follow convnetion with the SMC_templates
            # which expect a single parameter as input. We also do the initial gradient evaliation
            # here since we can use them later for the monte-carlo L-kernel approach.
            for i in range(self.N):
                grad_x[i] = egrad(self.p.logpdf)(x[i])
                Leapfrog_params = np.vstack([x[i], v[i], grad_x[i]])
                x_new[i], v_new[i] = self.q.rvs(x_cond=Leapfrog_params)

            # Make sure evaluations of likelihood are vectorised
            p_logpdf_x_new = self.p.logpdf(x_new)

            # Update log weights
            logw_new = self.update_weights(x, x_new, logw, p_logpdf_x,
                                           p_logpdf_x_new, v, v_new, grad_x)

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


        # Final quantities to be assigned to self
        self.x = x
        self.logw = logw


    def update_weights(self, x, x_new, logw, p_logpdf_x,
                       p_logpdf_x_new, v, v_new, grad_x):
        """
        Description
        -----------
        Used to update the log weights of a new set of samples, using the
            weights of the samples from the previous iteration. This is
            either done using a Gaussian approximation or a Monte-Carlo
            approximation of the opimal L-kernel. For a Hamiltonian proposal,
            the forwards and backwards kernel are parameterised by the velocity 
            distributions (see https://arxiv.org/abs/2108.02498).

        Parameters
        ----------
        x : samples from the previous iteration

        v : velocity samples from the start of the trajectory

        x_new : samples from the current iteration

        v_new : velocity samples from the current iteration

        grad_x : gradient value at x

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

            # Collect v_new and x_new together into X
            X = np.hstack([-v_new, x_new])

            # Directly estimate the mean and covariance matrix of X
            mu_X = np.mean(X, axis=0)
            cov_X = np.cov(np.transpose(X))

            # Find mean of the joint distribution (p(v_-new, x_new))
            mu_negvnew, mu_xnew = mu_X[0:self.D], mu_X[self.D:2 * self.D]

            # Find covariance matrix of joint distribution (p(-v_new, x_new))
            (cov_negvnew_negv,
             cov_negvnew_xnew,
             cov_xnew_negvnew,
             cov_xnew_xnew) = (cov_X[0:self.D, 0:self.D],
                               cov_X[0:self.D, self.D:2 * self.D],
                               cov_X[self.D:2 * self.D, 0:self.D],
                               cov_X[self.D:2 * self.D, self.D:2 * self.D])

            # Define new L-kernel
            def L_logpdf(negvnew, x_new):

                # Mean of approximately optimal L-kernel
                mu = (mu_negvnew + cov_negvnew_xnew @ np.linalg.inv(cov_xnew_xnew) @
                      (x_new - mu_xnew))

                # Variance of approximately optimal L-kernel
                cov = (cov_negvnew_negv - cov_negvnew_xnew @
                       np.linalg.inv(cov_xnew_xnew) @ cov_xnew_negvnew)

                # Add ridge to avoid singularities
                cov += np.eye(self.D) * 1e-6

                # Log det covariance matrix
                sign, logdet = np.linalg.slogdet(cov)
                log_det_cov = sign * logdet

                # Inverse covariance matrix
                inv_cov = np.linalg.inv(cov)

                # Find log pdf
                logpdf = (-0.5 * log_det_cov -
                          0.5 * (negvnew - mu).T @ inv_cov @ (negvnew - mu))

                return logpdf

            # Find new weights, Backwards L kerenl parameterised on (-v_new, x_new)
            for i in range(self.N):
                logw_new[i] = (logw[i] +
                               p_logpdf_x_new[i] -
                               p_logpdf_x[i] +
                                L_logpdf(-v_new[i], x_new[i]) -
                               self.q.logpdf(v[i]))

        # Use Monte-Carlo approximation of the optimal L-kernel
        if self.optL == 'monte-carlo':

            for i in range(self.N):

                # Initialise what will be the denominator of our
                # weight-update equation
                den = np.zeros(1)
                
                # Realise Monte-Carlo estimate of denominator
                for j in range(self.N):
                    
                    # Calculate approx velocity to move from x^j to x^i
                    # This comes about by taking a low order truncation of a 
                    # Taylor expansion of approximating x(t) from x(0) for t>0
                    v_other= (1/self.T)*(x_new[i]-x[j]) - (self.T/2)*grad_x[j]
                    
                    den+=self.q.pdf(v_other)
                                
                den /= self.N

                # Calculate new log-weight
                logw_new[i] = p_logpdf_x_new[i] - np.log(den)


        # Use the forwards proposal as the L-kernel
        if self.optL == 'forwards-proposal':
            # Find new weights
            for i in range(self.N):
                logw_new[i] = (logw[i] +
                               p_logpdf_x_new[i] -
                               p_logpdf_x[i] +
                               self.q.logpdf(-v_new[i]) - 
                               self.q.logpdf(v[i]))
                        
        return logw_new
