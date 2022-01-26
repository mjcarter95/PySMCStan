import autograd.numpy as np
import importance_sampling as IS
from proposals.Hamiltonian import HMC_proposal
from autograd import elementwise_grad as egrad

class SMC_HMC():

    """
    Description
    -----------
    A base class for an SMC sampler with a Hamiltonian proposal.

    Parameters
    ----------
    N : no. of samples generated at each iteration

    D : dimension of the target distribution

    p : target distribution instance

    q0 : initial proposal instance

    grad_x: Gradient of the target distribution w.r.t. x

    v: Velocity distribution, initial distribution sampled from a standard Normal distribution.

    K : no. iterations to run

    h: Step-size of the Leapfrog method

    steps: Total number of steps before stopping

    Cov: scale of the diagonal matrix to generate samples from the momentum distribution

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
    L.J. Devlin and P.L.Green
    """

    def __init__(self, N, D, p, q0, K, h, steps, Cov, optL, verbose=False):

        # Assign variables to self
        self.N = N
        self.D = D
        self.p = p
        self.q0 = q0
        self.K = K
        self.optL = optL
        self.verbose = verbose
        self.T=h*steps
        self.q = HMC_proposal(self.D, p, h, steps, Cov)
        

    def generate_samples(self):

        """
        Description
        -----------
        Run SMC sampler to generate weighted samples from the target.

        """

        # Initialise arrays
        x_new = np.zeros([self.N, self.D])
        v_new = np.zeros([self.N, self.D])
        grad_x = np.zeros([self.N, self.D]) 

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

        # Sample x and v from prior and find initial evaluations of the
        # target and the prior. The velocity is sampled from a standard
        # normal distribution and used throughout the HMC proposal.
        # Note that, by default, we keep the log weights vertically stacked
         
        x = np.vstack(self.q0.rvs(size=self.N))
        v = np.vstack(self.q.v_rvs(size=self.N))

        p_logpdf_x = np.vstack(self.p.logpdf(x))
        p_q0_v = np.vstack(self.q.v_logpdf(v))

        # Find weights of prior samples
        logw = p_logpdf_x - p_q0_v

        # Main sampling loop
        for self.k in range(self.K):

            if self.verbose:
                print('\nIteration :', self.k)

            # Find normalised weights and realise estimates
            wn = IS.normalise_weights(logw)
            (self.mean_estimate[self.k],
             self.var_estimate[self.k]) = IS.estimate(x, wn, self.D, self.N)

            # --------------------------------------------------------------- #
            # EES recycling scheme #

            # Find the new values of l (equation (18) in
            # https://arxiv.org/pdf/2004.12838.pdf)
            lr = np.append(lr, np.sum(wn)**2 / np.sum(wn**2))

            # Initialise c array (also defined by equation (18) in
            # https://arxiv.org/pdf/2004.12838.pdf)
            c = np.array([])

            # Loop to recalculate the optimal c values
            for k_dash in range(self.k + 1):
                c = np.append(c, lr[k_dash] / np.sum(lr))

            # Loop to recalculate estimates of the mean
            for k_dash in range(self.k + 1):
                self.mean_estimate_EES[self.k] += (c[k_dash] *
                                                   self.mean_estimate[k_dash])

            # Loop to recalculate estimates of the variance / cov. matrix
            for k_dash in range(self.k + 1):

                # Define a 'correction' term, to account for the bias that
                # arises in the variance estimate as a result of using the
                # estimated mean
                correction = (self.mean_estimate_EES[self.k] -
                              self.mean_estimate[k_dash])**2

                # Variance estimate, including correction term
                self.var_estimate_EES[self.k] += (c[k_dash] *
                                                  (self.var_estimate[k_dash] + 
                                                   correction))

            # --------------------------------------------------------------- #

            # Record effective sample size at kth iteration
            self.Neff[self.k] = 1 / np.sum(np.square(wn))

            # Resample if effective sample size is below threshold
            if self.Neff[self.k] < self.N/2:

                self.resampling_points = np.append(self.resampling_points,
                                                   self.k)
                x, p_logpdf_x, wn = IS.resample(x, p_logpdf_x, wn, self.N)
                logw = np.log(wn)
        
            # Importance sampling step, calculate gradient to start Leapfrog
            for i in range(self.N):
                grad_x[i] = egrad(self.p.logpdf)(x[i])
                X = np.vstack([x[i], v[i], grad_x[i]])
                x_new[i], v_new[i] = self.q.rvs(x_cond=X)

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
            v = np.vstack(self.q.v_rvs(size=self.N))

        # Final quantities to be returned
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
            approximation of the opimal L-kernel.

        Parameters
        ----------
        x : samples from the previous iteration

        v : velocity samples from the previous iteration

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

            # Collect x and x_new together into X
            X = np.hstack([-v_new, x_new])

            # Directly estimate the mean and covariance matrix of X
            mu_X = np.mean(X, axis=0)
            cov_X = np.cov(np.transpose(X))

            # Find mean of the joint distribution (p(x, x_new))
            mu_x, mu_xnew = mu_X[0:self.D], mu_X[self.D:2 * self.D]

            # Find covariance matrix of joint distribution (p(-v_new, x_new))
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
                                L_logpdf(-v_new[i], x_new[i]) -
                               self.q.v_logpdf(v[i]))

        # Use Monte-Carlo approximation of the optimal L-kernel
        if self.optL == 'monte-carlo':

            for i in range(self.N):

                # Initialise what will be the denominator of our
                # weight-update equation
                den = np.zeros(1)
                
                # Realise Monte-Carlo estimate of denominator
                for j in range(self.N):
                    
                    #Calculate approx velocity to move from x^j to x^i
                    v_other= (1/self.T)*(x_new[i]-x[j]) - (self.T/2)*grad_x[j]
                    
                    den+=self.q.v_pdf(v_other)
                                
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
                               self.q.v_logpdf(-v_new[i]) - 
                               self.q.v_logpdf(v[i]))
                        
        return logw_new
