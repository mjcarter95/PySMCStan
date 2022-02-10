import autograd.numpy as np
import importance_sampling as IS
from SMC_TEMPLATES import Q_Base


class SMC():

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

    generate_samples : runs the SMC sampler to generate weighted
        samples from the target.

    update_weights : updates the log weight associated with each sample
        i.e. evaluates the incremental weights.

    Author
    ------
    P.L.Green and L.J. Devlin
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

        # Can either have a user-defined proposal or random walk
        # proposal in this implementation
        if(isinstance(proposal, Q_Base)):
            self.q = proposal
            self.proposal='user'
        elif(proposal == 'rw'):
            from proposals.random_walk import random_walk_proposal
            self.q = random_walk_proposal(self.D)
            self.proposal = 'rw'

    def generate_samples(self):

        """
        Description
        -----------
        Run SMC sampler to generate weighted samples from the target.

        """

        # Initialise arrays for storing samples (x_new) and for
        # the recycling coefficients (lr)
        x_new = np.zeros([self.N, self.D])
        lr = np.array([])

        # Initilise estimates of target mean and covariance matrix,
        # where 'EES' represents the overall estimate (i.e. after
        # recyling)
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
        p_log_q0_x = np.vstack(self.q0.logpdf(x))

        # Find log-weights of prior samples
        logw = p_logpdf_x - p_log_q0_x

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
            #region

            # Find the new values of lr (equation (18) in
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
            #endregion
            # --------------------------------------------------------------- #

            # Record effective sample size at kth iteration
            self.Neff[self.k] = 1 / np.sum(np.square(wn))

            # Resample if effective sample size is below threshold
            if self.Neff[self.k] < self.N/2:

                self.resampling_points = np.append(self.resampling_points,
                                                   self.k)
                x, p_logpdf_x, wn = IS.resample(x, p_logpdf_x, wn, self.N)
                logw = np.log(wn)

            # Propose new samples
            for i in range(self.N):
                x_new[i] = self.q.rvs(x_cond=x[i])

            # Make sure evaluations of likelihood are vectorised
            p_logpdf_x_new = self.p.logpdf(x_new)

            # Update log weights
            logw_new = self.update_weights(x, x_new, logw, p_logpdf_x,
                                           p_logpdf_x_new)

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
                       p_logpdf_x_new):
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
        # (see Section 4.1 of https://www.sciencedirect.com/science/article/pii/S0888327021004222
        #  or Section 4.1 of https://arxiv.org/pdf/2004.12838.pdf)
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
                               self.q.logpdf(x_new[i], x[i]))

        # Use Monte-Carlo approximation of the optimal L-kernel
        # (not published at the moment but see 
        # https://www.overleaf.com/project/6130ff176124735112a885b3)
        if self.optL == 'monte-carlo':
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
