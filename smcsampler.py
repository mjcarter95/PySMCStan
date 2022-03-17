import numpy as np
import matplotlib.pyplot as plt

from normal import Normal


class SMCSampler:
    def __init__(self, K, N, num_steps, step_size, target):
        self.target = target
        self.K = K
        self.N = N
        self.num_steps = num_steps
        self.step_size = step_size
        self.D = self.target.n_pars()
        cparam_names = self.target.constrained_param_names()
        if cparam_names is None:
            self.D_constrained = self.D
        else:
            self.D_constrained = len(cparam_names)

        # Outputs
        self.mean_estimate = np.zeros((self.K, self.D))
        self.mean_estimate_constrained = np.zeros((self.K, self.D_constrained))
        self.recycled_mean_estimate = np.zeros((self.K, self.D_constrained))
        self.ess = np.zeros((self.K,))
        self.log_likelihood = np.zeros((self.K,))
        self.recycling_constant = np.zeros((self.K,))

    def normalise_weights(self, logw):
        indices = ~np.isneginf(logw)
        logw[indices] = logw[indices] - np.max(logw[indices])
        weights = np.exp(logw)
        log_likelihood = np.log(np.sum(weights))
        wn = weights / np.sum(weights)
        return wn, log_likelihood

    def estimate(self, x, wn):
        mean = wn.T @ x
        return mean

    def leapfrog(self, x_cond, v_cond):
        """ TO DO: Replace RW with HMC proposal
        """
        grad = self.target.log_prob_grad(x_cond)
        v_prime = np.add(v_cond, np.multiply(grad, (self.step_size / 2)))
        x_prime = np.add(x_cond, self.step_size * v_prime)
        grad = self.target.log_prob_grad(x_prime)
        v_prime = np.add(v_prime, np.multiply(grad, (self.step_size / 2)))
        return x_prime, v_prime

    def run(self):
        # Sample proposal
        q = Normal(
            self.D,
            np.zeros(self.D),
            np.eye(self.D)
        )
        x = q.sample(self.N)
        x_new = np.zeros([self.N, self.D])

        # Momentum proposal for Hamiltonian
        p = Normal(
            self.D,
            np.zeros(self.D),
            np.eye(self.D)
        )
        v = p.sample(self.N)
        v_new = np.zeros([self.N, self.D])

        logw = np.vstack(np.zeros(self.N))
        logw_new = np.vstack(np.zeros(self.N))

        # Calculate initial weights
        for i in range(self.N):
            logw[i] = self.target.log_prob(x[i]) - q.log_prob(x[i])

        # Run SMC Sampler
        for k in range(self.K):
            print(f"Iteration {k}")
            wn, self.log_likelihood[k] = self.normalise_weights(logw)
            self.mean_estimate[k] = self.estimate(x, wn)
            self.mean_estimate_constrained[k] = self.target.constrain_pars(self.mean_estimate[k])

            # Calculate ESS and resample if required
            self.ess[k] = 1 / np.sum(np.square(wn))
            if self.ess[k] < self.N / 2:
                # From https://github.com/plgreenLIRU/SMC_approx_optL
                i = np.linspace(0, self.N-1, self.N, dtype=int)
                i_new = np.random.choice(i, self.N, p=wn[:, 0])
                wn_new = np.ones(self.N) / self.N
                x = x[i_new]
                logw = np.log(np.ones(self.N)  / self.N)

            # Propagate samples and calculate new weights (set L=q)
            for i in range(self.N):
                x_new[i] = x[i] + np.random.randn(self.D)
                logw_new[i] = (logw[i]
                               + self.target.log_prob(x_new[i])
                               - self.target.log_prob(x[i]))

            x = x_new.copy()
            v = v_new.copy()
            logw = logw_new.copy()

    def generate_plots(self):
        return
