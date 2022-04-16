import os
import json
import numpy as np

from pathlib import Path
from pybindstan.model import Model
from SMC_TEMPLATES import Target_Base

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

"""
Description
-----------
This module contains functions for reading JSON data associated
with a Stan model, the Stan code associated with a Stan model and
a class (StanModel) which wraps the PyBindStan interface to Stan.
"""


def read_data(model_name):
    """ Read JSON data associated with a Stan model
    """
    data_dir = f"{ROOT_DIR}/stan_models/{model_name}/{model_name}.json"
    with open(data_dir) as json_file:
        data = json.loads(json_file.read())
        return data


def read_model(model_name):
    """ Read Stan model code
    """
    model_dir = f"{ROOT_DIR}/stan_models/{model_name}/{model_name}.stan"
    with open(model_dir, "r") as f:
        stan_model = f.read()
        return stan_model


class StanModel(Target_Base):
    def __init__(self, model_name, stan_code, data={}):
        """
        Description
        -----------
        A wrapper for PyBindStan - a Python interface to the Stan probabilistic
        programming language.

        Parameters
        ----------
        model_name: name of the Stan model

        stan_code: Stan program code to compile

        data: data for the Stan model

        Methods
        -------

        logpdf : return the log probability of the unconstrained parameters.

        logpdfgrad : return the gradient of the log posterior evaluated at
            the unconstrained parameters.

        constrain_pars :  transform a sequence of unconstrained parameters
            to their defined support, optionally including transformed parameters
            and generated quantities.

        unconstrain_pars : reads constrained parameter values from their specified
            context and returns a sequence of unconstrained parameter values.
        
        print_ests: returns a dicationary where the keys are the parameter names and
            the values are a list of parameter estimates per iteration.

        Author
        ------
        M.J.Carter
        """

        self.stan_model = Model(model_name=model_name, program_code=stan_code, data=data)
        self.stan_model.compile()
        self.D = self.stan_model.n_pars()
        self.param_names = self.stan_model.constrained_param_names()
        if self.param_names is None:
            self.param_names = self.stan_model.unconstrained_param_names()
            self.constrained_D = self.D
        else:
            self.constrained_D = len(self.param_names)
        self.mean = np.zeros(self.D)
        self.cov = np.eye(self.D)

    def logpdf(self, upar, adjust_transform=True):
        N = upar.shape[0]
        p_logpdf_x_new = np.zeros(N)
        for i in range(N):
            p_logpdf_x_new[i] = self.stan_model.log_prob(upar[i])
        return p_logpdf_x_new
    
    def logpdfgrad(self, upar, adjust_transform=True):
        return self.stan_model.log_prob_grad(upar, adjust_transform)

    def constrain_mean(self, upar, include_tparams=True, include_gqs=True):
        constrained_estimates = np.zeros([upar.shape[0], self.D])
        transformed_estimates = np.zeros([upar.shape[0], (self.constrained_D - self.D)])
        for i in range(upar.shape[0]):
            cpar = self.stan_model.constrain_pars(upar[i])
            constrained_estimates[i] = cpar[:self.D]
            transformed_estimates[i] = cpar[self.D:]
        return constrained_estimates, transformed_estimates

    def constrain_var(self, upar, include_tparams=True, include_gqs=True):
        constrained_estimates = np.zeros([upar.shape[0], self.D])
        transformed_estimates = np.zeros([upar.shape[0], (self.constrained_D - self.D)])
        for i in range(upar.shape[0]):
            cpar = self.stan_model.constrain_pars(np.diag(upar[i]))
            constrained_estimates[i] = cpar[:self.D]
            transformed_estimates[i] = cpar[self.D:]
        return constrained_estimates, transformed_estimates

    def unconstrain_pars(self, cpar):
        return self.unconstrain_pars(cpar)

    def print_ests(self, K, cpar, tpar=None, verbose=False):
        param_estimates = {}
        for d in range(self.constrained_D):
            if not self.param_names[d] in param_estimates.keys():
                param_estimates[self.param_names[d]] = np.zeros(K)
            for k in range(K):
                if d < self.D:
                    param_estimates[self.param_names[d]][k] = cpar[k][d]
                else:
                    param_estimates[self.param_names[d]][k] = tpar[k][d-self.D]
            if verbose:
                print(f"{self.param_names[d]}: {param_estimates[self.param_names[d]]}")
        return param_estimates
