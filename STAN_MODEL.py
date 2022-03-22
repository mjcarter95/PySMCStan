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

        constrain_pars :  Transform a sequence of unconstrained parameters
            to their defined support, optionally including transformed parameters
            and generated quantities.

        unconstrain_pars : reads constrained parameter values from their specified
            context and returns a sequence of unconstrained parameter values.

        Author
        ------
        M.J.Carter
        """
        self.stan_model = Model(model_name=model_name, program_code=stan_code, data=data)
        self.stan_model.compile()
        self.D = self.stan_model.n_pars()
        cparam_names = self.stan_model.constrained_param_names()
        if cparam_names is None:
            self.constrained_D = self.D
        else:
            self.constrained_D = len(cparam_names)
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

    def constrain_pars(self, upar, include_tparams=True, include_gqs=True):
        constrained_estimates = np.zeros([upar.shape[0], self.constrained_D])
        for i in range(upar.shape[0]):
            constrained_estimates[i] = self.stan_model.constrain_pars(upar[i])
        return constrained_estimates

    def unconstrain_pars(self, cpar):
        return self.unconstrain_pars(cpar)
