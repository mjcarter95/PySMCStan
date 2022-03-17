import os
import json
import numpy as np

from pathlib import Path
from pybindstan.model import Model

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def read_data(model_name):
    data_dir = f"{ROOT_DIR}/stan_models/{model_name}/{model_name}.json"
    with open(data_dir) as json_file:
        data = json.loads(json_file.read())
        return data


def read_model(model_name):
    model_dir = f"{ROOT_DIR}/stan_models/{model_name}/{model_name}.stan"
    with open(model_dir, "r") as f:
        stan_model = f.read()
        return stan_model


class StanModel:
    def __init__(self, model_name, stan_code, data={}):
        self.stan_model = Model(model_name=model_name, program_code=stan_code, data=data)
        self.stan_model.compile()
        self.D = self.stan_model.n_pars()
        self.constrained_D = len(self.stan_model.constrained_param_names())

    def logpdf(self, upar, adjust_transform=True):
        N = upar.shape[0]
        p_logpdf_x_new = np.zeros(N)
        for i in range(N):
            p_logpdf_x_new[i] = self.stan_model.log_prob(upar[i])
        return p_logpdf_x_new
    
    def log_prob_grad(self, upar, adjust_transform=True):
        return self.stan_model.log_prob_grad(upar, adjust_transform)

    def constrain_pars(self, upar, include_tparams=True, include_gqs=True):
        constrained_estimates = np.zeros([upar.shape[0], self.constrained_D])
        for i in range(upar.shape[0]):
            constrained_estimates[i] = self.stan_model.constrain_pars(upar[i])
        return constrained_estimates

    def unconstrain_pars(self, cpar):
        return self.unconstrain_pars(cpar)
