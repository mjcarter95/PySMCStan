import os
import json

from pathlib import Path
from smcsampler import SMCSampler
from pybindstan.model import Model as StanModel

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


def main():
    model_name = "student_t_k"
    model_data = read_data(model_name)
    model_code = read_model(model_name)
    target = StanModel(
        model_name,
        model_code,
        model_data
    )
    target.compile()

    K = 100
    N = 200
    num_steps = 5
    step_size = 0.5

    smcs = SMCSampler(K, N, num_steps, step_size, target)
    smcs.run()
    print(smcs.mean_estimate_constrained[-1])


if __name__ == "__main__":
    main()
