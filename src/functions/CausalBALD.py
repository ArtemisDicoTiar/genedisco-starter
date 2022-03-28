from typing import AnyStr, List

import numpy as np
import torch
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction

from scipy import stats
from slingpy import AbstractDataSource, AbstractBaseModel

_eps = 1e-7


def random(mu_0, mu_1, t, pt, temperature):
    return np.ones_like(mu_0.mean(0))


def tau(mu_0, mu_1, t, pt, temperature):
    return (mu_1 - mu_0).var(0) ** (1 / temperature)


def mu(mu_0, mu_1, t, pt, temperature):
    return (t * mu_1.var(0) + (1 - t) * mu_0.var(0)) ** (1 / temperature)


def rho(mu_0, mu_1, t, pt, temperature):
    return tau(mu_0, mu_1, t, pt, temperature) / (
            mu(mu_0, mu_1, 1 - t, pt, temperature) + _eps
    )


def mu_rho(mu_0, mu_1, t, pt, temperature):
    return mu(mu_0, mu_1, t, pt, temperature) * rho(mu_0, mu_1, t, pt, temperature)


def pi(mu_0, mu_1, t, pt, temperature):
    return t * (1 - pt) + (1 - t) * pt


def mu_pi(mu_0, mu_1, t, pt, temperature):
    return mu(mu_0, mu_1, t, pt, temperature) * pi(mu_0, mu_1, t, pt, temperature)


def sundin(mu_0, mu_1, t, pt, temperature):
    tau = mu_1 - mu_0
    gammas = np.clip(stats.norm().cdf(-np.abs(tau) / np.sqrt(2)), _eps, 1 - _eps)
    gamma = gammas.mean(0)
    predictive_entropy = stats.bernoulli(gamma).entropy()
    conditional_entropy = stats.bernoulli(gammas).entropy().mean(0)
    # it can get negative very small number because of numerical instabilities
    mi = predictive_entropy - conditional_entropy
    return mi


class CausalBALDAcquisition(BaseBatchAcquisitionFunction):
    acquisition_function = pi

    @staticmethod
    def predict_mus(preds, batch, batch_size=None):
        mu_0 = []
        mu_1 = []

        covariates = torch.cat([batch[0][:, :-1], batch[0][:, :-1]], 0)
        treatments = torch.cat(
            [
                torch.zeros_like(batch[0][:, -1:]),
                torch.ones_like(batch[0][:, -1:]),
            ],
            0,
        )
        inputs = torch.cat([covariates, treatments], -1)
        mu = preds.mean
        mus = torch.split(mu, mu.shape[0] // 2, dim=0)
        mu_0.append(mus[0])
        mu_1.append(mus[1])

        return (
            torch.cat(mu_0, 0).numpy(),
            torch.cat(mu_1, 0).numpy(),
        )

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 model: AbstractBaseModel = None,
                 ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        model_predictions = model.predict(avail_dataset_x, return_std_and_margin=True)

        # Predict pool set
        mu_0, mu_1 = self.predict_mus(model_predictions, available_indices)
        # Get acquisition scores
        scores = (
            self.acquisition_function(
                mu_0=mu_0,
                mu_1=mu_1,
                t=None,
                pt=None,
                temperature=None,
            )
        )
        # Sample acquired points
        p = scores / scores.sum()
        numerical_selected_indices = np.random.choice(
            range(len(p)),
            replace=False,
            p=p,
            size=select_size,
        )
        selected_indices = [available_indices[i] for i in numerical_selected_indices]
        return selected_indices


"""
active_learning_loop  \
    --cache_directory=./genedisco_cache \
    --output_directory=./genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="custom" \
    --acquisition_function_path=./src/functions/CausalBALD.py \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=8 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
"""
