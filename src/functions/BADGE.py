import pdb
from typing import List, AnyStr, Optional, Type

import numpy as np
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction

from genedisco.models.meta_models import PytorchMLPRegressorWithUncertainty
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from slingpy import AbstractDataSource, AbstractBaseModel


"""
active_learning_loop  \
    --cache_directory=./genedisco_cache \
    --output_directory=./genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="custom" \
    --acquisition_function_path=./src/functions/BADGE.py \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=20 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
"""


class BADGE(BaseBatchAcquisitionFunction):

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: PytorchMLPRegressorWithUncertainty) -> List:
        U__back_slash__S = dataset_x.subset(available_indices)
        gradient_embedding: np.ndarray = last_model.get_gradient_embedding(U__back_slash__S).numpy()
        S_t = self.kmeans_algorithm(gradient_embedding, batch_size)
        # print(U__back_slash__S.get_shape())
        # print(S_t)
        # print(batch_size)
        selected_queries = [available_indices[idx] for idx in S_t]
        return selected_queries


    """
    For kmeans algorithms provided by sklearn, 
    they are designed to perform clustering.
    Therefore, the algorithms cannot fit into BADGE
    as it requires initialisation scheme of kmeans++.
    """
    @staticmethod
    def kmeans_algorithm(gradient_embedding, k) -> List:
        # kmeans++
        # ref: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        # ref (code):
        #   https://github.com/JordanAsh/badge/blob/284e6dee4e42fb05d6febc3645540368d94a6b4c/query_strategies/badge_sampling.py

        # np.linalg -> linear algebra methods
        # norm(_, 2) -> L2 norm (Euclid norm) -> sqrt(sum(x^2))
        ind = np.argmax([np.linalg.norm(s, 2) for s in gradient_embedding])
        mu = [gradient_embedding[ind]]
        indsAll = [ind]
        centInds = [0.] * len(gradient_embedding)
        cent = 0
        while len(mu) < k:
            if len(mu) == 1:
                D2 = pairwise_distances(gradient_embedding, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(gradient_embedding, [mu[-1]]).ravel().astype(float)
                for i in range(len(gradient_embedding)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(gradient_embedding[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

