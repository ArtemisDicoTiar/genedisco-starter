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
    --num_active_learning_cycles=16 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
"""


class BADGE(BaseBatchAcquisitionFunction):

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel) -> List:
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
        # np.argmax -> index of max value

        # 최초 centeroid로 가장 모든 값에서 멀리 떨어진 index를 찾음.
        ind = np.argmax([np.linalg.norm(s, 2) for s in gradient_embedding])
        # 최초 center의 값을 리스트에 담음.
        mu = [gradient_embedding[ind]]
        indsAll = [ind]

        # 필요한 center 인덱스 리스트로 초기화
        centInds = [0.] * len(gradient_embedding)
        cent = 0
        while len(mu) < k:
            # 최초에만 실행.
            if len(mu) == 1:
                # 두 점 (gradient_embedding과 mu 사이의 L2 거리)
                # .ravel() == .flatten()
                D2 = pairwise_distances(gradient_embedding, mu).ravel().astype(float)
            # 이후 시행.
            else:
                # 마지막에 추가한 center와 다른 point 사이의 L2 거리
                newD = pairwise_distances(gradient_embedding, [mu[-1]]).ravel().astype(float)
                # 새로 생성한 Distance와 이전 center에서 생성된 Distance를 비교 하는 loop
                for i in range(len(gradient_embedding)):
                    # 만약 이전에 생성한 거리가 새로 생성된것보다 길다면.
                    if D2[i] > newD[i]:
                        centInds[i] = cent      # 중심의 값을 cent로 변경.
                        D2[i] = newD[i]     # 새로 생성한 거리를 D2에 override.

            D2 = D2.ravel().astype(float)       # flatten

            # 편중 확률 분포
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            # scipy.stats.rv_discrete
            # random variable _ discrete으로 discrete 랜덤 변수를 생성.
            # 단 특정 분포를 따른다.
            # 아래의 경우, (0, 1, ..., D2 의 길이)의 범위에 대해 Ddist 의 확률을 리턴하게 설정.
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            # 위에서 설정한 분포에서의 첫번째 값을 가져와서
            ind = customDist.rvs(size=1)[0]
            # mu에 해당 index의 값을 추가한다.
            mu.append(gradient_embedding[ind])
            # indsAll에 해당 index를 추가한다.
            indsAll.append(ind)
            # center count ++
            cent += 1
        return indsAll


if __name__ == '__main__':
    BADGE.kmeans_algorithm([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], 3)
