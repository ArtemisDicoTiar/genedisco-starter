from typing import List, AnyStr

import numpy as np
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from slingpy import AbstractDataSource, AbstractBaseModel


class BADGE(BaseBatchAcquisitionFunction):

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel) -> List:
        """
        Nominate experiments for the next learning round.

        Args:
            dataset_x: The dataset containing all training samples.
            batch_size: Size of the batch to acquire.
            available_indices: The list of the indices (names) of the samples not
                chosen in the previous rounds.
            last_selected_indices: The set of indices selected in the previous
                cycle. (S_t)
            last_model: The prediction model trained by labeled samples chosen so far.

        Returns:
            A list of indices (names) of the samples chosen for the next round.
        """
        U__back_slash__S = dataset_x.subset(available_indices)
        gradient_embedding: np.ndarray = last_model.get_gradient_embedding(U__back_slash__S).numpy()
        S_t = kmeans_algorithm(gradient_embedding)
        return S_t



