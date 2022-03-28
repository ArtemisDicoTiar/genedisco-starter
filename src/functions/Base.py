from typing import AnyStr, List

import numpy as np
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from slingpy import AbstractDataSource, AbstractBaseModel



"""
active_learning_loop  \
    --cache_directory=./genedisco_cache \
    --output_directory=./genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="custom" \
    --acquisition_function_path=./src/functions/Base.py \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=16 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
"""


class RandomFunction(BaseBatchAcquisitionFunction):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 model: AbstractBaseModel = None,
                 ) -> List:
        """
        Nominate experiments for the next learning round.

        Args:
            dataset_x: The dataset containing all training samples.
                slingpy.data_access.data_sources.composite_data_source.CompositeDataSource

            batch_size: Size of the batch to acquire.
            available_indices: The list of the indices (names) of the samples not
               chosen in the previous rounds.
            last_selected_indices: The set of indices selected in the previous
               cycle. (S_t)
            last_model: The prediction model trained by labeled samples chosen so far.
                when bayesian_mlp: genedisco.models.meta_models.PytorchMLPRegressorWithUncertainty
        Returns:
           A list of indices (names) of the samples chosen for the next round.
        """
        pass
