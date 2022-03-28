from collections import Counter
from typing import AnyStr, List

from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from slingpy import AbstractDataSource, AbstractBaseModel

from src.functions.BADGE import BADGE
from src.functions.BALD import TopUncertainAcquisition, SoftUncertainAcquisition


class EnsembleBALDnBADGE(BaseBatchAcquisitionFunction):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 model: AbstractBaseModel = None,
                 ) -> List:
        coef = 3
        buffer_size = select_size * coef

        top_uncertain = TopUncertainAcquisition().__call__(
            dataset_x,
            buffer_size,
            available_indices,
            last_selected_indices,
            model,
        )

        soft_uncertain = SoftUncertainAcquisition().__call__(
            dataset_x,
            buffer_size,
            available_indices,
            last_selected_indices,
            model,
            temperature=0.9
        )

        badge = BADGE().__call__(
            dataset_x,
            buffer_size,
            available_indices,
            last_selected_indices,
            model,
        )

        total_concat = top_uncertain + soft_uncertain + badge
        total_count = sorted(
            Counter(total_concat).items(),
            key=lambda item: -item[1]
        )

        candidates = total_count[:select_size]
        return candidates

