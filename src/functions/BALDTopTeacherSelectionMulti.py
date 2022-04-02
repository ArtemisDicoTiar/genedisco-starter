import pickle
from collections import Counter
from pathlib import Path

import scipy
import numpy as np
from typing import AnyStr, List

from sklearn.metrics import r2_score
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction

"""
active_learning_loop  \
    --cache_directory=./genedisco_cache \
    --output_directory=./genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="custom" \
    --acquisition_function_path=./src/functions/BALDTopTeacherSelectionMulti.py \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=32 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
"""


class BALDTopTeacherMulti(BaseBatchAcquisitionFunction):
    teachers = []
    selections = []

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 last_model: AbstractBaseModel = None,
                 ) -> List:
        coef = 3
        buffer_size = select_size * coef

        U__back_slash__S = dataset_x.subset(available_indices)

        self.teachers.append(last_model)
        best_models = self.get_best_models(U__back_slash__S)

        model_predictions = [
            best_model.predict(U__back_slash__S, return_std_and_margin=True)
            for best_model in best_models
        ]

        if len(model_predictions[0]) != 3:
            raise TypeError("The provided model does not output uncertainty.")

        pred_uncertainties = [
            model_prediction[1]
            for model_prediction in model_predictions
        ]

        # np.flip => [::-1]
        numerical_selected_indices = [
            np.flip(
                # np.argsort => index of sorted by value.
                np.argsort(pred_uncertainty)

            )[:buffer_size]  # until selection size
            for pred_uncertainty in pred_uncertainties
        ]
        selected_indices = Counter([
            available_indices[idx]
            for numerical_selected_idx in numerical_selected_indices
            for idx in numerical_selected_idx
        ]).items()

        total_count = sorted(
            selected_indices,
            key=lambda item: -item[1]
        )
        candidates = total_count[:select_size]
        with Path('./selections(BALD).txt').open("wb") as f:
            pickle.dump(self.selections, f)
        with Path('./selectedQueries(BALD).txt').open("ab") as f:
            pickle.dump(list(selected_indices), f)

        return candidates

    def get_best_models(self, dataset):
        scores = []
        for teacher in self.teachers:
            preds = teacher.predict(dataset)
            _, real, _ = teacher.get_outputs(dataset)
            scores.append(self.get_accuracy(preds[0], real.detach().tolist()))

        best_idxs = np.flip(
            # np.argsort => index of sorted by value.
            np.argsort(np.array(scores))

        )
        self.selections.append(best_idxs[:3])
        return [
            self.teachers[i]
            for i in best_idxs[:3]
        ]

    @staticmethod
    def get_accuracy(pred, real):
        return r2_score(pred, real)
