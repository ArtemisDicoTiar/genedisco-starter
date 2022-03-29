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
    --acquisition_function_path=./src/functions/BALDTopTeacherSelection.py \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=32 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
"""


def softmax_temperature(x, temperature=1):
    """Computes softmax probabilities from unnormalized values

    Args:

        x: array-like list of energy values.
        temperature: a positive real value.

    Returns:
        outputs: ndarray or list (dependin on x type) that is
            exp(x / temperature) / sum(exp(x / temperature)).
    """
    if isinstance(x, list):
        y = np.array(x)
    else:
        y = x
    y = np.exp(y / temperature)
    out_np = scipy.special.softmax(y)
    if any(np.isnan(out_np)):
        raise ValueError("Temperature is too extreme.")
    if isinstance(x, list):
        return [out_item for out_item in out_np]
    else:
        return out_np


class BALDTopTeacher(BaseBatchAcquisitionFunction):
    teachers = []

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 last_model: AbstractBaseModel = None,
                 ) -> List:
        U__back_slash__S = dataset_x.subset(available_indices)

        self.teachers.append(last_model)
        best_model = self.get_best_model(U__back_slash__S)

        avail_dataset_x = dataset_x.subset(available_indices)
        model_pedictions = best_model.predict(avail_dataset_x, return_std_and_margin=True)

        if len(model_pedictions) != 3:
            raise TypeError("The provided model does not output uncertainty.")

        pred_mean, pred_uncertainties, _ = model_pedictions

        if len(pred_mean) < select_size:
            raise ValueError("The number of query samples exceeds"
                             "the size of the available data.")

        # np.flip => [::-1]
        numerical_selected_indices = np.flip(
            # np.argsort => index of sorted by value.
            np.argsort(pred_uncertainties)

        )[:select_size]  # until selection size
        selected_indices = [available_indices[i] for i in numerical_selected_indices]

        return selected_indices

    def get_best_model(self, dataset):
        scores = []
        for teacher in self.teachers:
            preds = teacher.predict(dataset)
            _, real, _ = teacher.get_outputs(dataset)
            scores.append(self.get_accuracy(preds[0], real.detach().tolist()))

        best_idx = np.argmax(np.array(scores))
        return self.teachers[best_idx]

    @staticmethod
    def get_accuracy(pred, real):
        return r2_score(pred, real)
