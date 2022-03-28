"""
Copyright 2021 Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from typing import List, AnyStr, Type, Optional

import numpy as np
from genedisco.models.abstract_meta_models import AbstractMetaModel
from sklearn import ensemble
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.models.pickleable_base_model import PickleableBaseModel
from slingpy.models.sklearn_model import SklearnModel


class SklearnRandomForestRegressor(AbstractMetaModel, SklearnModel):
    """The random forest regressor model from sklean (non-deep models)"""

    def __init__(self,
                 base_module: ensemble.BaseEnsemble,
                 num_target_samples: Optional[int] = None):
        """The wrapping model for scikit learn ensemble regressors.

        The wrapping model augments the base_module with the capability to
        output uncertainty along with the average prediction.

        Args:
            base_module: The base forest ensemble module. It can be from
            {RandomForestRegressor, ExtraTreesRegressor}
        """

        SklearnModel.__init__(self, base_module)
        self.n_estimators = base_module.n_estimators
        self.num_target_samples = num_target_samples

    def save(self, file_path: AnyStr):
        AbstractMetaModel.save(self, file_path)

    def save_folder(self, save_folder_path, overwrite=True):
        file_path = os.path.join(save_folder_path, "model.pickle")
        PickleableBaseModel.save(self, file_path)

    @classmethod
    def load_folder(cls: Type["TarfileSerialisationBaseModel"], save_folder_path: AnyStr) -> AbstractBaseModel:
        file_path = os.path.join(save_folder_path, "model.pickle")
        model = PickleableBaseModel.load(file_path)
        return model

    def get_samples(self, x: np.ndarray, k: Optional[int] = 1):
        k = self.n_estimators or k
        if k > self.n_estimators:
            raise ValueError("The number of requested samples cannot exceed" +
                             " the number of estimators in ensemble models.")
        else:
            random_indices = np.random.randint(self.n_estimators, size=k)
            y_samples = [self.model.estimators_[i].predict(x)
                         for i in random_indices]
            y_samples = np.swapaxes(y_samples, 0, 1)
        return y_samples

    def get_model_prediction(self, data, return_std_and_margin):
        if return_std_and_margin:
            y_samples = self.get_samples(data, self.num_target_samples)
            if y_samples.ndim == 3 and y_samples.ndimsy_samples.shape[-1] > 1:
                raise NotImplementedError("Output uncertainty is only" +
                                          "implemented for 1D output.")
            else:
                y_stds = np.std(y_samples, axis=1)
                y_margins = np.max(y_samples, axis=1) - np.min(y_samples, axis=1)
                y_preds = self.model.predict(data)
                return [y_preds, y_stds, y_margins]
        else:
            y_pred = self.model.predict(data)
            return y_pred

    def predict(self,
                dataset_x: AbstractDataSource,
                batch_size: int = 256,
                return_std_and_margin: bool = False) -> List[np.ndarray]:
        """
        Args:
            dataset_x: Input dataset to be evaluated.
            batch_size:
            return_std_and_margin: If True, return the epistemic uncertainty of the output.

        Returns: If return_std_and_margin is True, returns ([output_means], [output_stds]).
                        Otherwise, returns [output_means]
        """
        if self.model is None:
            self.model = self.build()

        available_indices = dataset_x.get_row_names()
        if return_std_and_margin:
            all_ids, y_preds, y_stds, y_trues, y_margins = [], [], [], [], []
        else:
            all_ids, y_preds, y_trues = [], [], []

        while len(available_indices) > 0:
            current_indices = available_indices[:batch_size]
            available_indices = available_indices[batch_size:]

            data = self.merge_strategy_x.resolve(
                dataset_x.subset(list(current_indices)).get_data())[0]

            if return_std_and_margin:
                y_pred, y_std, y_margin = self.get_model_prediction(data, return_std_and_margin=True)
                y_preds.append(y_pred)
                y_stds.append(y_std)
                y_margins.append(y_margin)
            else:
                y_pred = self.get_model_prediction(data, return_std_and_margin=False)
                y_preds.append(y_pred)
        y_preds = np.concatenate(y_preds, axis=0)
        if return_std_and_margin:
            y_stds = np.concatenate(y_stds, axis=0)
            y_margins = np.concatenate(y_margins, axis=0)
            return [y_preds, y_stds, y_margins]
        else:
            return [y_preds]

    def get_hyperopt_parameter_ranges(self):
        hyperopt_ranges = {
            "max_depth": (1, 2, 3),
            "num_estimators": (10, 50, 100, 200),
        }
        return hyperopt_ranges


