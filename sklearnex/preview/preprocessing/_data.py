# ===============================================================================
# Copyright Contributors to the oneDAL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

from sklearn.preprocessing import MinMaxScaler as _sklearn_MinMaxScaler

from onedal.statistics import IncrementalBasicStatistics as onedal_IncrementalBasicStatistics

from ...base import oneDALEstimator
from ..._device_offload import dispatch
from ..._utils import PatchingConditionsChain
from ...utils._array_api import enable_array_api, get_namespace
from ...utils.validation import validate_data


@enable_array_api
@control_n_jobs(decorated_methods=["fit", "partial_fit", "transform", "inverse_transform"])
class IncrementalMinMaxScaler(oneDALEstimator, _sklearn_MinMaxScaler):
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        super().__init__(feature_range=feature_range, copy=copy, clip=clip)
        self._need_to_finalize = False

    def fit(self, X, y=None):
        self._reset()
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_MinMaxScaler.fit,
            },
            X,
        )
        return self

    def partial_fit(self, X, y=None):
        dispatch(
            self,
            "partial_fit",
            {
                "onedal": self.__class__._onedal_partial_fit,
                "sklearn": _sklearn_MinMaxScaler.partial_fit,
            },
            X,
        )
        return self

    def transform(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "transform",
            {
                "onedal": self.__class__._onedal_transform,
                "sklearn": _sklearn_MinMaxScaler.transform,
            },
            X,
        )

    def inverse_transform(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "inverse_transform",
            {
                "onedal": self.__class__._onedal_inverse_transform,
                "sklearn": _sklearn_MinMaxScaler.inverse_transform,
            },
            X,
        )

    # --- oneDAL methods ---

    def _onedal_fit(self, X, queue=None):
        self._onedal_estimator = onedal_IncrementalBasicStatistics(result_options=["min", "max"])
        self._onedal_estimator.partial_fit(X, queue=queue)
        self._onedal_estimator.finalize_fit()
        self._copy_onedal_attributes()
        return self

    def _onedal_partial_fit(self, X, queue=None):
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = onedal_IncrementalBasicStatistics(result_options=["min", "max"])
        self._onedal_estimator.partial_fit(X, queue=queue)
        self._need_to_finalize = True
        return self

    def _onedal_finalize_fit(self):
        if self._need_to_finalize:
            self._onedal_estimator.finalize_fit()
            self._copy_onedal_attributes()
            self._need_to_finalize = False

    def _copy_onedal_attributes(self):
        # Copy min_, max_, data_min_, data_max_, data_range_, n_samples_seen_ from onedal estimator
        self.data_min_ = self._onedal_estimator.min_
        self.data_max_ = self._onedal_estimator.max_
        self.data_range_ = self.data_max_ - self.data_min_
        fr_min, fr_max = self.feature_range
        self.scale_ = (fr_max - fr_min) / np.where(self.data_range_ == 0, 1, self.data_range_)
        self.min_ = fr_min - self.data_min_ * self.scale_
        self.n_samples_seen_ = getattr(self._onedal_estimator, "n_samples_seen_", None)

    def _onedal_transform(self, X, queue=None):
        self._onedal_finalize_fit()
        X = validate_data(self, X, copy=self.copy, dtype=[np.float64, np.float32], reset=False)
        X = X * self.scale_ + self.min_
        if self.clip:
            X = np.clip(X, self.feature_range[0], self.feature_range[1])
        return X

    def _onedal_inverse_transform(self, X, queue=None):
        self._onedal_finalize_fit()
        X = X - self.min_
        X = X / self.scale_
        return X

    def _onedal_cpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.preprocessing.{self.__class__.__name__}.{method_name}"
        )
        X = data[0]
        patching_status.and_conditions([
            (not sp.issparse(X), "Sparse input is not supported"),
        ])
        return patching_status

    def _onedal_gpu_supported(self, method_name, *data):
        # Similar to CPU, but check for GPU support if needed
        return self._onedal_cpu_supported(method_name, *data)
