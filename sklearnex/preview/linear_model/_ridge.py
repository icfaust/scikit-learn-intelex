# ==============================================================================
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
# ==============================================================================

from sklearn.base import RegressorMixin
from sklearn.linear_model._ridge import _BaseRidge, _RidgeClassifierMixin
from sklearn.lienar_model import RidgeClassifier as _sklearn_RidgeClassifier

from daal4py.sklearn._utils import sklearn_check_version

from ..._device_offload import dispatch
from ...linear_model import Ridge

# oneDAL array API enabled based on Ridge version
class RidgeClassifier(_RidgeClassifierMixin, Ridge):
    __doc__ = RidgeClassifier.__doc__
    __sklearn_tags__ = _sklearnRidgeClassifier.__sklearn_tags__

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight, Y = self._prepare_data(X, y, sample_weight, self.solver)
        if sklearn_check_version("1.2"):
            self._validate_params()

        # It is necessary to properly update coefs for predict if we
        # fallback to sklearn in dispatch
        if hasattr(self, "_onedal_estimator"):
            del self._onedal_estimator

        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _BaseRidge.fit,
            },
            X,
            Y,
            sample_weight,
            )
        return self

    fit.__doc__ = _sklearn_RidgeClassifier.fit.__doc__


# remove RegressorMixin from the mro
RidgeClassifier.__bases__ = tuple(cls for cls in RidgeClassifier.__bases__ if cls is not RegressorMixin)

# Allow for isinstance calls without inheritance changes using ABCMeta
_sklearn_RidgeClassifier.register(RidgeClassifier)
