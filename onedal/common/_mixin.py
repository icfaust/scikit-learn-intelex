# ==============================================================================
# Copyright 2021 Intel Corporation
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


class ClusterMixin:
    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None, queue=None, **kwargs):
        self.fit(X, queue=queue, **kwargs)
        return self.labels_

    def _more_tags(self):
        return {"preserves_dtype": []}


class ClassifierMixin:
    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None, queue=None):
        from sklearn.metrics import accuracy_score

        return accuracy_score(
            y, self.predict(X, queue=queue), sample_weight=sample_weight
        )

    def _more_tags(self):
        return {"requires_y": True}


class RegressorMixin:
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None, queue=None):
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X, queue=queue), sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_y": True}


class TransformerMixin:
    _estimator_type = "transformer"

    def fit_transform(self, X, y=None, queue=None, **fit_params):
        if y is None:
            return self.fit(X, queue=queue, **fit_params).transform(X, queue=queue)
        else:
            return self.fit(X, y, queue=queue, **fit_params).transform(X, queue=queue)
