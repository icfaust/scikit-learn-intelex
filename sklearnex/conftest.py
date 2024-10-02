# ==============================================================================
# Copyright 2024 Intel Corporation
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

import io
import logging
import warnings

import pytest

from daal4py.sklearn._utils import sklearn_check_version
from sklearnex import config_context, patch_sklearn, unpatch_sklearn
from sklearnex.tests.utils import _IS_INTEL, _NONINTEL_DESELECTIONS


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_sklearn_fallback: mark test to not check for sklearnex usage"
    )
    config.addinivalue_line(
        "markers", "mpi: mark test to require MPI for distributed testing"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # setup logger to check for sklearn fallback
    if not item.get_closest_marker("allow_sklearn_fallback"):
        log_stream = io.StringIO()
        log_handler = logging.StreamHandler(log_stream)
        sklearnex_logger = logging.getLogger("sklearnex")
        level = sklearnex_logger.level
        sklearnex_stderr_handler = sklearnex_logger.handlers
        sklearnex_logger.handlers = []
        sklearnex_logger.addHandler(log_handler)
        sklearnex_logger.setLevel(logging.INFO)
        log_handler.setLevel(logging.INFO)

        yield

        sklearnex_logger.handlers = sklearnex_stderr_handler
        sklearnex_logger.setLevel(level)
        sklearnex_logger.removeHandler(log_handler)
        text = log_stream.getvalue()
        if "fallback to original Scikit-learn" in text:
            raise TypeError(
                f"test did not properly evaluate sklearnex functionality and fell back to sklearn:\n{text}"
            )
    else:
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config, items):
    """This deselects some tests which occur in GitHub Actions runners for 2024.2"""
    if not _IS_INTEL:
        print("\nNon-GenuineIntel hardware recognized, some tests deselected")
        nonintel_deselects = [i for i in items if i.name in _NONINTEL_DESELECTIONS]
        config.hook.pytest_deselected(items=nonintel_deselects)
    yield


@pytest.fixture
def with_sklearnex():
    patch_sklearn()
    yield
    unpatch_sklearn()


@pytest.fixture
def with_array_api():
    if sklearn_check_version("1.2"):
        with config_context(array_api_dispatch=True):
            yield
    else:
        yield
