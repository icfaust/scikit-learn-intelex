# ==============================================================================
# Copyright 2021 Intel Corporation
# Copyright 2024 Fujitsu Limited
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

import platform

from daal4py.sklearn._utils import daal_check_version

if "Windows" in platform.system():
    import os
    import site
    import sys

    arch_dir = platform.machine()
    plt_dict = {"x86_64": "intel64", "AMD64": "intel64", "aarch64": "arm"}
    arch_dir = plt_dict[arch_dir] if arch_dir in plt_dict else arch_dir
    path_to_env = site.getsitepackages()[0]
    path_to_libs = os.path.join(path_to_env, "Library", "bin")
    if sys.version_info.minor >= 8:
        if "DALROOT" in os.environ:
            dal_root_redist = os.path.join(os.environ["DALROOT"], "redist", arch_dir)
            if os.path.exists(dal_root_redist):
                os.add_dll_directory(dal_root_redist)
        try:
            os.add_dll_directory(path_to_libs)
        except FileNotFoundError:
            pass
    os.environ["PATH"] = path_to_libs + os.pathsep + os.environ["PATH"]

try:
    import onedal._onedal_py_dpc as _backend

    _is_dpc_backend = True
except ImportError:
    import onedal._onedal_py_host as _backend

    _is_dpc_backend = False

_is_spmd_backend = False

if _is_dpc_backend:
    try:
        import onedal._onedal_py_spmd_dpc as _spmd_backend

        _is_spmd_backend = True
    except ImportError:
        _is_spmd_backend = False


__all__ = ["covariance", "decomposition", "ensemble", "neighbors", "primitives", "svm"]

if _is_spmd_backend:
    __all__.append("spmd")

if daal_check_version((2023, "P", 100)):
    __all__ += ["basic_statistics", "linear_model"]

    if _is_spmd_backend:
        __all__ += [
            "spmd.basic_statistics",
            "spmd.decomposition",
            "spmd.linear_model",
            "spmd.neighbors",
        ]

if daal_check_version((2023, "P", 200)):
    __all__ += ["cluster"]

    if _is_spmd_backend:
        __all__ += ["spmd.cluster"]
