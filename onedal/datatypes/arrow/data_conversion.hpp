/*******************************************************************************
* Copyright Contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <array>
#include <cstdint>

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif // ONEDAL_DATA_PARALLEL

#include <pybind11/pybind11.h>

#include "onedal/datatypes/dlpack/arrow_utils.hpp"

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::python::arrow {

namespace py = pybind11;

dal::table convert_to_table(py::object obj, py::object q_obj = py::none(), bool recursed = false);

py::tuple get_arrow_device(const dal::table& input);
// necessary for extracting the device not available as a python attribute
py::tuple get_arrow_device(const py::object obj);

py::capsule construct_arrow(const dal::table& input);

} // namespace oneapi::dal::python::arrow
