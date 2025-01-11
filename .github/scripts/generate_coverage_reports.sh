#===============================================================================
# Copyright Contributors to the oneDAL project
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
#===============================================================================

# create coverage.py report
coverage combine .coverage.sklearnex .coverage.sklearn
coverage json -o coverage."${1}".json

# create gcov report (lcov format)
if [[ $OSTYPE == *"linux"* ]]; then
    # extract llvm tool for gcov processing
    if [[ -z "$2" ]]; then
        GCOV_EXE="$(dirname $(type -P -a icpx))/compiler/llvm-cov gcov"
    else
        GCOV_EXE="gcov-4"
        g++ --version
        gcov --version
    fi
    echo $GCOV_EXE
    FILTER=$(realpath ./onedal).*
    echo $FILTER
    cd build
    gcovr --gcov-executable "${GCOV_EXE}" -r ../ . --lcov -v --filter "${FITLER}" -o ../coverage"${1}".info
    sed -i "s|${PWD}/||g" ../coverage"${1}".info
    # remove absolute filepath to match coverage.py file
fi