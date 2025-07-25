# ==============================================================================
# Copyright 2014 Intel Corporation
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

# daal4py quantiles example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "quantiles.csv"

    # configure a quantiles object
    algo = d4p.quantiles()

    # let's provide a file directly, not a table/array
    result1 = algo.compute(str(infile))

    # We can also load the data ourselves and provide the numpy array
    data = readcsv(infile)
    result2 = algo.compute(data)

    # QUANTILES result objects provide quantiles
    assert np.allclose(result1.quantiles, result2.quantiles)

    return result1


if __name__ == "__main__":
    result = main()
    print("Quantiles:\n", result.quantiles)
    print("All looks good!")
