# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Distributed Random Forest (DRF) converter
"""

import unittest

from h2o import h2o
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.convert.h2o import convert
from onnxmltools.utils import dump_data_and_model
from h2o.estimators.random_forest import H2ORandomForestEstimator

from tests.h2o.h2o_train_util import _train_classifier, _convert_mojo

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


'''class H2OTestConverterDRF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_unsupported_algo(self):
        drf = H2ORandomForestEstimator(ntrees=7, max_depth=5)
        mojo_path, test_data = _train_classifier(drf, 2, is_str=True)
        with self.assertRaises(ValueError) as err:
            _convert_mojo(mojo_path)
        self.assertRegex(err.exception.args[0], "not supported")


if __name__ == "__main__":
    unittest.main()
'''