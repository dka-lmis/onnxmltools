# SPDX-License-Identifier: Apache-2.0

"""
Tests Principal Component Analysis (AGLM) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/AGLM.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OANOVAGLMEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_AGLM_dataset():
    # Import the prostate dataset
    train = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate_complete.csv.zip")

    # Set the predictors and response:
    x = ['AGE', 'VOL', 'DCAPS']
    y = 'CAPSULE'

    return x, y, train


@unittest.skip(reason='Export to MOJO not supported')
class H2OTestConverterAGLM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    @unittest.skip(reason='Export to MOJO not supported')
    def test_h2o_AGLM_support(self):
        x, y, train = _get_AGLM_dataset()
        model = H2OANOVAGLMEstimator(family='binomial',
                                     lambda_=0,
                                     missing_values_handling="skip")
        model.train(x=x, y=y, training_frame=train)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_AGLM_conversion(self):
        x, y, train = _get_AGLM_dataset()
        model = H2OANOVAGLMEstimator(family='binomial',
                                     lambda_=0,
                                     missing_values_handling="skip")
        model.train(x=x, y=y, training_frame=train)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_AGLM_test_conversion")


if __name__ == "__main__":
    unittest.main()
