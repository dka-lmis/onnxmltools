# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Generalized Linear Model (GLM) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html
"""
import unittest

from h2o import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model
from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_GLM_dataset():
    prostate = h2o.import_file("https://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
    prostate['CAPSULE'] = prostate['CAPSULE'].asfactor()
    prostate['RACE'] = prostate['RACE'].asfactor()
    prostate['DCAPS'] = prostate['DCAPS'].asfactor()
    prostate['DPROS'] = prostate['DPROS'].asfactor()

    predictors = ["AGE", "RACE", "VOL", "GLEASON"]
    response_col = "CAPSULE"

    return predictors, response_col, prostate


class H2OTestConverterGLM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_GLM_support(self):
        x, y, train = _get_GLM_dataset()
        model = H2OGeneralizedLinearEstimator(family="binomial", lambda_=0, compute_p_values=True)
        model.train(x=x, y=y, training_frame=train)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_GLM_conversion(self):
        x, y, train = _get_GLM_dataset()
        test = train
        model = H2OGeneralizedLinearEstimator(family="binomial", lambda_=0, compute_p_values=True)
        model.train(x=x, y=y, training_frame=train)
        mojo_path = _train_and_get_model_path(model, x, y, train, valid=test)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_GLM_test_conversion")


if __name__ == "__main__":
    unittest.main()
