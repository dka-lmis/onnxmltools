# SPDX-License-Identifier: Apache-2.0

"""
Tests Naive Bayes Classifier (NaiveBayes) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/naive-bayes.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2ONaiveBayesEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model
from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_NaiveBayes_dataset():
    # Import the prostate dataset into H2O:
    prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate.csv")

    # Set predictors and response; set the response as a factor:
    prostate["CAPSULE"] = prostate["CAPSULE"].asfactor()
    predictors = ["ID", "AGE", "RACE", "DPROS", "DCAPS", "PSA", "VOL", "GLEASON"]
    response = "CAPSULE"

    return predictors, response, prostate


@unittest.skip(reason='Export to MOJO not supported')
class H2OTestConverterNaiveBayes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    @unittest.skip(reason='Export to MOJO not supported')
    def test_h2o_NaiveBayes_support(self):
        x, y, train = _get_NaiveBayes_dataset()
        model = H2ONaiveBayesEstimator(laplace=0,
                                       nfolds=5,
                                       seed=1234)
        model.train(x=x,
                    y=y,
                    training_frame=train)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_NaiveBayes_conversion(self):
        x, y, train = _get_NaiveBayes_dataset()
        model = H2ONaiveBayesEstimator(laplace=0,
                                       nfolds=5,
                                       seed=1234)
        model.train(x=x,
                    y=y,
                    training_frame=train)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_NaiveBayes_test_conversion")


if __name__ == "__main__":
    unittest.main()
