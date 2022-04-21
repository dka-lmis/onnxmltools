# SPDX-License-Identifier: Apache-2.0

"""
Tests XGBoost (XGBoost) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OXGBoostEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model
from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_XGBoost_dataset():
    titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")

    # Set the predictors and response; set the response as a factor:
    titanic["survived"] = titanic["survived"].asfactor()
    predictors = titanic.columns
    response = "survived"

    # Split the dataset into a train and valid set:
    train, valid = titanic.split_frame(ratios=[.8], seed=1234)

    return predictors, response, train, valid


@unittest.skip(reason='Server Side Errors')
class H2OTestConverterXGBoost(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    @unittest.skip(reason='''h2o.exceptions.H2OResponseError: Server error water.exceptions.H2ONotFoundArgumentException:
    Error: POST /3/ModelBuilders/xgboost not found
    Request: POST /3/ModelBuilders/xgboost''')
    def test_h2o_XGBoost_support(self):
        x, y, train, test = _get_XGBoost_dataset()
        model = H2OXGBoostEstimator(booster='dart',
                                    normalize_type="tree",
                                    seed=1234)
        model.train(x=x,
                    y=y,
                    training_frame=train,
                    validation_frame=test)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_XGBoost_conversion(self):
        x, y, train, test = _get_XGBoost_dataset()
        model = H2OXGBoostEstimator(booster='dart',
                                    normalize_type="tree",
                                    seed=1234)
        model.train(x=x,
                    y=y,
                    training_frame=train,
                    validation_frame=test)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_XGBoost_test_conversion")


if __name__ == "__main__":
    unittest.main()
