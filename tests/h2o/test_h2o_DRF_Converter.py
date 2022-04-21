# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Distributed Random Forest (DRF) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html
"""
import unittest

from h2o import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model
from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_type_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_DRF_dataset():
    cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")
    cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()
    x = ["displacement", "power", "weight", "acceleration", "year"]
    y = "economy_20mpg"

    # Split the dataset into a train and valid set:
    train, valid = cars.split_frame(ratios=[.8], seed=1234)

    return x, y, train, valid


class H2OTestConverterDRF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_DRF_support(self):
        x, y, train, test = _get_DRF_dataset()
        model = H2ORandomForestEstimator(ntrees=10, max_depth=5, min_rows=10, binomial_double_trees=True)
        model = model.train(x=x, y=y, training_frame=train, validation_frame=test)
        _test_for_type_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_DRF_conversion(self):
        x, y, train, test = _get_DRF_dataset()
        model = H2ORandomForestEstimator(ntrees=10, max_depth=5, min_rows=10, binomial_double_trees=True)
        mojo_path = _train_and_get_model_path(model, x, y, train, test)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_DRF_test_conversion")


if __name__ == "__main__":
    unittest.main()
