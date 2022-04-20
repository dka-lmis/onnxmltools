# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Distributed Random Forest (DRF) converter
"""
import os
import unittest

from h2o import h2o
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.convert.h2o import convert
from onnxmltools.utils import dump_data_and_model
from h2o.estimators.random_forest import H2ORandomForestEstimator

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_DRF_dataset():
    cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")
    cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()
    x = ["displacement", "power", "weight", "acceleration", "year"]
    y = "economy_20mpg"

    # Split the dataset into a train and valid set:
    train, valid = cars.split_frame(ratios=[.8], seed=1234)

    return x, y, train, valid


def _get_DRF_model(model, x, y, train, valid):
    model = model.train(x=x, y=y, training_frame=train, validation_frame=valid)
    folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
    if not os.path.exists(folder):
        os.makedirs(folder)
    mojo_path = model.download_mojo(path=folder)
    return mojo_path


def convert_mojo(mojo_path):
    f = open(mojo_path, "rb")
    mojo_content = f.read()
    f.close()
    return convert(mojo_content, target_opset=TARGET_OPSET)


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
        mojo_path = _get_DRF_model(model, x, y, train, test)
        with self.assertRaises(ValueError) as err:
            convert_mojo(mojo_path)
        self.assertRegex(err.exception.args[0], "not supported")

    @unittest.skip(reason='not yet implemented')
    def test_h2o_DRF_conversion(self):
        x, y, train, test = _get_DRF_dataset()
        model = H2ORandomForestEstimator(ntrees=10, max_depth=5, min_rows=10, binomial_double_trees=True)
        mojo_path = _get_DRF_model(model, x, y, train, test)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2OReg-Dec4")


if __name__ == "__main__":
    unittest.main()
