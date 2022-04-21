# SPDX-License-Identifier: Apache-2.0

"""
Tests Principal Component Analysis (autoML) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/autoML.html
"""
import os
import unittest

from h2o import h2o
from h2o.automl import H2OAutoML
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_autoML_dataset():
    # Import a sample binary outcome train/test set into H2O
    train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
    test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

    # Identify predictors and response
    x = train.columns
    y = "response"
    x.remove(y)

    # For binary classification, response should be a factor
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    return x, y, train, test


class H2OTestConverterAutoML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_autoML_support(self):
        x, y, train, test = _get_autoML_dataset()
        model = H2OAutoML(max_models=2, seed=1)
        model.train(x=x, y=y, training_frame=train)
        # The leader model is stored here: aml.leader
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_autoML_conversion(self):
        x, y, train, test = _get_autoML_dataset()
        model = H2OAutoML(max_models=2, seed=1)
        model.train(x=x, y=y, training_frame=train)
        # The leader model is stored here: model.leader
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_autoML_test_conversion")


if __name__ == "__main__":
    unittest.main()
