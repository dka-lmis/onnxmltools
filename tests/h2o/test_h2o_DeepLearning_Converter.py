# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Deep Learning (NN) converter
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2ODeepLearningEstimator
from h2o.exceptions import H2OError
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, _train_and_get_model_path, H2OMojoWrapper

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_NN_dataset():
    insurance = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/glm_test/insurance.csv")

    # Set the factors:
    insurance["offset"] = insurance["Holders"].log()
    insurance["Group"] = insurance["Group"].asfactor()
    insurance["Age"] = insurance["Age"].asfactor()
    insurance["District"] = insurance["District"].asfactor()

    return insurance


class H2OTestConverterNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_NN_support(self):
        dataset = _get_NN_dataset()
        model = H2ODeepLearningEstimator(distribution="tweedie", hidden=[1], epochs=1000, seed=23123, activation="Tanh",
                                         train_samples_per_iteration=-1, reproducible=True, single_node_mode=False,
                                         balance_classes=False, force_load_balance=False, tweedie_power=1.5,
                                         score_training_samples=0, score_validation_samples=0, stopping_rounds=0)
        model.train(x=list(range(3)), y="Claims", training_frame=dataset)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        with self.assertRaises(H2OError) as err_h2o:
            _convert_mojo(mojo_path)
        self.assertRegex(err_h2o.exception.args[0], "Unable to print")

    @unittest.skip(reason='not yet implemented')
    def test_h2o_NN_conversion(self):
        dataset = _get_NN_dataset()
        model = H2ODeepLearningEstimator(distribution="tweedie", hidden=[1], epochs=1000, seed=23123, activation="Tanh",
                                         train_samples_per_iteration=-1, reproducible=True, single_node_mode=False,
                                         balance_classes=False, force_load_balance=False, tweedie_power=1.5,
                                         score_training_samples=0, score_validation_samples=0, stopping_rounds=0)
        model.train(x=list(range(3)), y="Claims", training_frame=dataset)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        # TODO: custom MOJOWrapper for DL models
        dump_data_and_model(
            dataset, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_NN_test_conversion")


if __name__ == "__main__":
    unittest.main()
