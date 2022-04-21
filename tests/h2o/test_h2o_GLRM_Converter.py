# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Distributed Random Forest (GLRM) converter
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator, H2OGeneralizedLowRankEstimator
from h2o.exceptions import H2OError
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, _train_and_get_model_path, H2OMojoWrapper

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_GLRM_dataset():
    arrestsH2O = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/pca_test/USArrests.csv")

    # Split the dataset into a train and valid set:
    train, valid = arrestsH2O.split_frame(ratios=[.8], seed=1234)

    return train, valid


class H2OTestConverterGLRM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass
        # h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        pass
        # h2o.cluster().shutdown()

    def test_h2o_GLRM_support(self):
        train, test = _get_GLRM_dataset()
        model = H2OGeneralizedLowRankEstimator(k=4, loss="quadratic", gamma_x=0.5, gamma_y=0.5, max_iterations=700,
                                               recover_svd=True, init="SVD", transform="standardize")
        model.train(training_frame=train)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        with self.assertRaises(H2OError) as err_h2o:
            _convert_mojo(mojo_path)
        self.assertRegex(err_h2o.exception.args[0], "Unable to print")

    @unittest.skip(reason='not yet implemented')
    def test_h2o_GLRM_conversion(self):
        train, test = _get_GLRM_dataset()
        model = H2OGeneralizedLowRankEstimator(k=4, loss="quadratic", gamma_x=0.5, gamma_y=0.5, max_iterations=700,
                                               recover_svd=True, init="SVD", transform="standardize")
        model.train(training_frame=train)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_GLRM_test_conversion")


if __name__ == "__main__":
    unittest.main()
