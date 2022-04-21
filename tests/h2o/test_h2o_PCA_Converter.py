# SPDX-License-Identifier: Apache-2.0

"""
Tests Principal Component Analysis (PCA) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/pca.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OPrincipalComponentAnalysisEstimator
from h2o.exceptions import H2OError
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_PCA_dataset():
    # Import the birds dataset into H2O:
    birds = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/pca_test/birds.csv")

    # Split the dataset into a train and valid set:
    train, valid = birds.split_frame(ratios=[.8], seed=1234)

    return train, valid


class H2OTestConverterPCA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_PCA_support(self):
        train, test = _get_PCA_dataset()
        model = H2OPrincipalComponentAnalysisEstimator(k=5,
                                                       use_all_factor_levels=True,
                                                       pca_method="glrm",
                                                       transform="standardize",
                                                       impute_missing=True)
        model.train(training_frame=train)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_PCA_conversion(self):
        train, test = _get_PCA_dataset()
        model = H2OPrincipalComponentAnalysisEstimator(k=5,
                                                       use_all_factor_levels=True,
                                                       pca_method="glrm",
                                                       transform="standardize",
                                                       impute_missing=True)
        model.train(training_frame=train)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_PCA_test_conversion")


if __name__ == "__main__":
    unittest.main()
