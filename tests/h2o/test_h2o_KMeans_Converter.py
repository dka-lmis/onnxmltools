# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O K-Means Clustering (KMeans) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/k-means.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OKMeansEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model
from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_KMeans_dataset():
    # Import the iris dataset into H2O:
    iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")

    # Set the predictors:
    predictors = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

    # Split the dataset into a train and valid set:
    train, valid = iris.split_frame(ratios=[.8], seed=1234)

    return predictors, train, valid


class H2OTestConverterKMeans(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_KMeans_support(self):
        x, train, test = _get_KMeans_dataset()
        model = H2OKMeansEstimator(k=10,
                                   estimate_k=True,
                                   standardize=False,
                                   seed=1234)
        model.train(x=x, training_frame=train, validation_frame=test)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_KMeans_conversion(self):
        x, train, test = _get_KMeans_dataset()
        model = H2OKMeansEstimator(k=10,
                                   estimate_k=True,
                                   standardize=False,
                                   seed=1234)
        model.train(x=x, training_frame=train, validation_frame=test)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_KMeans_test_conversion")


if __name__ == "__main__":
    unittest.main()
