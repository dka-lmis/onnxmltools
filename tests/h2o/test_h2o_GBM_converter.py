# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Gradient Boosting Machine (GBM) converter
"""
import os
import unittest

import numpy as np
from h2o import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER
from sklearn.datasets import load_diabetes, load_iris

from onnxmltools.utils import dump_data_and_model
from h2o.estimators.random_forest import H2ORandomForestEstimator
from tests.h2o.h2o_train_util import _train_classifier, _convert_mojo, _train_test_split_as_frames, _make_mojo, \
    H2OMojoWrapper, _prepare_one_hot

from onnxmltools.convert import convert_h2o

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class H2OTestConverterGBM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    @staticmethod
    def _get_dataset(force_y_numeric=False):
        # Import the prostate dataset into H2O:
        prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate.csv")

        # Set the predictors and response; set the factors:
        predictors = ["ID", "AGE", "RACE", "DPROS", "DCAPS", "PSA", "VOL", "GLEASON"]
        response = ["CAPSULE"]
        if force_y_numeric:
            prostate[response] = prostate[response].asnumeric()
        else:
            prostate[response] = prostate[response].asfactor()

        return predictors, response, prostate

    @staticmethod
    def _convert_mojo(mojo_path):
        f = open(mojo_path, "rb")
        mojo_content = f.read()
        f.close()
        return convert_h2o(mojo_content, target_opset=TARGET_OPSET)

    def test_h2o_unsupported_algo(self):
        gbm = H2ORandomForestEstimator(ntrees=7, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 2, is_str=True)
        with self.assertRaises(ValueError) as err:
            _convert_mojo(mojo_path)
        self.assertRegex(err.exception.args[0], "not supported")

    def test_h2o_regressor(self):
        diabetes = load_diabetes()
        train, test = _train_test_split_as_frames(diabetes.data, diabetes.target)
        dists = ["auto", "gaussian", "huber", "laplace", "quantile"]
        for d in dists:
            gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5, distribution=d)
            mojo_path = _make_mojo(gbm, train)
            onnx_model = _convert_mojo(mojo_path)
            self.assertIsNot(onnx_model, None)
            dump_data_and_model(
                test, H2OMojoWrapper(mojo_path),
                onnx_model, basename="H2OReg-Dec4")

    @unittest.skipIf(True, reason="Failure with latest version of h2o")
    def test_h2o_regressor_cat(self):
        y = "IsDepDelayed"
        train, test = _prepare_one_hot("airlines.csv", y, exclude_cols=["IsDepDelayed_REC"])
        gbm = H2OGradientBoostingEstimator(ntrees=8, max_depth=5)
        mojo_path = _make_mojo(gbm, train, y=train.columns.index(y))
        onnx_model = self._convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test.values.astype(np.float32),
            H2OMojoWrapper(mojo_path, list(test.columns)),
            onnx_model, basename="H2ORegCat-Dec4")

    def test_h2o_classifier_multi_2class(self):
        gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5, distribution="multinomial")
        mojo_path, test_data = _train_classifier(gbm, 2, is_str=True)
        with self.assertRaises(ValueError) as err:
            self._convert_mojo(mojo_path)
        self.assertRegex(err.exception.args[0], "not supported")

    def test_h2o_classifier_bin_cat(self):
        y = "IsDepDelayed_REC"
        train, test = _prepare_one_hot("airlines.csv", y, exclude_cols=["IsDepDelayed"])
        gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5)
        mojo_path = _make_mojo(gbm, train, y=train.columns.index(y))
        onnx_model = self._convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test.values.astype(np.float32),
            H2OMojoWrapper(mojo_path, list(test.columns)),
            onnx_model, basename="H2OClassBinCat")

    def test_h2o_classifier_multi_cat(self):
        y = "fYear"
        train, test = _prepare_one_hot("airlines.csv", y)
        gbm = H2OGradientBoostingEstimator(ntrees=8, max_depth=5)
        mojo_path = _make_mojo(gbm, train, y=train.columns.index(y))
        print("****", mojo_path)
        onnx_model = self._convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test.values.astype(np.float32),
            H2OMojoWrapper(mojo_path, list(test.columns)),
            onnx_model, basename="H2OClassMultiCat")

    @unittest.skipIf(True, reason="Failure with latest version of h2o")
    def test_h2o_classifier_bin_str(self):
        gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 2, is_str=True)
        onnx_model = self._convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test_data, H2OMojoWrapper(mojo_path), onnx_model,
            basename="H2OClassBinStr")

    def test_h2o_classifier_bin_int(self):
        gbm = H2OGradientBoostingEstimator(ntrees=8, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 2, is_str=False, force_y_numeric=True)
        onnx_model = self._convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test_data, H2OMojoWrapper(mojo_path), onnx_model,
            basename="H2OClassBinInt")

    def test_h2o_classifier_multi_str(self):
        gbm = H2OGradientBoostingEstimator(ntrees=10, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 11, is_str=True)
        onnx_model = self._convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test_data, H2OMojoWrapper(mojo_path), onnx_model,
            basename="H2OClassMultiStr")

    def test_h2o_classifier_multi_int(self):
        gbm = H2OGradientBoostingEstimator(ntrees=9, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 9, is_str=False)
        onnx_model = self._convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test_data, H2OMojoWrapper(mojo_path), onnx_model,
            basename="H2OClassMultiBin")

    def test_h2o_classifier_multi_discrete_int_labels(self):
        iris = load_iris()
        x = iris.data[:, :2]
        y = iris.target
        y[y == 0] = 10
        y[y == 1] = 20
        y[y == 2] = -30
        train, test = _train_test_split_as_frames(x, y, is_str=False, is_classifier=True)
        gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5)
        mojo_path = _make_mojo(gbm, train)
        onnx_model = self._convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path), onnx_model,
            basename="H2OClassMultiDiscInt")


if __name__ == "__main__":
    unittest.main()
