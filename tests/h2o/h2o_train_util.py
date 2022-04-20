# SPDX-License-Identifier: Apache-2.0

"""
Tests h2o's tree-based methods' converters.
"""
import unittest
import os
import sys
import numpy as np
import pandas as pd
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import h2o
from h2o import H2OFrame
from h2o.exceptions import H2OConnectionError
from h2o.estimators.anovaglm import H2OANOVAGLMEstimator
from h2o.estimators.coxph import H2OCoxProportionalHazardsEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.extended_isolation_forest import H2OExtendedIsolationForestEstimator
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
from h2o.estimators.psvm import H2OSupportVectorMachineEstimator
from h2o.estimators.rulefit import H2ORuleFitEstimator
from h2o.estimators.svd import H2OSingularValueDecompositionEstimator
from h2o.estimators.targetencoder import H2OTargetEncoderEstimator
from onnxmltools.convert import convert_h2o
from onnxmltools.utils import dump_data_and_model

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _make_mojo(model, train, y=-1, force_y_numeric=False):
    if y < 0:
        y = train.ncol + y
    if force_y_numeric:
        train[y] = train[y].asnumeric()
    x = list(range(0, train.ncol))
    x.remove(y)
    model.train(x=x, y=y, training_frame=train)
    folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
    if not os.path.exists(folder):
        os.makedirs(folder)
    return model.download_mojo(path=folder)


def _convert_mojo(mojo_path):
    f = open(mojo_path, "rb")
    mojo_content = f.read()
    f.close()
    return convert_h2o(mojo_content, target_opset=TARGET_OPSET)


def _prepare_one_hot(file, y, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    frame = h2o.import_file(dir_path + "/" + file)
    train, test = frame.split_frame([0.95], seed=42)

    cols_to_encode = []
    other_cols = []
    for name, ctype in test.types.items():
        if name == y or name in exclude_cols:
            pass
        elif ctype == "enum":
            cols_to_encode.append(name)
        else:
            other_cols.append(name)
    train_frame = train.as_data_frame()
    train_encode = train_frame.loc[:, cols_to_encode]
    train_other = train_frame.loc[:, other_cols + [y]]
    enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
    enc.fit(train_encode)
    colnames = []
    for cidx in range(len(cols_to_encode)):
        for val in enc.categories_[cidx]:
            colnames.append(cols_to_encode[cidx] + "." + val)

    train_encoded = enc.transform(train_encode.values).toarray()
    train_encoded = pd.DataFrame(train_encoded)
    train_encoded.columns = colnames
    train = train_other.join(train_encoded)
    train = H2OFrame(train)

    test_frame = test.as_data_frame()
    test_encode = test_frame.loc[:, cols_to_encode]
    test_other = test_frame.loc[:, other_cols]

    test_encoded = enc.transform(test_encode.values).toarray()
    test_encoded = pd.DataFrame(test_encoded)
    test_encoded.columns = colnames
    test = test_other.join(test_encoded)

    return train, test


def _train_test_split_as_frames(x, y, is_str=False, is_classifier=False):
    y = y.astype(np.str) if is_str else y.astype(np.int64)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
    f_train_x = H2OFrame(x_train)
    f_train_y = H2OFrame(y_train)
    f_train = f_train_x.cbind(f_train_y)
    if is_classifier:
        f_train[f_train.ncol - 1] = f_train[f_train.ncol - 1].asfactor()
    return f_train, x_test.astype(np.float32)


def _train_classifier(model, n_classes, is_str=False, force_y_numeric=False):
    x, y = make_classification(
        n_classes=n_classes, n_features=100, n_samples=1000,
        random_state=42, n_informative=7
    )
    train, test = _train_test_split_as_frames(x, y, is_str, is_classifier=True)
    mojo_path = _make_mojo(model, train, force_y_numeric=force_y_numeric)
    return mojo_path, test


class H2OMojoWrapper:

    def __init__(self, mojo_path, column_names=None):
        self._mojo_path = mojo_path
        self._mojo_model = h2o.upload_mojo(mojo_path)
        self._column_names = column_names

    def __getstate__(self):
        return {
            "path": self._mojo_path,
            "colnames": self._column_names}

    def __setstate__(self, state):
        self._mojo_path = state.path
        self._mojo_model = h2o.upload_mojo(state.path)
        self._column_names = state.colnames

    def predict(self, arr):
        return self.predict_with_probabilities(arr)[0]

    def predict_with_probabilities(self, data):
        data_frame = H2OFrame(data, column_names=self._column_names)
        preds = self._mojo_model.predict(data_frame).as_data_frame(use_pandas=True)
        if len(preds.columns) == 1:
            return [preds.to_numpy()]
        else:
            return [
                preds.iloc[:, 0].to_numpy().astype(np.str),
                preds.iloc[:, 1:].to_numpy()
            ]


class TestH2OGBMModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()


if __name__ == "__main__":
    # cl = TestH2OModels()
    # cl.setUpClass()
    # cl.test_h2o_classifier_multi_cat()
    unittest.main()
