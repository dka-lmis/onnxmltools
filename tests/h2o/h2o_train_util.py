# SPDX-License-Identifier: Apache-2.0

"""
Tests h2o's tree-based methods' converters.
"""
import unittest
import os
import h2o

from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER

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


def _train_and_get_model_path(model, x, y, train, valid):
    model = model.train(x=x, y=y, training_frame=train, validation_frame=valid)
    folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
    if not os.path.exists(folder):
        os.makedirs(folder)
    mojo_path = model.download_mojo(path=folder)
    return mojo_path


def _convert_mojo(mojo_path):
    f = open(mojo_path, "rb")
    mojo_content = f.read()
    f.close()
    return convert_h2o(mojo_content, target_opset=TARGET_OPSET)


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
        data_frame = h2o.H2OFrame(data, column_names=self._column_names)
        preds = self._mojo_model.predict(data_frame).as_data_frame(use_pandas=True)
        if len(preds.columns) == 1:
            return [preds.to_numpy()]
        else:
            return [
                preds.iloc[:, 0].to_numpy().astype(str),
                preds.iloc[:, 1:].to_numpy()
            ]


if __name__ == "__main__":
    # cl = TestH2OModels()
    # cl.setUpClass()
    # cl.test_h2o_classifier_multi_cat()
    unittest.main()
