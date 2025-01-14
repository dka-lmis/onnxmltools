# SPDX-License-Identifier: Apache-2.0

"""
Helper function for H2O Models and algorithms
"""
import unittest
import os
import h2o
from h2o.exceptions import H2OError

from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER

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


def _test_for_H2O_error(test: unittest.TestCase, model):
    folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
    if not os.path.exists(folder):
        os.makedirs(folder)
    mojo_path = model.download_mojo(path=folder)
    with test.assertRaises(H2OError) as err_h2o:
        _convert_mojo(mojo_path)
    test.assertRegex(err_h2o.exception.args[0], "Unable to print")


def _test_for_type_error(test: unittest.TestCase, model):
    folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
    if not os.path.exists(folder):
        os.makedirs(folder)
    mojo_path = model.download_mojo(path=folder)
    with test.assertRaises(ValueError) as err_type:
        _convert_mojo(mojo_path)
    test.assertRegex(err_type.exception.args[0], "not supported")


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
