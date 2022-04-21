# SPDX-License-Identifier: Apache-2.0

"""
Tests Principal Component Analysis (SVM) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/SVM.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OSupportVectorMachineEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_SVM_dataset():
    # Import the splice dataset into H2O:
    splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")
    return splice


@unittest.skip(reason='Export to MOJO not supported')
class H2OTestConverterSVM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    @unittest.skip(reason='h2o.exceptions.H2OValueError: Export to MOJO not supported')
    def test_h2o_SVM_support(self):
        train = _get_SVM_dataset()
        model = H2OSupportVectorMachineEstimator(gamma=0.01,
                                                 rank_ratio=0.1,
                                                 disable_training_metrics=False)
        model.train(y="C1", training_frame=train)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_SVM_conversion(self):
        train = _get_SVM_dataset()
        model = H2OSupportVectorMachineEstimator(gamma=0.01,
                                                 rank_ratio=0.1,
                                                 disable_training_metrics=False)
        model.train(y="C1", training_frame=train)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        # dump_data_and_model(
        #     test, H2OMojoWrapper(mojo_path),
        #     onnx_model, basename="H2O_SVM_test_conversion")


if __name__ == "__main__":
    unittest.main()
