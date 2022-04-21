# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Distributed Random Forest (IsoForest) converter
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OIsolationForestEstimator
from h2o.exceptions import H2OError
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_type_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_IsoForest_dataset():
    # Import the prostate dataset
    h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")

    # Split the data giving the training dataset 75% of the data
    train, test = h2o_df.split_frame(ratios=[0.75])

    return train, test


class H2OTestConverterIsoForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_IsoForest_support(self):
        train, test = _get_IsoForest_dataset()
        model = H2OIsolationForestEstimator(sample_rate=0.1,
                                            max_depth=20,
                                            ntrees=50)
        model.train(training_frame=train)
        _test_for_type_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_IsoForest_conversion(self):
        train, test = _get_IsoForest_dataset()
        model = H2OIsolationForestEstimator(sample_rate=0.1,
                                            max_depth=20,
                                            ntrees=50)
        model.train(training_frame=train)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_IsoForest_test_conversion")


if __name__ == "__main__":
    unittest.main()
