# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Distributed Random Forest (ExtendedIsoForest) converter
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_type_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_ExtendedIsoForest_dataset():
    # Import the prostate dataset
    h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")

    # Set the predictors
    predictors = ["AGE", "RACE", "DPROS", "DCAPS", "PSA", "VOL", "GLEASON"]

    return predictors, h2o_df


@unittest.skip(reason='Export to MOJO not supported')
class H2OTestConverterExtendedIsoForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    @unittest.skip(reason='Export to MOJO not supported')
    def test_h2o_ExtendedIsoForest_support(self):
        x, train = _get_ExtendedIsoForest_dataset()
        model = H2OExtendedIsolationForestEstimator(model_id="eif.hex",
                                                    ntrees=100,
                                                    sample_size=256,
                                                    extension_level=len(x) - 1)
        model.train(x=x, training_frame=train)
        _test_for_type_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_ExtendedIsoForest_conversion(self):
        x, train = _get_ExtendedIsoForest_dataset()
        model = H2OExtendedIsolationForestEstimator(model_id="eif.hex",
                                                    ntrees=100,
                                                    sample_size=256,
                                                    extension_level=len(x) - 1)
        model.train(x=x, training_frame=train)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_ExtendedIsoForest_test_conversion")


if __name__ == "__main__":
    unittest.main()
