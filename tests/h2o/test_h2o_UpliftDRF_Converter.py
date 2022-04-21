# SPDX-License-Identifier: Apache-2.0

"""
Tests Principal Component Analysis (UDRF) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/UDRF.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OPrincipalComponentAnalysisEstimator, H2OUpliftRandomForestEstimator
from h2o.exceptions import H2OError
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_UDRF_dataset():
    # Import the cars dataset into H2O:
    data = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/uplift/criteo_uplift_13k.csv")

    # Set the predictors, response, and treatment column:
    predictors = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
    # set the response as a factor
    response = "conversion"
    data[response] = data[response].asfactor()
    # set the treatment as a factor
    treatment_column = "treatment"
    data[treatment_column] = data[treatment_column].asfactor()

    # Split the dataset into a train and valid set:
    train, valid = data.split_frame(ratios=[.8], seed=1234)

    return predictors, response, treatment_column, train, valid


@unittest.skip(reason='Export to MOJO not supported')
class H2OTestConverterUDRF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    @unittest.skip(reason='Export to MOJO not supported')
    def test_h2o_UDRF_support(self):
        x, y, treatment, train, test = _get_UDRF_dataset()
        model = H2OUpliftRandomForestEstimator(ntrees=10, max_depth=5, treatment_column=treatment,
                                               uplift_metric="KL", min_rows=10, seed=1234, auuc_type="qini")
        model.train(x=x, y=y, training_frame=train, validation_frame=test)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_UDRF_conversion(self):
        x, y, treatment, train, test = _get_UDRF_dataset()
        model = H2OUpliftRandomForestEstimator(ntrees=10, max_depth=5, treatment_column=treatment,
                                               uplift_metric="KL", min_rows=10, seed=1234, auuc_type="qini")
        model.train(x=x, y=y, training_frame=train, validation_frame=test)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_UDRF_test_conversion")


if __name__ == "__main__":
    unittest.main()
