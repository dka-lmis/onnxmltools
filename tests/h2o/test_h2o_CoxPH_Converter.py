# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Distributed Random Forest (DRF) converter
"""
import unittest

from h2o.exceptions import H2OError
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER
from onnxmltools.utils import dump_data_and_model

from h2o import h2o
from h2o.estimators.coxph import H2OCoxProportionalHazardsEstimator

from tests.h2o.h2o_train_util import _convert_mojo, _train_and_get_model_path, H2OMojoWrapper
TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_CoxPH_dataset():
    # Import the heart dataset into H2O:
    heart = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/coxph_test/heart.csv")

    # Split the dataset into a train and test set:
    train, valid = heart.split_frame(ratios=[.8], seed=1234)
    x = 'age'
    y = 'event'

    return x, y, train, valid


class H2OTestConverterCoxPH(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    @unittest.skip(reason='''H2O does not generates an empty file when downloading a CoxPH Model. The model generates 
                          a correct zip-file with the data of the model though.''')
    def test_h2o_CoxPH_algo_support(self):
        x, y, train, test = _get_CoxPH_dataset()
        model = H2OCoxProportionalHazardsEstimator(start_column="start",
                                                   stop_column="stop",
                                                   ties="breslow")
        mojo_path = _train_and_get_model_path(model, x, y, train, test)
        with self.assertRaises(H2OError) as err:
            _convert_mojo(mojo_path)
        self.assertRegex(err.exception.args[0], "Unable to print")

    @unittest.skip(reason='not yet implemented')
    def test_h2o_CoxPH_conversion(self):
        x, y, train, test = _get_CoxPH_dataset()
        model = H2OCoxProportionalHazardsEstimator(start_column="start",
                                                   stop_column="stop",
                                                   ties="breslow")
        mojo_path = _train_and_get_model_path(model, x, y, train, test)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2OReg-Dec4")


if __name__ == "__main__":
    unittest.main()
