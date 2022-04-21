# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O RuleFit (RuleFit) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/rulefit.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2ORuleFitEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model
from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_RuleFit_dataset():
    # Import the titanic dataset and set the column types:
    f = "https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv"
    df = h2o.import_file(path=f, col_types={'pclass': "enum", 'survived': "enum"})

    # Split the dataset into train and test
    train, test = df.split_frame(ratios=[0.8], seed=1)

    # Set the predictors and response:
    x = ["age", "sibsp", "parch", "fare", "sex", "pclass"]
    y = "survived"

    return x, y, train, test


class H2OTestConverterRuleFit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_RuleFit_support(self):
        x, y, train, test = _get_RuleFit_dataset()
        model = H2ORuleFitEstimator(max_rule_length=10,
                                   max_num_rules=100,
                                   seed=1)
        model.train(training_frame=train, x=x, y=y)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_RuleFit_conversion(self):
        x, y, train, test = _get_RuleFit_dataset()
        model = H2ORuleFitEstimator(max_rule_length=10,
                                   max_num_rules=100,
                                   seed=1)
        model.train(training_frame=train, x=x, y=y)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = model.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_RuleFit_test_conversion")


if __name__ == "__main__":
    unittest.main()
