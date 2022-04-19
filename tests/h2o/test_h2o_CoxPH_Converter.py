# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Distributed Random Forest (DRF) converter
"""
import os
import unittest

from h2o import h2o
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.convert.h2o import convert
from onnxmltools.utils import dump_data_and_model
from h2o.estimators.coxph import H2OCoxProportionalHazardsEstimator

from tests.h2o.h2o_train_util import _train_classifier, _convert_mojo

'''TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class H2OTestConverterCoxPH(unittest.TestCase):

    @staticmethod
    def setup():
        # Import the heart dataset into H2O:
        heart = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/coxph_test/heart.csv")

        # Split the dataset into a train and test set:
        train, _ = heart.split_frame(ratios=[.8], seed=1234)

        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        return train, folder

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_algo_support(self):
        train, folder = self.setup()
        model = H2OCoxProportionalHazardsEstimator(start_column="start",
                                                   stop_column="stop",
                                                   ties="breslow")
        model.train(x="age",
                    y="event",
                    training_frame=train)

        mojo_path = model.download_mojo(path=folder)
        print(mojo_path)
        with self.assertRaises(ValueError) as err:
            _convert_mojo(mojo_path)
        self.assertRegex(err.exception.args[0], "not supported")


if __name__ == "__main__":
    unittest.main()
'''