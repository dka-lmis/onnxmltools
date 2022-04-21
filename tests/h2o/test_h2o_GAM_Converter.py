# SPDX-License-Identifier: Apache-2.0

"""
Tests H2O Generalized Additive Models (GAM) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gam.html
"""
import unittest

from h2o import h2o
from h2o.estimators import H2OGeneralizedAdditiveEstimator
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model
from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_GAM_dataset():
    # import the dataset
    h2o_data = h2o.import_file(
        "https://s3.amazonaws.com/h2o-public-test-data/smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv")

    # convert the C1, C2, and C11 columns to factors
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    h2o_data["C11"] = h2o_data["C11"].asfactor()

    # split into train and validation sets
    train, test = h2o_data.split_frame(ratios=[.8])

    # set the predictor and response columns
    y = "C11"
    x = ["C1", "C2"]
    return x, y, train, test


def _get_GAM_knots():
    # create frame knots
    knots1 = [-1.99905699, -0.98143075, 0.02599159, 1.00770987, 1.99942290]
    frameKnots1 = h2o.H2OFrame(python_obj=knots1)
    knots2 = [-1.999821861, -1.005257990, -0.006716042, 1.002197392, 1.999073589]
    frameKnots2 = h2o.H2OFrame(python_obj=knots2)
    knots3 = [-1.999675688, -0.979893796, 0.007573327, 1.011437347, 1.999611676]
    frameKnots3 = h2o.H2OFrame(python_obj=knots3)

    # specify the knots array
    num_knots = [5, 5, 5]
    return num_knots, frameKnots1, frameKnots2, frameKnots3


class H2OTestConverterGAM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_GAM_support(self):
        x, y, train, test = _get_GAM_dataset()
        num_knots, knots1, knots2, knots3 = _get_GAM_knots()
        model = H2OGeneralizedAdditiveEstimator(family='multinomial',
                                                gam_columns=["C6", "C7", "C8"],
                                                scale=[1, 1, 1],
                                                num_knots=num_knots,
                                                knot_ids=[knots1.key, knots2.key, knots3.key])
        model = model.train(x=x, y=y, training_frame=train, validation_frame=test)
        _test_for_H2O_error(self, model)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_GAM_conversion(self):
        x, y, train, test = _get_GAM_dataset()
        num_knots, knots1, knots2, knots3 = _get_GAM_knots()
        model = H2OGeneralizedAdditiveEstimator(family='multinomial',
                                                gam_columns=["C6", "C7", "C8"],
                                                scale=[1, 1, 1],
                                                num_knots=num_knots,
                                                knot_ids=[knots1.key, knots2.key, knots3.key])
        mojo_path = _train_and_get_model_path(model, x, y, train, test)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_GAM_test_conversion")


if __name__ == "__main__":
    unittest.main()
