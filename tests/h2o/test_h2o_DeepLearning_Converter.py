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
