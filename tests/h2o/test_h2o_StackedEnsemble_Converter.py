# SPDX-License-Identifier: Apache-2.0

"""
Tests Principal Component Analysis (StackedEnsemble) converter
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
"""
import os
import unittest

from h2o import h2o
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator, H2OStackedEnsembleEstimator
from h2o.grid import H2OGridSearch
from onnx.defs import onnx_opset_version
from onnxconverter_common import DEFAULT_OPSET_NUMBER

from onnxmltools.utils import dump_data_and_model

from tests.h2o.h2o_train_util import _convert_mojo, H2OMojoWrapper, _test_for_H2O_error

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _get_StackedEnsemble_dataset():
    # Import a sample binary outcome train/test set into H2O
    train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
    test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

    # Identify predictors and response
    x = train.columns
    y = "response"
    x.remove(y)

    # For binary classification, response should be a factor
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    return x, y, train, test


class H2OTestConverterStackedEnsemble(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_StackedEnsemble_1_support(self):
        x, y, train, test = _get_StackedEnsemble_dataset()
        # Number of CV folds (to generate level-one data for stacking)
        nfolds = 5

        # There are a few ways to assemble a list of models to stack together:
        # 1. Train individual models and put them in a list
        # 2. Train a grid of models
        # 3. Train several grids of models
        # Note: All base models must have the same cross-validation folds and
        # the cross-validated predicted values must be kept.

        # 1. Generate a 2-model ensemble (GBM + RF)

        # Train and cross-validate a GBM
        my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",
                                              ntrees=10,
                                              max_depth=3,
                                              min_rows=2,
                                              learn_rate=0.2,
                                              nfolds=nfolds,
                                              fold_assignment="Modulo",
                                              keep_cross_validation_predictions=True,
                                              seed=1)
        my_gbm.train(x=x, y=y, training_frame=train)

        # Train and cross-validate a RF
        my_rf = H2ORandomForestEstimator(ntrees=50,
                                         nfolds=nfolds,
                                         fold_assignment="Modulo",
                                         keep_cross_validation_predictions=True,
                                         seed=1)
        my_rf.train(x=x, y=y, training_frame=train)

        # Train a stacked ensemble using the GBM and GLM above
        ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                               base_models=[my_gbm, my_rf])
        ensemble.train(x=x, y=y, training_frame=train)
        _test_for_H2O_error(self, ensemble)

    def test_h2o_StackedEnsemble_2_support(self):
        x, y, train, test = _get_StackedEnsemble_dataset()
        # Number of CV folds (to generate level-one data for stacking)
        nfolds = 5

        # There are a few ways to assemble a list of models to stack together:
        # 1. Train individual models and put them in a list
        # 2. Train a grid of models
        # 3. Train several grids of models
        # Note: All base models must have the same cross-validation folds and
        # the cross-validated predicted values must be kept.

        # 2. Generate a random grid of models and stack them together

        # Specify GBM hyperparameters for the grid
        hyper_params = {"learn_rate": [0.01, 0.03],
                        "max_depth": [3, 4, 5, 6, 9],
                        "sample_rate": [0.7, 0.8, 0.9, 1.0],
                        "col_sample_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
        search_criteria = {"strategy": "RandomDiscrete", "max_models": 3, "seed": 1}

        # Train the grid
        grid = H2OGridSearch(model=H2OGradientBoostingEstimator(ntrees=10,
                                                                seed=1,
                                                                nfolds=nfolds,
                                                                fold_assignment="Modulo",
                                                                keep_cross_validation_predictions=True),
                             hyper_params=hyper_params,
                             search_criteria=search_criteria,
                             grid_id="gbm_grid_binomial")
        grid.train(x=x, y=y, training_frame=train)

        # Train a stacked ensemble using the GBM grid
        ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_gbm_grid_binomial",
                                               base_models=grid.model_ids)
        ensemble.train(x=x, y=y, training_frame=train)
        _test_for_H2O_error(self, ensemble)

    @unittest.skip(reason='not yet implemented')
    def test_h2o_StackedEnsemble_conversion(self):
        x, y, train, test = _get_StackedEnsemble_dataset()
        # Number of CV folds (to generate level-one data for stacking)
        nfolds = 5

        # There are a few ways to assemble a list of models to stack together:
        # 1. Train individual models and put them in a list
        # 2. Train a grid of models
        # 3. Train several grids of models
        # Note: All base models must have the same cross-validation folds and
        # the cross-validated predicted values must be kept.

        # 1. Generate a 2-model ensemble (GBM + RF)

        # Train and cross-validate a GBM
        my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",
                                              ntrees=10,
                                              max_depth=3,
                                              min_rows=2,
                                              learn_rate=0.2,
                                              nfolds=nfolds,
                                              fold_assignment="Modulo",
                                              keep_cross_validation_predictions=True,
                                              seed=1)
        my_gbm.train(x=x, y=y, training_frame=train)

        # Train and cross-validate a RF
        my_rf = H2ORandomForestEstimator(ntrees=50,
                                         nfolds=nfolds,
                                         fold_assignment="Modulo",
                                         keep_cross_validation_predictions=True,
                                         seed=1)
        my_rf.train(x=x, y=y, training_frame=train)

        # Train a stacked ensemble using the GBM and GLM above
        ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                               base_models=[my_gbm, my_rf])
        ensemble.train(x=x, y=y, training_frame=train)
        _test_for_H2O_error(self, ensemble)
        folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
        if not os.path.exists(folder):
            os.makedirs(folder)
        mojo_path = ensemble.download_mojo(path=folder)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test, H2OMojoWrapper(mojo_path),
            onnx_model, basename="H2O_StackedEnsemble_test_conversion")


if __name__ == "__main__":
    unittest.main()
