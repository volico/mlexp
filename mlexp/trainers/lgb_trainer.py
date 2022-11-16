import os
from typing import Callable, Iterable

import lightgbm as lgb
import mlflow
import numpy as np

from mlexp.trainers._base_logger import _BaseLogger
from mlexp.trainers._base_trainer import _BaseTrainer
from mlexp.trainers._utils import _save_metric_curves, _save_models


class LgbTrainer(_BaseTrainer, _BaseLogger):
    """Training, logging and hyperparameters search for lightgbm."""

    def __init__(
        self,
        validation_metric: Callable[[np.ndarray, np.ndarray], float],
        direction: str,
        saved_files_path: str,
        use_average_n_estimators_on_test_fold: bool = True,
        optimization_metric: str = "metric_mean_cv",
    ):
        """
        :param validation_metric: Score function or loss function with signature validation_metric(y_true, y_pred),
            must return float/integer value of metric.
        :param direction: Direction of optimization.
        :param saved_files_path: Directory to save logging files.
        :param use_average_n_estimators_on_test_fold:  Whether to train model on test fold
            with mean number of n_estimators from validation folds or use n_estimators from params_func.
        :param optimization_metric: Metric to optimize.
        """

        assert isinstance(validation_metric, Callable)
        assert (
            type(use_average_n_estimators_on_test_fold) == bool
        ), "log_metric_curves must be bool"

        super().__init__(
            direction,
            saved_files_path,
            optimization_metric,
            validation_metric,
            model_type="lgb",
        )

        self.use_average_n_estimators_on_test_fold = (
            use_average_n_estimators_on_test_fold
        )
        os.makedirs(r"{}/saved_metric_curves/".format(saved_files_path))

    def _initiate_neptune_run(
        self, neptune_run_params: dict, upload_files: Iterable[str] = []
    ) -> None:
        """Initiation of neptune run.

        :param neptune_run_params: Neptune run parameters (will be passed to `neptune.init_run <https://docs.neptune.ai/api-reference/neptune#.init_run>`_).
        :param upload_files: List of paths to files which will be logged in neptune run.
        """

        self.run[
            "use_average_n_estimators_on_test_fold"
        ] = self.use_average_n_estimators_on_test_fold

    def _initiate_mlflow_run(
        self,
        tracking_uri: str,
        experiment_name: str,
        mlflow_run_params: dict,
        upload_files: Iterable[str] = [],
    ) -> None:
        """Initiation of mlflow run.

        :param tracking_uri: URI of mlflow server (will be passed to `mlflow.set_tracking_uri <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri>`_).
        :param experiment_name: Name of mlflow experiment for logging (will be passed to `mlflow.set_experiment <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment>`_)
        :param mlflow_run_params: Mlflow run parameters (will be passed to `mlflow.start_run <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run>`_).
        :param upload_files: List of paths to files which will be logged in neptune run.
        """

        mlflow.log_param(
            "use_average_n_estimators_on_test_fold",
            self.use_average_n_estimators_on_test_fold,
        )

    def _scoring_wrapper(self, validation_metric, direction):
        def lgb_scoring(
            y_hat, data, validation_metric=validation_metric, direction=direction
        ):

            y_true = data.get_label()

            if direction == "maximize":
                is_higher_better = True
            else:
                is_higher_better = False
            return "val_score", validation_metric(y_true, y_hat), is_higher_better

        return lgb_scoring

    def _train_validation(self, X, y, params, cv, lgb_scoring):

        train_data = lgb.Dataset(X, y, **params["lgb_data_set_params"])
        validation_results = lgb.cv(
            params=params["model_params"],
            train_set=train_data,
            folds=cv[:-1],
            feval=lgb_scoring,
            return_cvbooster=True,
        )

        num_iters = []
        for booster in validation_results["cvbooster"].boosters:
            num_iters.append(booster.current_iteration())
        mean_num_iters = int(round(np.mean(num_iters)))

        return (
            validation_results["val_score-mean"],
            validation_results["val_score-stdv"],
            mean_num_iters,
            validation_results["cvbooster"].boosters,
        )

    def _train_test(self, X, y, params, cv, lgb_scoring):

        X_train = X[cv[-1][0], :]
        y_train = y[cv[-1][0]]
        X_test = X[cv[-1][1], :]
        y_test = y[cv[-1][1]]
        train_data = lgb.Dataset(X_train, y_train, **params["lgb_data_set_params"])
        test_data = lgb.Dataset(X_test, y_test, **params["lgb_data_set_params"])
        test_results = {}
        trained_test_model = lgb.train(
            params=params["model_params"],
            train_set=train_data,
            valid_sets=[test_data],
            valid_names=["test_data"],
            feval=lgb_scoring,
            callbacks=[lgb.record_evaluation(test_results)],
        )

        return test_results["test_data"]["val_score"], trained_test_model

    def _run_iteration(
        self, X: np.ndarray, y: np.ndarray, cv: list, params: dict, trial_number: str
    ) -> dict:
        """Train, evaluate lightgbm model with defined parameters and log metrics.
        :param X: training data
        :type X: ndarray
        :param y: target values
        :type y: ndarray
        :param cv: indexes of cross validation
        :type cv: iterable of (train_inex, test_index)
        :param params: dictionary of parameters
        :type params: dict
        :param trial_number: number of current trial
        :type trial_number: int
        :return: metrics of current iteration
        :rtype: dictionary with keys ('metric_mean_cv' - mean metric on cross validation,
                                      'metric_std_cv' - standard deviation of metric on cross validation
                                      'metric_test' - metric on test data)
        """

        initial_n_estimators = params["model_params"]["n_estimators"]
        lgb_scoring = self._scoring_wrapper(self.validation_metric, self.direction)

        (
            metric_mean_validation_curve,
            metric_std_validation_curve,
            n_estimators,
            trained_validation_models,
        ) = self._train_validation(X, y, params, cv, lgb_scoring)

        if self.use_average_n_estimators_on_test_fold == True:
            params["model_params"]["n_estimators"] = n_estimators
            params["validation_mean_estimators"] = n_estimators

        metric_test_curve, trained_test_model = self._train_test(
            X, y, params, cv, lgb_scoring
        )
        params["model_params"]["n_estimators"] = initial_n_estimators

        model_file_paths = _save_models(
            [*trained_validation_models, trained_test_model],
            self.saved_files_path,
            trial_number,
        )

        metric_curves = {
            "mean_validation_metric": metric_mean_validation_curve,
            "fold_test_metric": metric_test_curve,
        }

        metric_curves_file = _save_metric_curves(
            metric_curves, self.saved_files_path, trial_number
        )

        return {
            "metrics": {
                "metric_mean_cv": metric_mean_validation_curve[-1],
                "metric_std_cv": metric_std_validation_curve[-1],
                "metric_test": metric_test_curve[-1],
            },
            "file_paths": [*model_file_paths, metric_curves_file],
            "params": params,
        }
