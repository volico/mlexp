import os
import pickle
from typing import Callable, Iterable, Type

import mlflow
import numpy as np
import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from mlexp.trainers._base_logger import _BaseLogger
from mlexp.trainers._base_trainer import _BaseTrainer
from mlexp.trainers._utils import _save_models


class SklearnTrainer(_BaseTrainer, _BaseLogger):
    """Training, logging and hyperparameters search for scikit-learn models."""

    def __init__(
        self,
        sklearn_estimator: Type[sklearn.base.BaseEstimator],
        validation_metric: Callable[[np.ndarray, np.ndarray], float],
        direction: str,
        saved_files_path: str,
        optimization_metric: str = "metric_mean_cv",
    ):
        """
        :param sklearn_estimator: Scikit-learn estimator to be fitted.
        :param validation_metric: Score function or loss function with signature validation_metric(y_true, y_pred),
            must return float/integer value of metric.
        :param direction: Direction of optimization.
        :param saved_files_path: Directory to save logging files.
        :param optimization_metric: Metric to optimize.
        """

        assert isinstance(validation_metric, Callable)

        super().__init__(
            direction,
            saved_files_path,
            optimization_metric,
            validation_metric,
            model_type="sklearn",
        )

        self.sklearn_estimator = sklearn_estimator

    def _initiate_neptune_run(
        self, neptune_run_params: dict, upload_files: Iterable[str] = []
    ) -> None:
        """Initiation of neptune run.

        :param neptune_run_params: Neptune run parameters (will be passed to `neptune.init_run <https://docs.neptune.ai/api-reference/neptune#.init_run>`_).
        :param upload_files: List of paths to files which will be logged in neptune run.
        """

        file_name = r"{}/saved_utils/sklearn_estimator.pickle".format(
            self.saved_files_path
        )
        with open(file_name, "wb") as f:
            pickle.dump(self.sklearn_estimator, f)
        self.run[file_name.split(self.saved_files_path)[-1]].upload(
            file_name, wait=True
        )
        os.remove(file_name)

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
        :param upload_files: List of paths to files which will be logged in mlflow run.
        """

        file_name = r"{}/saved_utils/sklearn_estimator.pickle".format(
            self.saved_files_path
        )
        with open(file_name, "wb") as f:
            pickle.dump(self.sklearn_estimator, f)
        mlflow.log_artifact(
            file_name, file_name.split(self.saved_files_path)[-1].split("/")[1]
        )
        os.remove(file_name)

    def _run_iteration(self, X, y, cv, params, trial_number):
        """Train, evaluate scikit-learn model with defined parameters and log metrics.

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

        if self.direction == "maximize":
            is_higher_better = True
        else:
            is_higher_better = False

        sklearn_scoring = make_scorer(
            self.validation_metric, greater_is_better=is_higher_better
        )

        estimator = self.sklearn_estimator
        estimator_initialised = estimator(**params["model_params"])
        cv_results = cross_validate(
            estimator_initialised,
            X,
            y,
            cv=cv,
            scoring=sklearn_scoring,
            return_estimator=True,
            error_score="raise",
        )

        model_file_paths = _save_models(
            cv_results["estimator"], self.saved_files_path, trial_number
        )

        if self.direction == "maximize":
            coef = 1
        else:
            coef = -1

        return {
            "metrics": {
                "metric_mean_cv": np.mean(cv_results["test_score"][:-1] * coef),
                "metric_std_cv": np.std(cv_results["test_score"][:-1]),
                "metric_test": cv_results["test_score"][-1] * coef,
            },
            "file_paths": [*model_file_paths],
            "params": params,
        }
