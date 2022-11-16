import pickle
from abc import ABC, abstractmethod
from typing import Callable

import mlflow
import numpy as np
import optuna


class _BaseTrainer(ABC):
    """Base class for all trainers."""

    def __int__(
        self, direction: str, saved_files_path: str, optimization_metric: str, kwargs
    ):
        """
        :param direction: Direction of optimization.
        :param saved_files_path: Directory to save logging files.
        :param optimization_metric: Metric to optimize.
        """

        self.direction = direction
        self.saved_files_path = saved_files_path
        self.optimization_metric = optimization_metric

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: list,
        n_trials: int,
        params_func: Callable[[optuna.trial.Trial], dict],
        sampler: optuna.samplers.BaseSampler,
    ):
        """Run training, hyperparameters search and logging.

        :param X: Training features.
        :param y: Target values.
        :param cv: Validation indexes. All but last element of list will be used for hyperparameters search, last
            element - test fold.

            Example:

            .. code-block:: python

                [[[0, 1, 2, 3], [4, 5]], # first validation fold
                 [[6, 7, 8, 9], [10, 11]], # second validation fold
                 [[12, 13, 14, 15], [16, 17]] # test fold
                 ]

            Observation with indexes [0, 1, 2, 3] will be used to train model, then this model will be tested on
            observations with indexes [4, 5]

            Observation with indexes [6, 7, 8, 9] will be used to train model, then this model will be tested on
            observations with indexes [10, 11]

            Observation with indexes [12, 13, 14, 15] will be used to train model, then this model will be tested on
            observations with indexes [16, 17]

            Metrics from 1st two folds will be used during hyperparameters optimization, metric from last fold will be just logged.

        :param n_trials: Number of iterations to search for hyperparamenets.
        :param params_func: Function which accepts optuna.trial.Trial and returns dict with hyperparameters.
            Read more about params_func in :ref:`the User Guide <ParamsFunc>`.

        :param sampler: Hyperparameters sampler from `optuna.samplers <https://optuna.readthedocs.io/en/stable/reference/samplers.html>`_
        """

        assert type(X) == np.ndarray, "X must be numpy.ndarray"
        assert type(y) == np.ndarray, "y must be numpy.ndarray"
        assert isinstance(
            sampler, optuna.samplers.BaseSampler
        ), "Sampler must be initiated sampler from optuna.samplers"
        for fold in cv:
            assert (
                len(fold) == 2
            ), "Each fold in cv must contain 2 sublists: one corresponding train indexes, and one corresponding test indexes"

        self.study = optuna.create_study(
            sampler=sampler, direction=self.direction, study_name="optuna_study"
        )

        self.study.optimize(
            lambda trial: self._objective(trial, X, y, cv, params_func),
            n_trials=n_trials,
            callbacks=[],
        )

        if self.logging_server == "neptune":
            self.run.stop()

        elif self.logging_server == "mlflow":
            mlflow.end_run()

    @abstractmethod
    def _run_iteration(
        self, X: np.ndarray, y: np.ndarray, cv: list, params: dict, trial_number: int
    ) -> dict:

        return

    def _objective(self, trial, X, y, cv, params_func):
        params = params_func(trial)

        results_dict = self._run_iteration(X, y, cv, params, trial.number)

        self._log_metrics(results_dict["metrics"], trial)
        self._log_params(results_dict["params"])
        if "file_paths" in results_dict.keys():
            self._log_files(results_dict["file_paths"])

        study_path = r"{}/saved_studies/optuna_study_{}.pickle".format(
            self.saved_files_path, trial.number
        )
        with open(study_path, "wb") as f:
            pickle.dump(self.study, f)

        self._log_files([study_path])

        return results_dict["metrics"][self.optimization_metric]
