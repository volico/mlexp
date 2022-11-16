import inspect
import os
import types
from types import ModuleType
from typing import Callable, Iterable

import mlflow
import numpy as np
import pytorch_lightning as pl

from mlexp.trainers._base_logger import _BaseLogger
from mlexp.trainers._base_trainer import _BaseTrainer
from mlexp.trainers._utils import _save_metric_curves


class TorchTrainer(_BaseTrainer, _BaseLogger):
    """Training, logging and hyperparameters search for pytorch-lightning neural network."""

    def __init__(
        self,
        nn_model_module: ModuleType,
        data_loaders_module: ModuleType,
        metrics_callback_module: ModuleType,
        validation_metric: Callable[[np.ndarray, np.ndarray], float],
        direction: str,
        saved_files_path: str,
        use_average_epochs_on_test_fold: bool = True,
        optimization_metric: str = "metric_mean_cv",
    ):
        """
        :param nn_model_module: Module with class nn_model, which inherits from `pytorch_lightning.LightningModule <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_.
        :param data_loaders_module: Module with function train_val_data_loaders, which has signature
            callable([numpy.ndarray, numpy.ndarray, list[list[int], list[int]]], [torch.utils.data.DataLoader, torch.utils.data.DataLoader])

        :param metrics_callback_module: Module with MetricsCallback class, which inherits from pytorch_lightning.Callback.

            Must have these 2 methods:

            - *get_metric* must return list with metric by epoches
            - *get_n_epochs* must return number of epoches as int


        :param validation_metric: Score function or loss function with signature validation_metric(y_true, y_pred),
            must return float/integer value of metric.
        :param direction: Direction of optimization.
        :param saved_files_path: Directory to save logging files.
        :param use_average_epochs_on_test_fold:  Whether to train model on test fold
            with mean number of epoches from validation folds or use number of epoches from params_func.
        :param optimization_metric: Metric to optimize.
        """

        assert (
            type(nn_model_module) == types.ModuleType
        ), "nn_model_module must be module"
        assert (
            type(data_loaders_module) == types.ModuleType
        ), "data_loaders_module must be module"
        assert (
            type(metrics_callback_module) == types.ModuleType
        ), "metrics_callback_module must be module"
        assert isinstance(validation_metric, Callable)
        assert (
            type(use_average_epochs_on_test_fold) == bool
        ), "log_metric_curves must be bool"

        super().__init__(
            direction,
            saved_files_path,
            optimization_metric,
            validation_metric,
            model_type="torch",
        )

        self.nn_model_module = nn_model_module
        self.data_loaders_module = data_loaders_module
        self.metrics_callback_module = metrics_callback_module
        self.use_average_epochs_on_test_fold = use_average_epochs_on_test_fold
        os.makedirs(r"{}/saved_metric_curves/".format(saved_files_path))

    def _initiate_neptune_run(
        self, neptune_run_params: dict, upload_files: Iterable[str] = []
    ) -> None:
        """Initiation of neptune run.

        :param neptune_run_params: Neptune run parameters (will be passed to `neptune.init_run <https://docs.neptune.ai/api-reference/neptune#.init_run>`_).
        :param upload_files: List of paths to files which will be logged in neptune run.
        """

        self.run[
            "use_average_epochs_on_test_fold"
        ] = self.use_average_epochs_on_test_fold

        nn_model_module_code_lines = inspect.getsource(self.nn_model_module)
        file_name = r"{}/saved_utils/nn_model_module.py".format(self.saved_files_path)
        with open(file_name, "w") as f:
            f.write(nn_model_module_code_lines)
        self.run[file_name.split(self.saved_files_path)[-1]].upload(
            file_name, wait=True
        )
        os.remove(file_name)

        data_loaders_module_code_lines = inspect.getsource(self.data_loaders_module)
        file_name = r"{}/saved_utils/data_loaders_module.py".format(
            self.saved_files_path
        )
        with open(file_name, "w") as f:
            f.write(data_loaders_module_code_lines)
        self.run[file_name.split(self.saved_files_path)[-1]].upload(
            file_name, wait=True
        )
        os.remove(file_name)

        metrics_callback_module_code_lines = inspect.getsource(
            self.metrics_callback_module
        )
        file_name = r"{}/saved_utils/metrics_callback_module.py".format(
            self.saved_files_path
        )
        with open(file_name, "w") as f:
            f.write(metrics_callback_module_code_lines)
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
        :param upload_files: List of paths to files which will be logged in neptune run.
        """

        mlflow.log_param(
            "use_average_epochs_on_test_fold", self.use_average_epochs_on_test_fold
        )

        nn_model_module_code_lines = inspect.getsource(self.nn_model_module)
        file_name = r"{}/saved_utils/nn_model_module.py".format(self.saved_files_path)
        with open(file_name, "w") as f:
            f.write(nn_model_module_code_lines)
        mlflow.log_artifact(
            file_name, file_name.split(self.saved_files_path)[-1].split("/")[1]
        )
        os.remove(file_name)

        data_loaders_module_code_lines = inspect.getsource(self.data_loaders_module)
        file_name = r"{}/saved_utils/data_loaders_module.py".format(
            self.saved_files_path
        )
        with open(file_name, "w") as f:
            f.write(data_loaders_module_code_lines)
        mlflow.log_artifact(
            file_name, file_name.split(self.saved_files_path)[-1].split("/")[1]
        )
        os.remove(file_name)

        metrics_callback_module_code_lines = inspect.getsource(
            self.metrics_callback_module
        )
        file_name = r"{}/saved_utils/metrics_callback_module.py".format(
            self.saved_files_path
        )
        with open(file_name, "w") as f:
            f.write(metrics_callback_module_code_lines)
        mlflow.log_artifact(
            file_name, file_name.split(self.saved_files_path)[-1].split("/")[1]
        )
        os.remove(file_name)

    def _train_model(self, X, y, fold, params):

        metrics_callback = self.metrics_callback_module.MetricsCallback()
        train_loader, val_loader = self.data_loaders_module.train_val_data_loaders(
            X, y, fold, **params["data_loaders_params"]
        )

        trainer = pl.Trainer(
            **params["trainer_params"],
            callbacks=[
                pl.callbacks.early_stopping.EarlyStopping(
                    **params["EarlyStopping_params"]
                ),
                metrics_callback,
            ]
        )

        my_model = self.nn_model_module.nn_model(
            **params["model_params"], validation_metric=self.validation_metric
        )

        trainer.fit(my_model, train_loader, val_loader)

        return trainer, metrics_callback

    def _run_iteration(self, X, y, cv, params, trial_number):
        """Train, evaluate pytorch-lightning neural network with defined parameters and log metrics.

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

        # Список метрик с кросс валидации
        metric_cv = []
        number_of_iterations = []
        metric_curves = {}
        model_file_paths = []

        initial_min_epoches = params["trainer_params"]["min_epochs"]
        initial_max_epoches = params["trainer_params"]["max_epochs"]

        for fold_num, fold in enumerate(cv):

            if fold_num == len(cv) - 1:
                fold_num = "test"
                if self.use_average_epochs_on_test_fold == True:
                    mean_number_of_iterations = int(
                        round(np.mean(number_of_iterations))
                    )
                    params["trainer_params"]["min_epochs"] = mean_number_of_iterations
                    params["trainer_params"]["max_epochs"] = mean_number_of_iterations
                    params["validation_mean_epochs"] = mean_number_of_iterations

            trainer, metrics_callback = self._train_model(X, y, fold, params)

            if fold_num == len(cv) - 1:
                params["trainer_params"]["min_epochs"] = initial_min_epoches
                params["trainer_params"]["max_epochs"] = initial_max_epoches

            model_file_path = r"{}/saved_models/model_trial_{}_fold_{}.ckpt".format(
                self.saved_files_path, trial_number, fold_num
            )
            trainer.save_checkpoint(model_file_path)

            model_file_paths.append(model_file_path)

            if fold_num != "test":
                metric_cv.append(float(metrics_callback.get_metric()[-1]))
                number_of_iterations.append(metrics_callback.get_n_epochs())

            else:
                metric_test = metrics_callback.get_metric()[-1]
            metric_curves["fold_{}_metric".format(str(fold_num))] = [
                str(x) for x in metrics_callback.get_metric()
            ]

        metric_curves_file = _save_metric_curves(
            metric_curves, self.saved_files_path, trial_number
        )

        return {
            "metrics": {
                "metric_mean_cv": np.mean(metric_cv),
                "metric_std_cv": np.std(metric_cv),
                "metric_test": metric_test,
            },
            "file_paths": [*model_file_paths, metric_curves_file],
            "params": params,
        }
