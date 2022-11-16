import json
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Literal

import mlflow
import neptune.new as neptune
import numpy as np


class _BaseLogger(ABC):
    """Class for experiment logging."""

    def __init__(
        self,
        direction: str,
        saved_files_path: str,
        optimization_metric: str,
        validation_metric: Callable[[np.ndarray, np.ndarray], float],
        model_type: str,
    ):
        """
        :param direction: Direction of optimization.
        :param saved_files_path: Directory to save logging files.
        :param optimization_metric: Metric to optimize.
        :param validation_metric: Score function or loss function with signature validation_metric(y_true, y_pred),
            must return float/integer value of metric.
        :param model_type: name of trained model
        """

        self.validation_metric = validation_metric
        self.direction = direction
        self.saved_files_path = saved_files_path
        self.optimization_metric = optimization_metric
        self.model_type = model_type
        if os.path.isdir(self.saved_files_path):
            shutil.rmtree(self.saved_files_path)
        os.makedirs(r"{}/saved_models/".format(saved_files_path))
        os.makedirs(r"{}/saved_studies/".format(saved_files_path))
        os.makedirs(r"{}/saved_utils/".format(saved_files_path))

    @abstractmethod
    def _initiate_neptune_run(self, neptune_run_params, upload_files):
        return

    @abstractmethod
    def _initiate_mlflow_run(
        self, tracking_uri, experiment_name, mlflow_run_params, upload_files
    ):
        return

    def init_run(
        self,
        logging_server: Literal["neptune", "optuna"],
        upload_files: Iterable[str] = [],
        **run_params
    ) -> str:
        """Initiation of logging server run.

        :param logging_server: logging server
        :param upload_files: List of paths to files which will be logged in initiated run.
        :param run_params: If logging server == "mlflow":
            Mlflow run parameters as kwargs (will be passed to `mlflow.start_run <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run>`_).

            If logging_server == "neptune":
            Kwarg neptune_run_params as dict of Neptune run parameters (will be passed to `neptune.init_run <https://docs.neptune.ai/api-reference/neptune#.init_run>`_).

        :return: run id of created run.
        """

        if logging_server == "neptune":

            run_id = self._base_initiate_neptune_run(
                upload_files=upload_files, **run_params
            )
            self._initiate_neptune_run(upload_files=upload_files, **run_params)

        elif logging_server == "mlflow":

            run_id = self._base_initiate_mlflow_run(
                upload_files=upload_files, **run_params
            )
            self._initiate_mlflow_run(upload_files=upload_files, **run_params)

        return run_id

    def _base_initiate_neptune_run(self, neptune_run_params, upload_files=[]):

        self.run = neptune.init(**neptune_run_params, run=None)
        self.run["direction"] = self.direction
        self.run["model_type"] = self.model_type
        self.run["optimization_metric"] = self.optimization_metric

        file_name = r"{}/saved_utils/validation_metric.pickle".format(
            self.saved_files_path
        )
        with open(file_name, "wb") as f:
            pickle.dump(self.validation_metric, f)
        self.run[file_name.split(self.saved_files_path)[-1]].upload(
            file_name, wait=True
        )
        os.remove(file_name)

        if len(upload_files) > 0:
            for file_name in upload_files:
                self.run["saved_files/{}".format(file_name.split("/")[-1])].upload(
                    file_name, wait=True
                )

        self.logging_server = "neptune"

        return self.run["sys"]["id"].fetch()

    def _base_initiate_mlflow_run(
        self,
        experiment_name,
        upload_files=[],
        tracking_uri=None,
        mlflow_run_params=None,
    ):

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(**mlflow_run_params)
        mlflow.log_param("direction", self.direction)
        mlflow.log_param("model_type", self.model_type)
        mlflow.log_param("optimization_metric", self.optimization_metric)

        file_name = r"{}/saved_utils/validation_metric.pickle".format(
            self.saved_files_path
        )
        with open(file_name, "wb") as f:
            pickle.dump(self.validation_metric, f)
        mlflow.log_artifact(
            file_name, file_name.split(self.saved_files_path)[-1].split("/")[1]
        )
        os.remove(file_name)

        if (len(upload_files) > 0) and (type(upload_files != str)):
            for file_name in upload_files:
                mlflow.log_artifact(
                    file_name, "saved_files/{}".format(file_name.split("/")[-1])
                )

        self.logging_server = "mlflow"

        return self.run.info.run_id

    def _log_metrics_neptune(self, metrics_dict, trial):

        for metric in metrics_dict.keys():
            self.run[metric].log(metrics_dict[metric])

    def _log_metrics_mlflow(self, metrics_dict, trial):

        for metric in metrics_dict.keys():
            mlflow.log_metric(metric, metrics_dict[metric], step=trial.number)

    def _log_metrics(self, metrics_dict, trial):

        if self.logging_server == "neptune":
            self._log_metrics_neptune(metrics_dict, trial)

        elif self.logging_server == "mlflow":
            self._log_metrics_mlflow(metrics_dict, trial)

    def _log_params_neptune(self, params):

        self.run["params"].log(params)

    def _log_params_mlflow(self, params):

        try:
            mlflow.tracking.MlflowClient().download_artifacts(
                self.run.info.run_uuid,
                "params.json",
                dst_path=r"{}/saved_utils".format(self.saved_files_path),
            )
            with open(
                r"{}/saved_utils/params.json".format(self.saved_files_path)
            ) as file:
                params_logged = json.load(file)
        except:
            params_logged = []

        params_logged.append(params)
        mlflow.log_dict(params_logged, "params.json")

    def _log_params(self, params):

        if self.logging_server == "neptune":
            self._log_params_neptune(params)

        elif self.logging_server == "mlflow":
            self._log_params_mlflow(params)

    def _log_files_neptune(self, file_paths):
        for file_path in file_paths:
            self.run[file_path.split(self.saved_files_path)[-1]].upload(
                file_path, wait=True
            )
            os.remove(file_path)

    def _log_files_mlflow(self, file_paths):
        for file_path in file_paths:
            mlflow.log_artifact(
                file_path, file_path.split(self.saved_files_path)[-1].split("/")[1]
            )
            os.remove(file_path)

    def _log_files(self, file_paths):

        if self.logging_server == "neptune":
            self._log_files_neptune(file_paths)
        elif self.logging_server == "mlflow":
            self._log_files_mlflow(file_paths)
