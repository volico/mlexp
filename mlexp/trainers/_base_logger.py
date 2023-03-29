from abc import ABC, abstractmethod
import os
import shutil
import pickle
import json
import numpy as np
from typing import Callable, Iterable, Literal, TypedDict
import neptune.new as neptune
import mlflow


class _Logger(ABC):
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

        self.logger = None

        super().__init__()

    def init_run(
        self,
        logging_server: Literal["neptune", "mlflow"],
        upload_files: Iterable[str] = [],
        **run_params
    ) -> str:
        """Initiation of logging server run.

        :param logging_server: logging server ("neptune" or "mlflow").
        :param upload_files: List of paths to files which will be logged in initiated run.
        :param run_params: If logging server == "mlflow":
            Mlflow run parameters as kwargs (will be passed to `mlflow.start_run <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run>`_).

            If logging_server == "neptune":
            Kwarg neptune_run_params as dict of Neptune run parameters (will be passed to `neptune.init_run <https://docs.neptune.ai/api-reference/neptune#.init_run>`_).

        :return: run id of created run.
        """

        if logging_server == "neptune":
            self.logger = NeptuneLogger(self.saved_files_path)
        elif logging_server == "mlflow":
            self.logger = MLFlowLogger(self.saved_files_path)

        return self.logger.init_run(
            self.direction,
            self.optimization_metric,
            self.validation_metric,
            self.model_type,
            upload_files,
            run_params,
            self._get_run_info(),
        )

    def log_metrics(self, metrics_dict, trial):
        self.logger.log_metrics(metrics_dict, trial)

    def log_params(self, params):
        self.logger.log_params(params)

    def log_files(self, file_paths):
        self.logger.log_files(file_paths)

    def stop_run(self):
        self.logger.stop_run()

    @abstractmethod
    def _get_run_info(
        self,
    ) -> TypedDict("run_info", {"metadata": dict, "artifact_paths": list[str]}):
        """Constructing dict with info to upload to the run."""
        pass


class LoggerStrategy(ABC):
    def __init__(self, saved_files_path: str):
        self.saved_files_path = saved_files_path

    @abstractmethod
    def init_run(
        self,
        direction,
        optimization_metric,
        validation_metric,
        model_type,
        upload_files: Iterable[str],
        run_params,
        model_info,
    ) -> str:
        pass

    @abstractmethod
    def log_metrics(self, metrics_dict, trial):
        pass

    @abstractmethod
    def log_params(self, params):
        pass

    @abstractmethod
    def log_files(self, file_paths):
        pass

    @abstractmethod
    def stop_run(self):
        pass


class NeptuneLogger(LoggerStrategy):
    def init_run(
        self,
        direction,
        optimization_metric,
        validation_metric,
        model_type,
        upload_files: Iterable[str],
        run_params,
        model_info,
    ) -> str:
        self.run = neptune.init(**run_params, run=None)
        self.run["direction"] = direction
        self.run["model_type"] = model_type
        self.run["optimization_metric"] = optimization_metric

        file_name = r"{}/saved_utils/validation_metric.pickle".format(
            self.saved_files_path
        )
        with open(file_name, "wb") as f:
            pickle.dump(validation_metric, f)
        self.run[file_name.split(self.saved_files_path)[-1]].upload(
            file_name, wait=True
        )
        os.remove(file_name)

        if len(upload_files) > 0:
            for file_name in upload_files:
                self.run["saved_files/{}".format(file_name.split("/")[-1])].upload(
                    file_name, wait=True
                )

        if len(model_info["metadata"]) > 0:
            for key, value in model_info["metadata"].items():
                self.run[key] = value

        if len(model_info["artifact_paths"]) > 0:
            for path in model_info["artifact_paths"]:
                self.run[path.split(self.saved_files_path)[-1]].upload(path, wait=True)

        return self.run["sys"]["id"].fetch()

    def log_metrics(self, metrics_dict, trial):
        for metric in metrics_dict.keys():
            self.run[metric].log(metrics_dict[metric])

    def log_params(self, params):
        self.run["params"].log(params)

    def log_files(self, file_paths):
        for file_path in file_paths:
            self.run[file_path.split(self.saved_files_path)[-1]].upload(
                file_path, wait=True
            )
            os.remove(file_path)

    def stop_run(self):
        self.run.stop()


class MLFlowLogger(LoggerStrategy):
    def init_run(
        self,
        direction,
        optimization_metric,
        validation_metric,
        model_type,
        upload_files: Iterable[str],
        run_params,
        model_info,
    ) -> str:
        mlflow.set_tracking_uri(run_params["tracking_uri"])
        mlflow.set_experiment(run_params["experiment_name"])
        self.run = mlflow.start_run(**run_params["start_run_params"])
        mlflow.log_param("direction", direction)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("optimization_metric", optimization_metric)

        file_name = r"{}/saved_utils/validation_metric.pickle".format(
            self.saved_files_path
        )
        with open(file_name, "wb") as f:
            pickle.dump(validation_metric, f)
        mlflow.log_artifact(
            file_name, file_name.split(self.saved_files_path)[-1].split("/")[1]
        )
        os.remove(file_name)

        if (len(upload_files) > 0) and (type(upload_files != str)):
            for file_name in upload_files:
                mlflow.log_artifact(
                    file_name, "saved_files/{}".format(file_name.split("/")[-1])
                )

        if len(model_info["metadata"]) > 0:
            for key, value in model_info["metadata"].items():
                mlflow.log_param(key, value)

        if len(model_info["artifact_paths"]) > 0:
            for path in model_info["artifact_paths"]:
                mlflow.log_artifact(
                    path, path.split(self.saved_files_path)[-1].split("/")[1]
                )

        return self.run.info.run_id

    def log_metrics(self, metrics_dict, trial):
        for metric in metrics_dict.keys():
            mlflow.log_metric(metric, metrics_dict[metric], step=trial.number)

    def log_params(self, params):
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

    def log_files(self, file_paths):
        for file_path in file_paths:
            mlflow.log_artifact(
                file_path, file_path.split(self.saved_files_path)[-1].split("/")[1]
            )
            os.remove(file_path)

    def stop_run(self):
        mlflow.end_run()
