from abc import ABC, abstractmethod
from typing import Union


class _BaseServerInference(ABC):
    """Base class for inference any model from any server."""

    def __init__(self, downloaded_files_path):
        self.downloaded_files_path = downloaded_files_path

    @abstractmethod
    def get_run_params(self) -> tuple:
        """Get parameters of run.

        :return: direction of optimization, model type, validation metric.

            If torch_inference, also returns use_average_epochs_on_test_fold.

            If lgb_inference, also returns use_average_n_estimators_on_test_fold.
        """

        return

    @abstractmethod
    def get_best_step(self, direction: str, metric: str) -> int:
        """Get index of best step (with best value of validation metric) in run.

        :param direction: Direction of optimization.
        :param metric: name of metric to get from mlflow server
        :return: Index of best step in run.
        """

        return

    @abstractmethod
    def get_metric(self, step: int, metric: str) -> dict:
        """Get metric from step.

        :param step: Index of step in run.
        :param metric: name of metric to get from mlflow server
        :return: Dictionary with metrics.

            Key of dictionary: metric
        """

        return

    @abstractmethod
    def get_step_params(self, step: int) -> dict:
        """Get hyperparameters of particular step in run.

        :param step: Index of step in run.
        :type step: int
        :return: Hyperparameters of step in run.
        :rtype: dict
        """

        return

    @abstractmethod
    def get_file(self, server_file_path: str, download_file_path: str) -> str:
        """Download file with name server_file_name from neptune run

        :param server_file_path: Path of file to download from neptune
        :param download_file_path: Path indicating where to save file
        :return: path to downloaded file
        """

        return

    @abstractmethod
    def stop(self):
        pass


class _BaseModelInference(ABC):
    """Downloading logged parameters, hyperparameters and lightgbm model."""

    @abstractmethod
    def get_params_model(
        self,
        metric: str = "metric_mean_cv",
        step: Union[int, str] = "best",
        fold_num: Union[int, str] = "test",
        trained_model: bool = True,
    ) -> dict:
        """Get logged parameters, hyperparameters, metrics and lightgbm model from particular step and fold in run.

        :param metric: Metric to get from server and find best step.
        :param step: Index of step in run. If int - index of step.
            If 'best' - best step (with best value of validation metric).
        :param fold_num: Cross validation fold.
            If int - index of validation fold. If 'test' - test fold.
        :param trained_model: Whether to downloaded trained model or initialise new model.
        :return: Dictionary with downloaded parameters of run, hyperparameters, model and logged metrics.
        """

        return
