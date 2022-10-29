import ast
import json
import os
import pickle

import mlflow
import pandas as pd

from mlexp.inference._base_inference._base_inference import _BaseServerInference


class _MlflowInference(_BaseServerInference):
    """Base class for inference from mlflow."""

    def __init__(self, run_params: dict, downloaded_files_path):

        self.run_id = run_params["run_id"]
        self.mlflow_client = mlflow.tracking.MlflowClient(run_params["tracking_uri"])
        super().__init__(downloaded_files_path)

        pass

    def get_run_params(self) -> tuple:
        """Get parameters of run.

        :return: direction of optimization, model type, validation metric.

            If torch_inference, also returns use_average_epochs_on_test_fold.

            If lgb_inference, also returns use_average_n_estimators_on_test_fold.
        """

        direction = self.mlflow_client.get_run(self.run_id).data.params["direction"]
        model_type = self.mlflow_client.get_run(self.run_id).data.params["model_type"]
        validation_metric_directory = self.mlflow_client.download_artifacts(
            self.run_id,
            "saved_utils/validation_metric.pickle",
            r"{}/downloaded_utils/".format(self.downloaded_files_path),
        )
        with open(validation_metric_directory, "rb") as f:
            validation_metric = pickle.load(f)

        try:
            use_average_epochs_on_test_fold = self.mlflow_client.get_run(
                self.run_id
            ).data.params["use_average_epochs_on_test_fold"]

            return (
                direction,
                model_type,
                validation_metric,
                use_average_epochs_on_test_fold,
            )

        except:
            try:
                use_average_n_estimators_on_test_fold = self.mlflow_client.get_run(
                    self.run_id
                ).data.params["use_average_n_estimators_on_test_fold"]

                return (
                    direction,
                    model_type,
                    validation_metric,
                    use_average_n_estimators_on_test_fold,
                )

            except:

                return (direction, model_type, validation_metric)

    def get_best_step(self, direction: str, metric: str) -> int:
        """Get index of best step (with best value of validation metric) in run.

        :param direction: Direction of optimization.
        :param metric: name of metric to get from mlflow server
        :return: Index of best step in run.
        """

        metric_df = pd.DataFrame(
            {
                metric: [
                    step.value
                    for step in self.mlflow_client.get_metric_history(
                        self.run_id, metric
                    )
                ],
                "step": [
                    step.step
                    for step in self.mlflow_client.get_metric_history(
                        self.run_id, metric
                    )
                ],
            }
        )

        if direction == "maximize":
            step = metric_df[metric_df[metric] == metric_df[metric].max()][
                "step"
            ].values[0]
        else:
            step = metric_df[metric_df[metric] == metric_df[metric].min()][
                "step"
            ].values[0]

        return int(step)

    def get_metric(self, step: int, metric: str) -> dict:
        """Get metric from step.

        :param step: Index of step in run.
        :param metric: name of metric to get from mlflow server
        :return: Dictionary with metrics.

            Key of dictionary: metric
        """

        return {
            metric: [
                step.value
                for step in self.mlflow_client.get_metric_history(self.run_id, metric)
            ][step]
        }

    def get_step_params(self, step: int) -> dict:
        """Get hyperparameters of particular step in run.

        :param step: Index of step in run.
        :return: Hyperparameters of step in run.
        """

        params_directory = self.mlflow_client.download_artifacts(
            self.run_id, "params.json", r"{}/".format(self.downloaded_files_path)
        )
        with open(params_directory) as file:
            steps_params = json.load(file)
        steps_params = [str(x) for x in steps_params]
        steps_params = pd.DataFrame({"value": steps_params})
        steps_params["step"] = range(0, len(steps_params))

        step_params = ast.literal_eval(
            steps_params[steps_params["step"] == step]["value"].values[0]
        )

        return step_params

    def get_file(self, server_file_path: str, download_file_path: str) -> str:
        """Download file with name server_file_name from neptune run

        :param server_file_path: Path of file to download from neptune
        :param download_file_path: Path indicating where to save file
        :return: path to downloaded file
        """

        file_directory, file_name = os.path.split(download_file_path)
        file_path = self.mlflow_client.download_artifacts(
            self.run_id, server_file_path, file_directory
        )

        return file_path

    def stop(self):

        pass
