import ast
import pickle

import neptune.new as neptune

from mlexp.inference._base_inference._base_inference import _BaseServerInference


class _NeptuneInference(_BaseServerInference):
    """Base class for inference from neptune.ai."""

    def __init__(self, run_params: dict, downloaded_files_path: str):
        """
        :param run_params: Neptune run parameters (will be passed to `neptune.init_run <https://docs.neptune.ai/api-reference/neptune#.init_run>`_).
            Parameters must be such that an existing experiment is selected.
        """
        self.run = neptune.init(**run_params)
        super().__init__(downloaded_files_path)

    def get_run_params(self) -> tuple:
        """Get parameters of run.

        :return: direction of optimization, model type, validation metric.

            If torch_inference, also returns use_average_epochs_on_test_fold.

            If lgb_inference, also returns use_average_n_estimators_on_test_fold.
        """

        direction = self.run["direction"].fetch()
        model_type = self.run["model_type"].fetch()
        self.run["saved_utils/validation_metric.pickle"].download(
            destination=r"{}/downloaded_utils/validation_metric.pickle".format(
                self.downloaded_files_path
            )
        )
        with open(
            r"{}/downloaded_utils/validation_metric.pickle".format(
                self.downloaded_files_path
            ),
            "rb",
        ) as f:
            validation_metric = pickle.load(f)

        try:
            use_average_epochs_on_test_fold = self.run[
                "use_average_epochs_on_test_fold"
            ].fetch()

            return (
                direction,
                model_type,
                validation_metric,
                use_average_epochs_on_test_fold,
            )

        except:
            try:
                use_average_n_estimators_on_test_fold = self.run[
                    "use_average_n_estimators_on_test_fold"
                ].fetch()

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

        metric_list = self.run[metric].fetch_values()

        if direction == "maximize":
            step = metric_list[metric_list["value"] == metric_list["value"].max()][
                "step"
            ].values[0]
        else:
            step = metric_list[metric_list["value"] == metric_list["value"].min()][
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

        metric_list = self.run[metric].fetch_values()

        return {metric: metric_list[metric_list["step"] == step]["value"].values[0]}

    def get_step_params(self, step: int) -> dict:
        """Get hyperparameters of particular step in run.

        :param step: Index of step in run.
        :type step: int
        :return: Hyperparameters of step in run.
        :rtype: dict
        """

        steps_params = self.run["params"].fetch_values()

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

        self.run[server_file_path].download(destination=download_file_path)

        return download_file_path

    def stop(self):

        self.run.stop()
