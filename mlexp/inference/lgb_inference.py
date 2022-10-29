import os
import pickle
import shutil
from typing import Literal, Union

from mlexp.inference._base_inference import _BaseModelInference, SERVER_INFERENCES


class LgbInference(_BaseModelInference):
    """Downloading logged parameters, hyperparameters and scikit-learn model from particular server."""

    def __init__(
        self,
        downloaded_files_path: str,
        inference_server_params: dict,
        server: Literal[list(SERVER_INFERENCES.keys())],
    ):
        """
        :param downloaded_files_path: Directory to which files from mlflow server will be downloaded.
        :param inference_server_params: Server params to download file from.
            For mlflow - dict with 'run_id' of particular run and 'tracking_uri' of your mlflow server.
            For neptune - 'project' in form of ({User}/{Project Name}) and 'run' as run id of your experiment.
        :param server: Type of server.
        """

        self.downloaded_files_path = downloaded_files_path
        self.downloaded_params = {}
        self.server_inference = SERVER_INFERENCES[server](
            inference_server_params, downloaded_files_path
        )

        if os.path.isdir(self.downloaded_files_path):
            shutil.rmtree(self.downloaded_files_path)
        os.makedirs(r"{}/downloaded_models/".format(downloaded_files_path))
        os.makedirs(r"{}/downloaded_utils/".format(downloaded_files_path))
        os.makedirs(r"{}/downloaded_studies/".format(downloaded_files_path))

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

        (
            self.downloaded_params["direction"],
            self.downloaded_params["model_type"],
            self.downloaded_params["validation_metric"],
            self.downloaded_params["use_average_n_estimators_on_test_fold"],
        ) = self.server_inference.get_run_params()
        if step == "best":
            step = self.server_inference.get_best_step(
                self.downloaded_params["direction"], metric
            )
        self.downloaded_params["step"] = step
        self.downloaded_params["metric"] = self.server_inference.get_metric(
            step, metric
        )
        self.downloaded_params["params"] = self.server_inference.get_step_params(step)
        optuna_study_path = self.server_inference.get_file(
            r"saved_studies/optuna_study_{}.pickle".format(step),
            r"{}/downloaded_studies/optuna_study_{}.pickle".format(
                self.downloaded_files_path, step
            ),
        )
        with open(optuna_study_path, "rb") as f:
            self.downloaded_params["optuna_study"] = pickle.load(f)

        if trained_model:
            trained_model_path = self.server_inference.get_file(
                r"saved_models/model_trial_{}_fold_{}.pickle".format(step, fold_num),
                r"{}/downloaded_models/model_trial_{}_fold_{}.pickle".format(
                    self.downloaded_files_path, step, fold_num
                ),
            )
            with open(trained_model_path, "rb") as f:
                self.downloaded_params["trained_model"] = pickle.load(f)

        else:
            pass

        self.server_inference.stop()

        return self.downloaded_params
