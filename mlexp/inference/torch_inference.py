import importlib.util
import os
import pickle
import shutil
import sys
from types import ModuleType
from typing import Literal, Union

import pytorch_lightning as pl

from mlexp.inference._base_inference import _BaseModelInference, SERVER_INFERENCES


class TorchInference(_BaseModelInference):
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

    def _load_module(self, module_name: str, module_directory: str) -> ModuleType:
        """Load module from module_directory with name module_name
        :param module_name: Name of module to use
        :param module_directory: Path to module
        :return: Loaded module
        """

        spec = importlib.util.spec_from_file_location(module_name, module_directory)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return sys.modules[module_name]

    def get_params_model(
        self,
        metric: str = "metric_mean_cv",
        step: Union[int, str] = "best",
        fold_num: Union[int, str] = "test",
        trained_model: bool = True,
        nn_model_params: dict = {},
    ) -> dict:
        """Get logged parameters, hyperparameters, metrics and pytorch-lightning neural network from particular step and fold in run.

        :param metric: Metric to get from server and find best step.
        :param step: Index of step in run. If int - index of step.
            If 'best' - best step (with best value of validation metric).
        :param fold_num: Cross validation fold.
            If int - index of validation fold. If 'test' - test fold.
        :param trained_model: Whether to downloaded trained model or initialise new model.
        :param nn_model_params: Dictionary with hyperparameters of neural network to be overwritten.
        :return: Dictionary with downloaded parameters of run, hyperparameters, model and logged metrics.
        """

        (
            self.downloaded_params["direction"],
            self.downloaded_params["model_type"],
            self.downloaded_params["validation_metric"],
            self.downloaded_params["use_average_epochs_on_test_fold"],
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
        initial_min_epoches = self.downloaded_params["params"]["trainer_params"][
            "min_epochs"
        ]
        initial_max_epoches = self.downloaded_params["params"]["trainer_params"][
            "max_epochs"
        ]

        nn_model_module_path = self.server_inference.get_file(
            "saved_utils/nn_model_module.py",
            r"{}/downloaded_utils/nn_model_module.py".format(
                self.downloaded_files_path
            ),
        )
        data_loaders_module_path = self.server_inference.get_file(
            "saved_utils/data_loaders_module.py",
            r"{}/downloaded_utils/data_loaders_module.py".format(
                self.downloaded_files_path
            ),
        )
        metrics_callback_module_path = self.server_inference.get_file(
            "saved_utils/metrics_callback_module.py",
            r"{}/downloaded_utils/metrics_callback_module.py".format(
                self.downloaded_files_path
            ),
        )
        self.downloaded_params["nn_model"] = self._load_module(
            "nn_model_module", nn_model_module_path
        ).nn_model
        self.downloaded_params["data_loaders"] = self._load_module(
            "data_loaders_module", data_loaders_module_path
        ).train_val_data_loaders
        self.downloaded_params["metrics_callback"] = self._load_module(
            "metrics_callback_module", metrics_callback_module_path
        ).MetricsCallback

        if trained_model:

            weigths_args_path = self.server_inference.get_file(
                r"saved_models/model_trial_{}_fold_{}.ckpt".format(step, fold_num),
                r"{}/downloaded_models/model_trial_{}_fold_{}.ckpt".format(
                    self.downloaded_files_path, step, fold_num
                ),
            )
            self.downloaded_params["trained_model"] = self.downloaded_params[
                "nn_model"
            ].load_from_checkpoint(weigths_args_path, **nn_model_params)

            if self.downloaded_params["use_average_epochs_on_test_fold"]:
                self.downloaded_params["params"]["trainer_params"][
                    "min_epochs"
                ] = self.downloaded_params["params"]["validation_mean_epochs"]
                self.downloaded_params["params"]["trainer_params"][
                    "max_epochs"
                ] = self.downloaded_params["params"]["validation_mean_epochs"]

            metrics_callback = self.downloaded_params["metrics_callback"]()
            self.downloaded_params["trainer"] = pl.Trainer(
                **self.downloaded_params["params"]["trainer_params"],
                callbacks=[
                    pl.callbacks.early_stopping.EarlyStopping(
                        self.downloaded_params["params"]["EarlyStopping_params"]
                    ),
                    metrics_callback,
                ]
            )

            self.downloaded_params["params"]["trainer_params"][
                "min_epochs"
            ] = initial_min_epoches
            self.downloaded_params["params"]["trainer_params"][
                "max_epochs"
            ] = initial_max_epoches

        else:

            self.downloaded_params["model"] = self.downloaded_params["nn_model"](
                **{
                    **self.downloaded_params["params"]["model_params"],
                    **nn_model_params,
                    **{
                        "validation_metric": self.downloaded_params["validation_metric"]
                    },
                }
            )

            if self.downloaded_params["use_average_epochs_on_test_fold"]:
                self.downloaded_params["params"]["trainer_params"][
                    "min_epochs"
                ] = self.downloaded_params["params"]["validation_mean_epochs"]
                self.downloaded_params["params"]["trainer_params"][
                    "max_epochs"
                ] = self.downloaded_params["params"]["validation_mean_epochs"]

            metrics_callback = self.downloaded_params["metrics_callback"]()
            self.downloaded_params["trainer"] = pl.Trainer(
                **self.downloaded_params["params"]["trainer_params"],
                callbacks=[
                    pl.callbacks.early_stopping.EarlyStopping(
                        self.downloaded_params["params"]["EarlyStopping_params"]
                    ),
                    metrics_callback,
                ]
            )

            self.downloaded_params["params"]["trainer_params"][
                "min_epochs"
            ] = initial_min_epoches
            self.downloaded_params["params"]["trainer_params"][
                "max_epochs"
            ] = initial_max_epoches

        return self.downloaded_params
