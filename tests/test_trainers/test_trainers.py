import mlflow
import optuna
import pytest
from mlexp.trainers import LgbTrainer, SklearnTrainer, TorchTrainer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tests.data_dummy import TrainerDummyData
from tests.test_trainers.params_funcs import (
    LGB_fixed_params_func,
    LGB_search_params_func,
    Sklearn_fixed_params_func,
    Sklearn_search_params_func,
    Torch_fixed_params_func,
    Torch_search_params_func,
)
from tests.test_trainers.torch_utils import data_loaders, metrics_callback, nn_model
from tests.test_trainers.utils import add_server_auth

TRAINERS = {"LGB": LgbTrainer, "Sklearn": SklearnTrainer, "Torch": TorchTrainer}
SAMPLERS = {"TPESampler": optuna.samplers.TPESampler}
PARAMS_FUNCS = {
    "LGB": {"search": LGB_search_params_func, "fixed": LGB_fixed_params_func},
    "Sklearn": {
        "search": Sklearn_search_params_func,
        "fixed": Sklearn_fixed_params_func,
    },
    "Torch": {"search": Torch_search_params_func, "fixed": Torch_fixed_params_func},
}
RUN_PARAMS = {
    "neptune": {"neptune_run_params": {"name": "Test", "description": "1"}},
    "mlflow": {
        "experiment_name": "tests",
        "mlflow_run_params": {"run_name": "Test", "tags": {"Test tag": "1"}},
    },
}
VALIDATION_METRICS = [mean_absolute_error, mean_squared_error]
OPTIMIZATION_METRICS = ["metric_mean_cv", "metric_test"]
dummy_data = TrainerDummyData()


@pytest.mark.parametrize(
    "trainer_name,trainer_specific_params",
    [
        ["LGB", {"use_average_n_estimators_on_test_fold": True}],
        ["LGB", {"use_average_n_estimators_on_test_fold": False}],
        ["Sklearn", {"sklearn_estimator": Ridge}],
        [
            "Torch",
            {
                "nn_model_module": nn_model,
                "data_loaders_module": data_loaders,
                "metrics_callback_module": metrics_callback,
                "use_average_epochs_on_test_fold": True,
            },
        ],
        [
            "Torch",
            {
                "nn_model_module": nn_model,
                "data_loaders_module": data_loaders,
                "metrics_callback_module": metrics_callback,
                "use_average_epochs_on_test_fold": False,
            },
        ],
    ],
)
class TestTrainer:
    def init_trainer(
        self,
        trainer_name,
        trainer_specific_params,
        validation_metric,
        direction,
        optimization_metric,
    ):

        return TRAINERS[trainer_name](
            validation_metric=validation_metric,
            direction=direction,
            saved_files_path=r"temp_files/",
            optimization_metric=optimization_metric,
            **trainer_specific_params
        )

    @pytest.mark.dependency(name="test_init_trainer")
    def test_init_trainer(self, trainer_name, trainer_specific_params):
        self.init_trainer(
            trainer_name,
            trainer_specific_params,
            VALIDATION_METRICS[0],
            "minimize",
            OPTIMIZATION_METRICS[0],
        )

    @pytest.mark.parametrize("logging_server", list(RUN_PARAMS.keys()))
    @pytest.mark.parametrize("upload_files", [["requirements.txt"], []])
    @pytest.mark.dependency(name="test_init_run", depends=["test_init_trainer"])
    def test_init_run(
        self,
        trainer_name,
        trainer_specific_params,
        logging_server,
        upload_files,
        logging_server_auth,
    ):
        trainer = self.init_trainer(
            trainer_name,
            trainer_specific_params,
            VALIDATION_METRICS[0],
            "minimize",
            OPTIMIZATION_METRICS[0],
        )
        run_params = add_server_auth(
            logging_server, RUN_PARAMS[logging_server], logging_server_auth
        )

        trainer.init_run(logging_server, upload_files, **run_params)

        if logging_server == "neptune":
            trainer.run.stop()

        elif logging_server == "mlflow":
            mlflow.end_run()

    @pytest.mark.parametrize("logging_server", list(RUN_PARAMS.keys()))
    @pytest.mark.parametrize("sampler_name", list(SAMPLERS.keys()))
    @pytest.mark.parametrize("n_trials", [1, 3])
    @pytest.mark.parametrize("search_hparams", ["search", "fixed"])
    @pytest.mark.parametrize("validation_metric", VALIDATION_METRICS)
    @pytest.mark.parametrize("direction", ["maximize", "minimize"])
    @pytest.mark.parametrize("optimization_metric", OPTIMIZATION_METRICS)
    @pytest.mark.dependency(
        name="test_train", depends=["test_init_trainer", "test_init_run"]
    )
    def test_train(
        self,
        trainer_name,
        trainer_specific_params,
        validation_metric,
        direction,
        optimization_metric,
        logging_server,
        sampler_name,
        search_hparams,
        n_trials,
        logging_server_auth,
    ):
        trainer = self.init_trainer(
            trainer_name,
            trainer_specific_params,
            validation_metric,
            direction,
            optimization_metric,
        )

        run_params = add_server_auth(
            logging_server, RUN_PARAMS[logging_server], logging_server_auth
        )

        run_id = trainer.init_run(logging_server, **run_params)

        trainer.train(
            X=dummy_data.X,
            y=dummy_data.y,
            cv=dummy_data.cv_list,
            n_trials=n_trials,
            params_func=PARAMS_FUNCS[trainer_name][search_hparams],
            sampler=SAMPLERS[sampler_name](),
        )
