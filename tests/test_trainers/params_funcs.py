def LGB_search_params_func(trial):
    return {
        "model_params": {
            "objective": "l2",
            "boosting": "gbdt",
            "n_jobs": -1,
            "n_estimators": 5,
            "random_state": 1230011199988,
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.01, 1),
            "num_leaves": 10,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.5),
        },
        "lgb_data_set_params": {},
    }


def LGB_fixed_params_func(trial):
    return {
        "model_params": {
            "objective": "l2",
            "boosting": "gbdt",
            "n_jobs": -1,
            "n_estimators": 5,
            "random_state": 1230011199988,
            "num_leaves": 10,
            "learning_rate": 0.5,
        },
        "lgb_data_set_params": {},
    }


# Function which returns set of parameters
def Sklearn_search_params_func(trial):
    return {
        "model_params": {
            "random_state": 1230011199988,
            "max_iter": 200,
            "alpha": trial.suggest_float("alpha", 0.01, 1),
        },
    }


def Sklearn_fixed_params_func(trial):
    return {
        "model_params": {"random_state": 1230011199988, "max_iter": 200, "alpha": 0.1},
    }


def Torch_search_params_func(trial):
    return {
        "model_params": {
            "optimizer": "SGD",
            "objective": "MSE",
            "lr": trial.suggest_float("lr", 0.001, 1),
            "input_size": 10,
            "embedding_size": 20,
            "weight_decay": 0.05,
        },
        "trainer_params": {"min_epochs": 2, "max_epochs": 4, "num_sanity_val_steps": 0},
        "data_loaders_params": {},
        "EarlyStopping_params": {
            "min_delta": 0.001,
            "patience": 3,
            "monitor": "validation_metric",
        },
    }


def Torch_fixed_params_func(trial):
    return {
        "model_params": {
            "optimizer": "SGD",
            "objective": "MSE",
            "lr": 0.05,
            "input_size": 10,
            "embedding_size": 20,
            "weight_decay": 0.05,
        },
        "trainer_params": {"min_epochs": 2, "max_epochs": 4, "num_sanity_val_steps": 0},
        "data_loaders_params": {},
        "EarlyStopping_params": {
            "min_delta": 0.001,
            "patience": 3,
            "monitor": "validation_metric",
        },
    }
