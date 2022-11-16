import json
import pickle


def _save_metric_curves(metric_curves, saved_files_path, trial_number):
    file_name = r"{}/saved_metric_curves/metric_curves_trial_{}.json".format(
        saved_files_path, trial_number
    )

    with open(file_name, "w") as f:
        json.dump(metric_curves, f)

    return file_name


def _save_models(models, saved_files_path, trial_number):
    model_file_paths = []

    for fold_num, model in enumerate(models):
        if fold_num == len(models) - 1:
            fold_num = "test"

        model_file_path = r"{}/saved_models/model_trial_{}_fold_{}.pickle".format(
            saved_files_path, trial_number, fold_num
        )

        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        model_file_paths.append(model_file_path)

    return model_file_paths
