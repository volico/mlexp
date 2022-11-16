def add_server_auth(logging_server, run_params, logging_server_auth):
    if logging_server == "neptune":
        run_params["neptune_run_params"]["project"] = logging_server_auth["neptune"]
    elif logging_server == "mlflow":
        run_params["tracking_uri"] = logging_server_auth["mlflow"]

    return run_params
