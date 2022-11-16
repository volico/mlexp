def server_auth(logging_server, run_id, logging_server_auth):

    if logging_server == "neptune":
        inference_server_params = {"run": run_id}
        inference_server_params["project"] = logging_server_auth["neptune"]
    elif logging_server == "mlflow":
        inference_server_params = {"run_id": run_id}
        inference_server_params["tracking_uri"] = logging_server_auth["mlflow"]

    return inference_server_params
