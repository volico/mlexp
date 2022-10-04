import pytest


def pytest_addoption(parser):
    parser.addoption("--neptune_project", action="store", type=str,
                     help='Neptune project in form <neptune user>/<neptune project>')
    parser.addoption("--mlflow_tracking_uri", action="store", type=str, default='http://127.0.0.1:5000/',
                     help='Mlflow server uri')


@pytest.fixture(scope='module', autouse=True)
def logging_server_auth(request):
    return {'neptune': request.config.getoption("--neptune_project"),
            'mlflow': request.config.getoption("--mlflow_tracking_uri")}


@pytest.fixture(scope='module', autouse=True)
def neptune_run_params(request):
    return {'neptune_run_params': {'project': request.config.getoption("--neptune_project"), 'name': 'Test',
                                   'description': '1'}}


@pytest.fixture(scope='module', autouse=True)
def mlflow_run_params(request):
    return {'experiment_name': 'tests', 'tracking_uri': request.config.getoption("--mlflow_tracking_uri"),
            'mlflow_run_params': {'run_name': 'Test', 'tags': {'Test tag': '1'}}}
