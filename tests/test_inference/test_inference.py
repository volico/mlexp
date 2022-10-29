import pytest

from mlexp.inference import LgbInference, SklearnInference, TorchInference

from tests.test_inference.utils import server_auth

INFERENCES = {"LGB": LgbInference, "Sklearn": SklearnInference, "Torch": TorchInference}

# @pytest.mark.last
# @pytest.mark.parametrize('inference_name', ['LGB', 'Sklearn', 'Torch'])
# TODO figure out how to test fixed set of runs from neptune and mlflow
# @pytest.mark.parametrize('run_id, logging_server', pytest.runs)
# class TestInference:
#
#     def init_inference(self, inference_name, inference_server_params, logging_server):
#         return INFERENCES[inference_name](downloaded_files_path=r'temp_files/',
#                                           inference_server_params=inference_server_params,
#                                           server=logging_server)
#
#     @pytest.mark.dependency(name='test_init_inference', depends=['test_train'], scope='session')
#     def test_init_inference(self, inference_name, run_id, logging_server, logging_server_auth):
#         inference_server_params = server_auth(logging_server, run_id, logging_server_auth)
#         self.init_inference(inference_name, inference_server_params, logging_server)
#
#     @pytest.mark.parametrize('step', [0, 1, 'best'])
#     @pytest.mark.parametrize('fold_num', ['test'])
#     @pytest.mark.parametrize('trained_model', [True, False])
#     @pytest.mark.dependency(name='test_inference', depends=['test_init_inference'], scope='session')
#     def test_inference(self, inference_name, run_id, step, fold_num, trained_model, logging_server,
#                        logging_server_auth):
#         inference_server_params = server_auth(logging_server, run_id, logging_server_auth)
#         inference = self.init_inference(inference_name, inference_server_params, logging_server)
#
#         inference.get_params_model(step=step, fold_num=fold_num, trained_model=trained_model)
