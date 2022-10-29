from mlexp.inference._base_inference._base_inference import _BaseModelInference
from mlexp.inference._base_inference._mlflow_inference import _MlflowInference
from mlexp.inference._base_inference._neptune_inference import _NeptuneInference

SERVER_INFERENCES = {"mlflow": _MlflowInference, "neptune": _NeptuneInference}
