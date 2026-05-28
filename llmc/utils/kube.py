import json
import os

from kubernetes import client, config
from kubernetes.client import V1ConfigMap
from kubernetes.client.rest import ApiException

# ConfigMap name (bud-runtime polls this; do not rename without updating
# services/budcluster get_quantization_status.yaml).
CONFIGMAP_NAME = 'quantization-progress'
NAMESPACE = os.getenv('NAMESPACE')


class QuantizeConfigMap:
    # NOTE: these are class attributes on purpose. create_or_update_configmap
    # iterates instance.__dict__, so a field is only written to the ConfigMap
    # once it has been explicitly assigned on the instance. This keeps
    # 'quantization_eval' ABSENT until the eval step sets it -- bud-runtime
    # treats the presence of that key as the terminal "run_evaluation" signal.
    base_model_eval: bool = False
    base_model_eval_score: list = []
    quantization_progress: str = ''
    quantization_eval: bool = False
    quantization_eval_score: list = []


def create_or_update_configmap(data, name=CONFIGMAP_NAME, namespace=NAMESPACE):
    # Load Kubernetes config (inside the cluster)
    config.load_incluster_config()

    data_dict = {}
    for k, v in data.__dict__.items():
        if isinstance(v, str):
            data_dict[k] = v
        else:
            data_dict[k] = json.dumps(v)

    configmap = V1ConfigMap(
        api_version='v1',
        kind='ConfigMap',
        metadata={'name': name},
        data=data_dict,
    )

    v1 = client.CoreV1Api()
    try:
        try:
            existing = v1.read_namespaced_config_map(name, namespace)
            existing.data.update(configmap.data)
            v1.replace_namespaced_config_map(name, namespace, existing)
            print('Updated ConfigMap successfully.')
        except ApiException as e:
            if e.status == 404:
                v1.create_namespaced_config_map(namespace=namespace, body=configmap)
                print('Created ConfigMap successfully.')
            else:
                raise e
    except ApiException as e:
        print(f'Exception when handling ConfigMap: {e}')
        raise e
