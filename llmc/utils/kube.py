import random
import time
import os
from kubernetes import client, config

# ConfigMap name
CONFIGMAP_NAME = "quantization-progress"
NAMESPACE = os.getenv("NAMESPACE")


class QuantizeConfigMap:
    base_model_eval: bool = False
    base_model_eval_score: list = []
    quantization_progress: str = ""
    quantization_eval: bool = False
    quantization_eval_score: list = []


def create_or_update_configmap(data: QuantizeConfigMap):
    """Create or update a Kubernetes ConfigMap with intermediate results."""
    # Load Kubernetes config (inside the cluster)
    config.load_incluster_config()

    # Kubernetes API client
    v1 = client.CoreV1Api()
    try:
        # Check if ConfigMap exists
        existing = v1.read_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE)
        
        # Update existing ConfigMap
        existing.data.update(data)
        v1.replace_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE, existing)
        print("Updated ConfigMap successfully.")
    
    except client.exceptions.ApiException as e:
        if e.status == 404:
            # Create new ConfigMap if it doesn't exist
            cm = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(name=CONFIGMAP_NAME),
                data=data
            )
            v1.create_namespaced_config_map(namespace=NAMESPACE, body=cm)
            print("Created new ConfigMap.")
        else:
            print(f"Error updating ConfigMap: {e}")
