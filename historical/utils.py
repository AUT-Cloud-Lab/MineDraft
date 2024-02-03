from typing import Dict, List, Tuple
from historical.data import PodPlacement
from historical.common import Deployment, Node


def get_deployment_from_name_in_hpa(deployment_name_in_hpa: str) -> str:
    return deployment_name_in_hpa[:-4]  # remove "-hpa" suffix


def get_deployment_name_from_pod_name(pod_name: str) -> str:
    return pod_name[
        : pod_name.find("-deployment")
    ]  # the pattern is <deployment_name>-deployment-<random_string>


def get_nodes_of_a_deployment(
    pod_placement: PodPlacement, deployment: Deployment
) -> List[Node]:
    nodes = []
    for node, pods in pod_placement.node_pods.items():
        for pod in pods:
            if pod == deployment:
                nodes.append(node)
    return nodes
