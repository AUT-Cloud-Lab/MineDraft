import os
from typing import Any, Dict, List, Tuple

from historical.common import Deployment, Node
from historical.config import Config
from historical.data import Cycle, PodPlacement


def get_nodes_of_a_deployment(
    pod_placement: PodPlacement, deployment: Deployment
) -> List[Node]:
    nodes = []
    for node, pods in pod_placement.node_pods.items():
        for pod in pods:
            if pod == deployment:
                nodes.append(node)
    return nodes


def calculate_pod_count_for_deployment(cycle: Cycle, target: Deployment) -> int:
    pod_count = 0

    for node, pods in cycle.pod_placement.node_pods.items():
        for pod in pods:
            if pod.name != target.name:
                continue
            pod_count += 1

    return pod_count


def calculate_edge_pod_count_for_deployment(cycle: Cycle, target: Deployment) -> int:
    pod_count = 0

    for node, pods in cycle.pod_placement.node_pods.items():
        if not node.is_on_edge:
            continue

        for pod in pods:
            if pod.name != target.name:
                continue
            pod_count += 1

    return pod_count


def calculate_placement_for_deployment(
    cycle: Cycle, target: Deployment
) -> Tuple[int, int]:
    cloud_count = 0
    edge_count = 0
    for node, pods in cycle.pod_placement.node_pods.items():
        for pod in pods:
            if pod.name != target.name:
                continue

            if node.is_on_edge:
                edge_count += 1

            if not node.is_on_edge:
                cloud_count += 1

    return cloud_count, edge_count


def calculate_edge_usage_sum(cycle: Cycle) -> Tuple[float, float]:
    cpu_sum = 0
    memory_sum = 0
    for node, pods in cycle.pod_placement.node_pods.items():
        if node.is_on_edge:
            for pod in pods:
                cpu_sum += pod.resources[0]
                memory_sum += pod.resources[1]

    return cpu_sum, memory_sum


def calculate_cluster_usage_sum(cycle: Cycle) -> Tuple[float, float]:
    cpu_sum = 0
    memory_sum = 0
    for node, pods in cycle.pod_placement.node_pods.items():
        for pod in pods:
            cpu_sum += pod.resources[0]
            memory_sum += pod.resources[1]
    return cpu_sum, memory_sum


def calculate_resource_usage_for_node(
    cycle: Cycle, desired_node: Node
) -> Tuple[float, float]:
    cpu_usage = 0
    memory_usage = 0
    for pod in cycle.pod_placement.node_pods[desired_node]:
        cpu_usage += pod.resources[0]
        memory_usage += pod.resources[1]

    return (cpu_usage / desired_node.resources[0]), (
        memory_usage / desired_node.resources[1]
    )


def ensure_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def merge_lists_by_average(*lists):
    max_length = max(len(lst) for lst in lists)

    sums = [0] * max_length
    counts = [0] * max_length

    for lst in lists:
        for i, val in enumerate(lst):
            sums[i] += val
            counts[i] += 1

    return [sums[i] / counts[i] if counts[i] != 0 else 0 for i in range(max_length)]


def merge_lists_by_sum(*lists):
    max_length = max(len(lst) for lst in lists)

    sums = [0] * max_length

    for lst in lists:
        for i, val in enumerate(lst):
            sums[i] += val

    return [sums[i] for i in range(max_length)]


def merge_for_each_deployment(
    results: List[Dict[Deployment, Any]]
) -> Dict[Deployment, Any]:
    res = {}
    for deployment in results[0].keys():
        res[deployment] = sum([result[deployment] for result in results]) / len(results)
    return res
