from typing import List

from historical.common import Deployment, Node
from historical.config import Config
from historical.data import PodPlacement, Cycle


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


def calculate_cloud_pod_count(cycle: Cycle) -> (int, int, int, int):
    a_count = 0
    b_count = 0
    c_count = 0
    d_count = 0

    for node, placement in cycle.pod_placement.node_pods.items():
        if node.is_on_edge:
            continue

        for deployment in placement:
            if deployment.name == "a":
                a_count += 1

            if deployment.name == "b":
                b_count += 1

            if deployment.name == "c":
                c_count += 1

            if deployment.name == "d":
                d_count += 1

    return a_count, b_count, c_count, d_count


def calculate_edge_pod_count(cycle: Cycle) -> (int, int, int, int):
    a_count = 0
    b_count = 0
    c_count = 0
    d_count = 0

    for node, placement in cycle.pod_placement.node_pods.items():
        if not node.is_on_edge:
            continue

        for deployment in placement:
            if deployment.name == "a":
                a_count += 1

            if deployment.name == "b":
                b_count += 1

            if deployment.name == "c":
                c_count += 1

            if deployment.name == "d":
                d_count += 1

    return a_count, b_count, c_count, d_count


def calculate_deployments_request_portion(cycle: Cycle) -> (int, int, int, int):
    a_request_count = 0
    b_request_count = 0
    c_request_count = 0
    d_request_count = 0
    all_requests_count = 0

    for deployment, hpa_value in cycle.hpa.deployment_metrics.items():
        if deployment.name == "a":
            a_request_count = hpa_value * 15 / 5

        if deployment.name == "b":
            b_request_count = hpa_value * 15 / 5

        if deployment.name == 'c':
            c_request_count = hpa_value * 15 / 5

        if deployment.name == 'd':
            d_request_count = hpa_value * 15 / 5

    all_requests_count = a_request_count + b_request_count + c_request_count + d_request_count

    return a_request_count / all_requests_count, \
           b_request_count / all_requests_count, \
           c_request_count / all_requests_count, \
           d_request_count / all_requests_count


def get_edge_placed_pods(cycle: Cycle) -> list[Deployment]:
    edge_placed_pods = []
    for node, pod in cycle.pod_placement.node_pods.items():
        if node.is_on_edge:
            edge_placed_pods.extend(pod)

    return edge_placed_pods


def calculate_edge_usage_sum(config: Config, edge_pods: List[Deployment]) -> (float, float):
    cpu_sum = 0
    memory_sum = 0
    for pod in edge_pods:
        cpu_sum += config.deployments[pod.name].resources[0]
        memory_sum += config.deployments[pod.name].resources[1]

    return cpu_sum, memory_sum


def calculate_cluster_usage_sum(cycle: Cycle, config: Config) -> (float, float):
    cpu_sum = 0
    memory_sum = 0
    for node, pods in cycle.pod_placement.node_pods.items():
        for pod in pods:
            cpu_sum += pod.resources[0]
            memory_sum += pod.resources[1]
    return cpu_sum, memory_sum
