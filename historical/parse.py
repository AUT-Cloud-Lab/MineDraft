from pytimeparse import parse as parse_time

from historical.config import Config
from historical.data import Cycle, History, PodPlacement, HpaState
from historical.common import Deployment, Node
from historical.utils import (
    get_deployment_name_from_pod_name,
    get_deployment_from_name_in_hpa,
)


def parse_config(json_data: dict) -> Config:
    """Parse a config from the JSON data."""
    deployments = {}
    nodes = {}

    for deployment_data in json_data["deployments"]:
        deployment = Deployment(
            deployment_data["name"], list(map(float, deployment_data["resources"]))
        )
        deployments[deployment.name] = deployment

    for node_data in json_data["nodes"]:
        node = Node(
            node_data["name"],
            node_data["is_on_edge"],
            list(map(float, node_data["resources"])),
        )
        nodes[node.name] = node

    return Config(deployments, nodes)


def parse_pod_placement(config: Config, json_data: dict) -> PodPlacement:
    """Parse a pod placement from the JSON data."""
    node_pods = {}

    for node_name, pod_names in json_data.items():
        if node_name not in config.nodes:
            continue

        node = config.nodes[node_name]
        deployments = []
        for pod_name in pod_names:
            deployment_name = get_deployment_name_from_pod_name(pod_name)
            if deployment_name not in config.deployments:
                continue
            deployments.append(
                config.deployments[get_deployment_name_from_pod_name(pod_name)]
            )

        node_pods[node] = deployments

    return PodPlacement(node_pods)


def parse_hpa_state(config: Config, json_data: dict) -> HpaState:
    """Parse an HPA state from the JSON data."""
    deployment_metrics = {}

    for deployment_data in json_data:
        deployment_name_in_hpa = deployment_data["name"]
        metric = deployment_data["metric_value"]

        deployment_name = get_deployment_from_name_in_hpa(deployment_name_in_hpa)
        if deployment_name not in config.deployments:
            continue
        # print(f"deployment {deployment_name}: {metric}")

        deployment = config.deployments[deployment_name]
        deployment_metrics[deployment] = metric

    return HpaState(deployment_metrics)


def parse_cycle(config: Config, json_data: dict) -> Cycle:
    """Parse a cycle from the JSON data."""
    timestamp = parse_time(json_data["timestamp"])
    hpa = parse_hpa_state(config, json_data["hpa"])
    pod_placement = parse_pod_placement(config, json_data["pod_placement"])

    return Cycle(timestamp, hpa, pod_placement)


def parse_history(config: Config, json_data: dict) -> History:
    """Parse a history from the JSON data."""
    cycles = [parse_cycle(config, cycle) for cycle in json_data]

    return History(cycles)
