import math
import os.path
import statistics
from math import ceil
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from extractors.decorator import register_extractor
from historical.common import Deployment
from historical.config import Config
from historical.data import History, Cycle, Migration
from historical.utils import calculate_edge_usage_sum, calculate_cluster_usage_sum, calculate_resource_usage_for_node, \
    calculate_placement_for_deployment, calculate_pod_count_for_deployment, get_nodes_of_a_deployment, get_edge_placed_pods

CLOUD_RESPONSE_TIME = 300
EDGE_RESPONSE_TIME = 50

ECMUS_INDEX = 0
KUBE_SCHEDULE_INDEX = 1
ECMUS_NO_MIGRATION_INDEX = 2
RANDOM_INDEX = 3
CLOUD_FIRST_INDEX = 4
SMALLEST_EDGE_FIRST_INDEX = 5
BIGGEST_EDGE_FIRST_INDEX = 6
ECMUS_QOS_AWARE_INDEX = 7

INDEX_COUNT = 8

@register_extractor
def calc_migrations(config: Config, histories: List[History], save_path: str) -> None:
    output_file = open(save_path, 'w')
    assert len(histories) == 1
    history: History = histories[0]

    def is_cycle_valid(cycle: Cycle) -> bool:
        """
        Check if number of running pods are equal to the desired number of pods.
        """
        for deployment_name in config.deployments:
            number_of_pods = 0
            for (_, pods) in cycle.pod_placement.node_pods.items():
                number_of_pods += len(
                    list(filter(lambda pod: pod.name == deployment_name, pods))
                )

            if ceil(cycle.hpa.deployment_metrics[deployment]) != number_of_pods:
                return False

        return True

    for deployment in config.deployments.values():
        migrations: List[Migration] = []
        for (start, cycle) in enumerate(history.cycles):
            if not is_cycle_valid(cycle):
                continue
            for (_end, next_cycle) in enumerate(history.cycles[start + 1:]):
                end = start + _end + 1
                if not is_cycle_valid(next_cycle):
                    continue

                if ceil(cycle.hpa.deployment_metrics[deployment]) != ceil(
                        next_cycle.hpa.deployment_metrics[deployment]
                ):
                    continue

                # found two cycles with possible migrations!
                source_nodes = sorted(
                    get_nodes_of_a_deployment(cycle.pod_placement, deployment),
                    key = lambda node: node.name,
                )
                target_nodes = sorted(
                    get_nodes_of_a_deployment(next_cycle.pod_placement, deployment),
                    key = lambda node: node.name,
                )

                print(list(map(lambda node: node.name, source_nodes)), file = output_file)
                print(list(map(lambda node: node.name, target_nodes)), file = output_file)

                real_sources = []
                real_targets = []

                it_source = 0
                it_target = 0
                while it_source < len(source_nodes) and it_target < len(target_nodes):
                    if source_nodes[it_source] == target_nodes[it_target]:
                        it_source += 1
                        it_target += 1
                    elif source_nodes[it_source].name < target_nodes[it_target].name:
                        real_sources.append(source_nodes[it_source])
                        it_source += 1
                    else:
                        real_targets.append(target_nodes[it_target])
                        it_target += 1

                while it_source < len(source_nodes):
                    real_sources.append(source_nodes[it_source])
                    it_source += 1

                while it_target < len(target_nodes):
                    real_targets.append(target_nodes[it_target])
                    it_target += 1

                assert len(real_sources) == len(real_targets)
                print(f"here with {len(real_sources)}", file = output_file)
                print(f"{start}, {end}", file = output_file)
                for i in range(len(real_sources)):
                    migrations.append(
                        Migration(
                            deployment = deployment,
                            source = real_sources[i],
                            target = real_targets[i],
                            start = start,
                            end = end,
                        )
                    )

        print(f"Number of migrations for {deployment.name}: {len(migrations)}", file = output_file)
        for migration in migrations:
            print(migration, file = output_file)


@register_extractor
def check_equality(config: Config, histories: List[History], save_path: str) -> None:
    output_file = open(save_path, 'w')
    """
    Check if the histories are equal.
    """
    number_of_differences: Dict[Deployment, int] = {
        deployment: 0 for _, deployment in config.deployments.items()
    }
    print(len(histories[0].cycles), file = output_file)
    print(len(histories[1].cycles), file = output_file)

    for cycle_number in range(max(map(lambda history: len(history.cycles), histories))):
        cycles: List[Cycle] = []
        for history in histories:
            if cycle_number < len(history.cycles):
                cycles.append(history.cycles[cycle_number])
            else:
                for _, deployment in config.deployments.items():
                    number_of_differences[deployment] += 1

        main_cycle = cycles[0]
        for other_cycle in cycles[1:]:
            for _, deployment in config.deployments.items():
                main_cycle_metric = ceil(main_cycle.hpa.deployment_metrics[deployment])
                other_cycle_metric = ceil(
                    other_cycle.hpa.deployment_metrics[deployment]
                )
                if main_cycle_metric != other_cycle_metric:
                    number_of_differences[deployment] += 1

    for deployment, number_of_differences in number_of_differences.items():
        print(f"{deployment}'s number of differences are {number_of_differences}!", file = output_file)


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


@register_extractor
def pod_count_linechart(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    kube_pod_count_list = []
    kube_timestamps_list = []

    ecmus_pod_count_list = []
    ecmus_timestamps_list = []

    ecmus_no_migration_pod_count_list = []
    ecmus_no_migration_timestamps_list = []

    random_pod_count_list = []
    random_timestamps_list = []

    cloud_first_pod_count_list = []
    cloud_first_timestamps_list = []

    smallest_edge_first_pod_count_list = []
    smallest_edge_first_timestamps_list = []

    biggest_edge_first_pod_count_list = []
    biggest_edge_first_timestamps_list = []

    ecmus_qos_aware_pod_count_list = []
    ecmus_qos_aware_timestamps_list = []

    for deployment in config.deployments.values():
        box_count = len(histories) // INDEX_COUNT

        kube_pod_count = {it: [] for it in range(box_count)}
        kube_timestamps = {it: [] for it in range(box_count)}

        ecmus_pod_count = {it: [] for it in range(box_count)}
        ecmus_timestamps = {it: [] for it in range(box_count)}

        ecmus_no_migration_pod_count = {it: [] for it in range(box_count)}
        ecmus_no_migration_timestamps = {it: [] for it in range(box_count)}

        random_pod_count = {it: [] for it in range(box_count)}
        random_timestamps = {it: [] for it in range(box_count)}

        cloud_first_pod_count = {it: [] for it in range(box_count)}
        cloud_first_timestamps = {it: [] for it in range(box_count)}

        smallest_edge_first_pod_count = {it: [] for it in range(box_count)}
        smallest_edge_first_timestamps = {it: [] for it in range(box_count)}

        biggest_edge_first_pod_count = {it: [] for it in range(box_count)}
        biggest_edge_first_timestamps = {it: [] for it in range(box_count)}

        ecmus_qos_aware_pod_count = {it: [] for it in range(box_count)}
        ecmus_qos_aware_timestamps = {it: [] for it in range(box_count)}

        for id, history in enumerate(histories):
            # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
            index = id % INDEX_COUNT
            box_id = id // INDEX_COUNT
            for cycle in history.cycles:
                pod_count = calculate_pod_count_for_deployment(cycle, deployment)
                if index == ECMUS_INDEX:
                    ecmus_pod_count[box_id].append(pod_count)
                    ecmus_timestamps[box_id].append(cycle.timestamp)

                if index == KUBE_SCHEDULE_INDEX:
                    kube_pod_count[box_id].append(pod_count)
                    kube_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_NO_MIGRATION_INDEX:
                    ecmus_no_migration_pod_count[box_id].append(pod_count)
                    ecmus_no_migration_timestamps[box_id].append(cycle.timestamp)

                if index == RANDOM_INDEX:
                    random_pod_count[box_id].append(pod_count)
                    random_timestamps[box_id].append(cycle.timestamp)

                if index == CLOUD_FIRST_INDEX:
                    cloud_first_pod_count[box_id].append(pod_count)
                    cloud_first_timestamps[box_id].append(cycle.timestamp)

                if index == SMALLEST_EDGE_FIRST_INDEX:
                    smallest_edge_first_pod_count[box_id].append(pod_count)
                    smallest_edge_first_timestamps[box_id].append(cycle.timestamp)

                if index == BIGGEST_EDGE_FIRST_INDEX:
                    biggest_edge_first_pod_count[box_id].append(pod_count)
                    biggest_edge_first_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_QOS_AWARE_INDEX:
                    ecmus_qos_aware_pod_count[box_id].append(pod_count)
                    ecmus_qos_aware_timestamps[box_id].append(cycle.timestamp)

        ecmus_pod_count = merge_lists_by_average(*[ecmus_pod_count[it] for it in range(box_count)])
        ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])

        kube_pod_count = merge_lists_by_average(*[kube_pod_count[it] for it in range(box_count)])
        kube_timestamps = merge_lists_by_average(*[kube_timestamps[it] for it in range(box_count)])

        ecmus_no_migration_pod_count = merge_lists_by_average(*[ecmus_no_migration_pod_count[it] for it in range(box_count)])
        ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])

        cloud_first_pod_count = merge_lists_by_average(*[cloud_first_pod_count[it] for it in range(box_count)])
        cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])

        random_pod_count = merge_lists_by_average(*[random_pod_count[it] for it in range(box_count)])
        random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])

        smallest_edge_first_pod_count = merge_lists_by_average(*[smallest_edge_first_pod_count[it] for it in range(box_count)])
        smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])

        biggest_edge_first_pod_count = merge_lists_by_average(*[biggest_edge_first_pod_count[it] for it in range(box_count)])
        biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])

        ecmus_qos_aware_pod_count = merge_lists_by_average(*[ecmus_qos_aware_pod_count[it] for it in range(box_count)])
        ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps[it] for it in range(box_count)])

        ecmus_pod_count_list.append(ecmus_pod_count)
        ecmus_timestamps_list.append(ecmus_timestamps)

        kube_pod_count_list.append(kube_pod_count)
        kube_timestamps_list.append(kube_timestamps)

        ecmus_no_migration_pod_count_list.append(ecmus_no_migration_pod_count)
        ecmus_no_migration_timestamps_list.append(ecmus_no_migration_timestamps)

        cloud_first_pod_count_list.append(cloud_first_pod_count)
        cloud_first_timestamps_list.append(cloud_first_timestamps)

        random_pod_count_list.append(random_pod_count)
        random_timestamps_list.append(random_timestamps)

        smallest_edge_first_pod_count_list.append(smallest_edge_first_pod_count)
        smallest_edge_first_timestamps_list.append(smallest_edge_first_timestamps)

        biggest_edge_first_pod_count_list.append(biggest_edge_first_pod_count)
        biggest_edge_first_timestamps_list.append(biggest_edge_first_timestamps)

        ecmus_qos_aware_pod_count_list.append(ecmus_qos_aware_pod_count)
        ecmus_qos_aware_timestamps_list.append(ecmus_qos_aware_timestamps)

        fig, ax = plt.subplots()
        plt.grid()
        fig.set_size_inches(10.5, 10.5)
        ax.plot(kube_timestamps, kube_pod_count, label = "Kube", marker='o')
        ax.plot(ecmus_timestamps, ecmus_pod_count, label = "KubeDSM", marker='s')
        ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_pod_count, label = "KubeDSMNoMigration", marker='D')
        ax.plot(random_timestamps, random_pod_count, label = "Random", marker='^')
        ax.plot(cloud_first_timestamps, cloud_first_pod_count, label = "CloudFirst", marker='v')
        ax.plot(biggest_edge_first_timestamps, biggest_edge_first_pod_count, label = "BiggestEdgeFirst", marker='x')
        ax.plot(smallest_edge_first_timestamps, smallest_edge_first_pod_count, label = "SmallestEdgeFirst", marker='+')
        ax.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_pod_count, label = "KubeDSMQOSAware", marker="*")
        ax.set_ylim(0, 20)
        ax.set_yticks(range(0, 20, 1))
        plt.xlabel("time(s)")
        plt.ylabel("pod count")
        plt.title(f"pod count - workload: {deployment.name}")
        plt.legend(loc="upper right")
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")

    deployment_count = len(config.deployments)

    ecmus_pod_count = merge_lists_by_sum(*[ecmus_pod_count_list[it] for it in range(deployment_count)])
    ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps_list[it] for it in range(deployment_count)])

    kube_pod_count = merge_lists_by_sum(*[kube_pod_count_list[it] for it in range(deployment_count)])
    kube_timestamps = merge_lists_by_average(*[kube_timestamps_list[it] for it in range(deployment_count)])

    ecmus_no_migration_pod_count = merge_lists_by_sum(*[ecmus_no_migration_pod_count_list[it] for it in range(deployment_count)])
    ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps_list[it] for it in range(deployment_count)])

    random_pod_count = merge_lists_by_sum(*[random_pod_count_list[it] for it in range(deployment_count)])
    random_timestamps = merge_lists_by_average(*[random_timestamps_list[it] for it in range(deployment_count)])

    cloud_first_pod_count = merge_lists_by_sum(*[cloud_first_pod_count_list[it] for it in range(deployment_count)])
    cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps_list[it] for it in range(deployment_count)])

    smallest_edge_first_pod_count = merge_lists_by_sum(*[smallest_edge_first_pod_count_list[it] for it in range(deployment_count)])
    smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps_list[it] for it in range(deployment_count)])

    biggest_edge_first_pod_count = merge_lists_by_sum(*[biggest_edge_first_pod_count_list[it] for it in range(deployment_count)])
    biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps_list[it] for it in range(deployment_count)])

    ecmus_qos_aware_pod_count = merge_lists_by_sum(*[ecmus_qos_aware_pod_count_list[it] for it in range(deployment_count)])
    ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps_list[it] for it in range(deployment_count)])

    fig, ax = plt.subplots()
    plt.grid()
    fig.set_size_inches(10.5, 10.5)
    ax.plot(kube_timestamps, kube_pod_count, label = "Kube", marker='o')
    ax.plot(ecmus_timestamps, ecmus_pod_count, label = "KubeDSM", marker='s')
    ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_pod_count, label = "KubeDSMNoMigration", marker='D')
    ax.plot(random_timestamps, random_pod_count, label = "Random", marker='^')
    ax.plot(cloud_first_timestamps, cloud_first_pod_count, label = "CloudFirst", marker='v')
    ax.plot(biggest_edge_first_timestamps, biggest_edge_first_pod_count, label = "BiggestEdgeFirst", marker='x')
    ax.plot(smallest_edge_first_timestamps, smallest_edge_first_pod_count, label = "SmallestEdgeFirst", marker='+')
    ax.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_pod_count, label = "KubeDSMQOSAware", marker="*")
    ax.set_ylim(0, 20)
    ax.set_yticks(range(0, 20, 1))
    plt.xlabel("time(s)")
    plt.ylabel("pod count")
    plt.title(f"pod count - workload total")
    plt.legend(loc="upper right")
    ensure_directory(save_path)
    plt.savefig(f"{save_path}/all.png")


@register_extractor
def average_latency_linechart(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    for deployment in config.deployments.values():
        box_count = len(histories) // INDEX_COUNT

        kube_latencies = {it: [] for it in range(box_count)}
        kube_timestamps = {it: [] for it in range(box_count)}

        ecmus_latencies = {it: [] for it in range(box_count)}
        ecmus_timestamps = {it: [] for it in range(box_count)}

        ecmus_no_migration_latencies = {it: [] for it in range(box_count)}
        ecmus_no_migration_timestamps = {it: [] for it in range(box_count)}

        random_latencies = {it: [] for it in range(box_count)}
        random_timestamps = {it: [] for it in range(box_count)}

        cloud_first_latencies = {it: [] for it in range(box_count)}
        cloud_first_timestamps = {it: [] for it in range(box_count)}

        smallest_edge_first_latencies = {it: [] for it in range(box_count)}
        smallest_edge_first_timestamps = {it: [] for it in range(box_count)}

        biggest_edge_first_latencies = {it: [] for it in range(box_count)}
        biggest_edge_first_timestamps = {it: [] for it in range(box_count)}

        ecmus_qos_aware_latencies = {it: [] for it in range(box_count)}
        ecmus_qos_aware_timestamps = {it: [] for it in range(box_count)}

        for id, history in enumerate(histories):
            # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
            index = id % INDEX_COUNT
            box_id = id // INDEX_COUNT
            for cycle in history.cycles:
                cloud_pods_count, edge_pods_count = calculate_placement_for_deployment(cycle, deployment)
                all_pods_count = cloud_pods_count + edge_pods_count
                # portion = calculate_request_portion_for_deployment(config, cycle, deployment)
                latency = (
                                  cloud_pods_count * CLOUD_RESPONSE_TIME + edge_pods_count * EDGE_RESPONSE_TIME) / all_pods_count

                if index == ECMUS_INDEX:
                    ecmus_latencies[box_id].append(latency)
                    ecmus_timestamps[box_id].append(cycle.timestamp)

                if index == KUBE_SCHEDULE_INDEX:
                    kube_latencies[box_id].append(latency)
                    kube_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_NO_MIGRATION_INDEX:
                    ecmus_no_migration_latencies[box_id].append(latency)
                    ecmus_no_migration_timestamps[box_id].append(cycle.timestamp)

                if index == RANDOM_INDEX:
                    random_latencies[box_id].append(latency)
                    random_timestamps[box_id].append(cycle.timestamp)

                if index == CLOUD_FIRST_INDEX:
                    cloud_first_latencies[box_id].append(latency)
                    cloud_first_timestamps[box_id].append(cycle.timestamp)

                if index == SMALLEST_EDGE_FIRST_INDEX:
                    smallest_edge_first_latencies[box_id].append(latency)
                    smallest_edge_first_timestamps[box_id].append(cycle.timestamp)

                if index == BIGGEST_EDGE_FIRST_INDEX:
                    biggest_edge_first_latencies[box_id].append(latency)
                    biggest_edge_first_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_QOS_AWARE_INDEX:
                    ecmus_qos_aware_latencies[box_id].append(latency)
                    ecmus_qos_aware_timestamps[box_id].append(cycle.timestamp)

        ecmus_latencies = merge_lists_by_average(*[ecmus_latencies[it] for it in range(box_count)])
        ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])

        kube_latencies = merge_lists_by_average(*[kube_latencies[it] for it in range(box_count)])
        kube_timestamps = merge_lists_by_average(*[kube_timestamps[it] for it in range(box_count)])

        ecmus_no_migration_latencies = merge_lists_by_average(*[ecmus_no_migration_latencies[it] for it in range(box_count)])
        ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])

        cloud_first_latencies = merge_lists_by_average(*[cloud_first_latencies[it] for it in range(box_count)])
        cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])

        random_latencies = merge_lists_by_average(*[random_latencies[it] for it in range(box_count)])
        random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])

        smallest_edge_first_latencies = merge_lists_by_average(*[smallest_edge_first_latencies[it] for it in range(box_count)])
        smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])

        biggest_edge_first_latencies = merge_lists_by_average(*[biggest_edge_first_latencies[it] for it in range(box_count)])
        biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])

        ecmus_qos_aware_latencies = merge_lists_by_average(*[ecmus_qos_aware_latencies[it] for it in range(box_count)])
        ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps[it] for it in range(box_count)])

        fig, ax = plt.subplots()
        plt.grid()
        fig.set_size_inches(10.5, 10.5)
        ax.plot(kube_timestamps, kube_latencies, label = "Kube", marker='o')
        ax.plot(ecmus_timestamps, ecmus_latencies, label = "KubeDSM", marker='s')
        ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_latencies, label = "KubeDSMNoMigration", marker='D')
        ax.plot(random_timestamps, random_latencies, label = "Random", marker='^')
        ax.plot(cloud_first_timestamps, cloud_first_latencies, label = "CloudFirst", marker='v')
        ax.plot(biggest_edge_first_timestamps, biggest_edge_first_latencies, label = "BiggestEdgeFirst", marker='x')
        ax.plot(smallest_edge_first_timestamps, smallest_edge_first_latencies, label = "SmallestEdgeFirst", marker='+')
        ax.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_latencies, label = "KubeDSMQOSAware", marker="*")
        ax.set_ylim(25, 325)
        ax.set_yticks(range(25, 325, 25))
        plt.xlabel("time(s)")
        plt.ylabel("average latency(ms)")
        plt.title(f"average latency - workload: {deployment.name}")
        plt.legend(loc="upper right")
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")


def ensure_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


@register_extractor
def average_latency_boxplot(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    data = {
        "Kube"               : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "KubeDSM"                        : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "KubeDSMNoMigration"           : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "Random"             : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "CloudFirst"        : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "BiggestEdgeFirst" : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "SmallestEdgeFirst": {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "KubeDSMQOSAware": {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
    }

    for deployment in config.deployments.values():
        for id, history in enumerate(histories):
            # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
            index = id % INDEX_COUNT
            for cycle in history.cycles:
                cloud_pods_count, edge_pods_count = calculate_placement_for_deployment(cycle, deployment)
                all_pods_count = cloud_pods_count + edge_pods_count
                latency = (
                                  cloud_pods_count * CLOUD_RESPONSE_TIME + edge_pods_count * EDGE_RESPONSE_TIME) / all_pods_count

                if index == ECMUS_INDEX:
                    data["KubeDSM"][deployment.name].append(latency)

                if index == KUBE_SCHEDULE_INDEX:
                    data["Kube"][deployment.name].append(latency)

                if index == ECMUS_NO_MIGRATION_INDEX:
                    data["KubeDSMNoMigration"][deployment.name].append(latency)

                if index == RANDOM_INDEX:
                    data["Random"][deployment.name].append(latency)

                if index == CLOUD_FIRST_INDEX:
                    data["CloudFirst"][deployment.name].append(latency)

                if index == SMALLEST_EDGE_FIRST_INDEX:
                    data["SmallestEdgeFirst"][deployment.name].append(latency)

                if index == BIGGEST_EDGE_FIRST_INDEX:
                    data["BiggestEdgeFirst"][deployment.name].append(latency)

                if index == ECMUS_QOS_AWARE_INDEX:
                    data["KubeDSMQOSAware"][deployment.name].append(latency)

    a_means = []
    b_means = []
    c_means = []
    d_means = []

    a_errors = []
    b_errors = []
    c_errors = []
    d_errors = []
    for scheduler, latencies in data.items():
        a_means.append(round(statistics.mean(latencies["a"]), 1))
        b_means.append(round(statistics.mean(latencies["b"]), 1))
        c_means.append(round(statistics.mean(latencies["c"]), 1))
        d_means.append(round(statistics.mean(latencies["d"]), 1))

        a_errors.append(round(statistics.stdev(latencies["a"]), 1))
        b_errors.append(round(statistics.stdev(latencies["b"]), 1))
        c_errors.append(round(statistics.stdev(latencies["c"]), 1))
        d_errors.append(round(statistics.stdev(latencies["d"]), 1))

    x = np.arange(len(data.keys()))
    width = 0.2

    fig, ax = plt.subplots(layout = "constrained")
    fig.set_size_inches(10.5, 10.5)

    rects1 = ax.bar(x - 3 * width / 2, a_means, width, label = 'a', yerr = a_errors, capsize = 10)
    ax.bar_label(rects1, padding = 10)
    rects2 = ax.bar(x - width / 2, b_means, width, label = 'b', yerr = b_errors, capsize = 10)
    ax.bar_label(rects2, padding = 10)
    rects3 = ax.bar(x + width / 2, c_means, width, label = 'c', yerr = c_errors, capsize = 10)
    ax.bar_label(rects3, padding = 10)
    rects4 = ax.bar(x + 3 * width / 2, d_means, width, label = 'd', yerr = d_errors, capsize = 10)
    ax.bar_label(rects4, padding = 10)

    plt.grid()
    ax.set_title('Grouped bar chart for schedulers')
    ax.set_xticks(x)
    ax.set_xticklabels(data.keys(), rotation = 0)
    ax.set_ylim(0, 350)
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")
    # plt.show()


@register_extractor
def edge_utilization_linechart(config: Config, _: str, histories: List[History], save_path: str) -> None:
    box_count = len(histories) // INDEX_COUNT

    ecmus_utilization = {it: [] for it in range(box_count)}
    ecmus_timestamps = {it: [] for it in range(box_count)}

    kube_schedule_utilization = {it: [] for it in range(box_count)}
    kube_schedule_timestamps = {it: [] for it in range(box_count)}

    ecmus_no_migration_utilization = {it: [] for it in range(box_count)}
    ecmus_no_migration_timestamps = {it: [] for it in range(box_count)}

    random_utilization = {it: [] for it in range(box_count)}
    random_timestamps = {it: [] for it in range(box_count)}

    cloud_first_utilization = {it: [] for it in range(box_count)}
    cloud_first_timestamps = {it: [] for it in range(box_count)}

    smallest_edge_first_utilization = {it: [] for it in range(box_count)}
    smallest_edge_first_timestamps = {it: [] for it in range(box_count)}

    biggest_edge_first_utilization = {it: [] for it in range(box_count)}
    biggest_edge_first_timestamps = {it: [] for it in range(box_count)}

    ecmus_qos_aware_utilization = {it: [] for it in range(box_count)}
    ecmus_qos_aware_timestamps = {it: [] for it in range(box_count)}

    for id, history in enumerate(histories):
        # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
        index = id % INDEX_COUNT
        box_id = id // INDEX_COUNT
        for cycle in history.cycles:
            edge_pods = get_edge_placed_pods(cycle)

            edge_usage_sum_cpu, edge_usage_sum_memory = calculate_edge_usage_sum(config, edge_pods)
            cluster_usage_sum_cpu, cluster_usage_sum_memory = calculate_cluster_usage_sum(cycle)

            utilization = math.sqrt(
                (edge_usage_sum_cpu / cluster_usage_sum_cpu) * (edge_usage_sum_memory / cluster_usage_sum_memory)
            )

            if index == ECMUS_INDEX:
                ecmus_timestamps[box_id].append(cycle.timestamp)
                ecmus_utilization[box_id].append(utilization)

            if index == KUBE_SCHEDULE_INDEX:
                kube_schedule_timestamps[box_id].append(cycle.timestamp)
                kube_schedule_utilization[box_id].append(utilization)

            if index == ECMUS_NO_MIGRATION_INDEX:
                ecmus_no_migration_timestamps[box_id].append(cycle.timestamp)
                ecmus_no_migration_utilization[box_id].append(utilization)

            if index == RANDOM_INDEX:
                random_timestamps[box_id].append(cycle.timestamp)
                random_utilization[box_id].append(utilization)

            if index == CLOUD_FIRST_INDEX:
                cloud_first_timestamps[box_id].append(cycle.timestamp)
                cloud_first_utilization[box_id].append(utilization)

            if index == SMALLEST_EDGE_FIRST_INDEX:
                smallest_edge_first_timestamps[box_id].append(cycle.timestamp)
                smallest_edge_first_utilization[box_id].append(utilization)

            if index == BIGGEST_EDGE_FIRST_INDEX:
                biggest_edge_first_timestamps[box_id].append(cycle.timestamp)
                biggest_edge_first_utilization[box_id].append(utilization)

            if index == ECMUS_QOS_AWARE_INDEX:
                ecmus_qos_aware_timestamps[box_id].append(cycle.timestamp)
                ecmus_qos_aware_utilization[box_id].append(utilization)

    ecmus_utilization = merge_lists_by_average(*[ecmus_utilization[it] for it in range(box_count)])
    ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])

    kube_schedule_utilization = merge_lists_by_average(*[kube_schedule_utilization[it] for it in range(box_count)])
    kube_schedule_timestamps = merge_lists_by_average(*[kube_schedule_timestamps[it] for it in range(box_count)])

    ecmus_no_migration_utilization = merge_lists_by_average(*[ecmus_no_migration_utilization[it] for it in range(box_count)])
    ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])

    cloud_first_utilization = merge_lists_by_average(*[cloud_first_utilization[it] for it in range(box_count)])
    cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])

    random_utilization = merge_lists_by_average(*[random_utilization[it] for it in range(box_count)])
    random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])

    smallest_edge_first_utilization = merge_lists_by_average(*[smallest_edge_first_utilization[it] for it in range(box_count)])
    smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])

    biggest_edge_first_utilization = merge_lists_by_average(*[biggest_edge_first_utilization[it] for it in range(box_count)])
    biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])

    ecmus_qos_aware_utilization = merge_lists_by_average(*[ecmus_qos_aware_utilization[it] for it in range(box_count)])
    ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps[it] for it in range(box_count)])

    fig, ax = plt.subplots()
    plt.plot(ecmus_timestamps, ecmus_utilization, label = "KubeDSM", marker='o')
    plt.plot(kube_schedule_timestamps, kube_schedule_utilization, label = "Kube", marker='s')
    plt.plot(ecmus_no_migration_timestamps, ecmus_no_migration_utilization, label = "KubeDSMNoMigration", marker='D')
    plt.plot(random_timestamps, random_utilization, label = "Random", marker='^')
    plt.plot(cloud_first_timestamps, cloud_first_utilization, label = "CloudFirst", marker='v')
    plt.plot(smallest_edge_first_timestamps, smallest_edge_first_utilization, label = "SmallestEdgeFirst", marker='x')
    plt.plot(biggest_edge_first_timestamps, biggest_edge_first_utilization, label = "BiggestEdgeFirst", marker='+')
    plt.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_utilization, label = "KubeDSMQOSAware", marker="*")

    fig.set_size_inches(10.5, 10.5)
    plt.grid()
    plt.xlabel("time (s)")
    plt.ylabel("edge utilization")
    plt.ylim(0, 1.10)
    plt.yticks(list(map(lambda x: x / 100.0, range(0, 110, 5))))
    plt.title("edge utilization - per algorithm")
    plt.legend(loc="upper right")
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")


@register_extractor
def placement_ratio_linechart(config: Config, _: str, histories: List[History], save_path: str) -> None:
    box_count = len(histories) // INDEX_COUNT

    ecmus_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_timestamps = {it: [] for it in range(box_count)}

    kube_schedule_edge_placement_ratio = {it: [] for it in range(box_count)}
    kube_schedule_cloud_placement_ratio = {it: [] for it in range(box_count)}
    kube_schedule_timestamps = {it: [] for it in range(box_count)}

    ecmus_no_migration_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_no_migration_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_no_migration_timestamps = {it: [] for it in range(box_count)}

    random_edge_placement_ratio = {it: [] for it in range(box_count)}
    random_cloud_placement_ratio = {it: [] for it in range(box_count)}
    random_timestamps = {it: [] for it in range(box_count)}

    cloud_first_edge_placement_ratio = {it: [] for it in range(box_count)}
    cloud_first_cloud_placement_ratio = {it: [] for it in range(box_count)}
    cloud_first_timestamps = {it: [] for it in range(box_count)}

    smallest_edge_first_cloud_placement_ratio = {it: [] for it in range(box_count)}
    smallest_edge_first_edge_placement_ratio = {it: [] for it in range(box_count)}
    smallest_edge_first_timestamps = {it: [] for it in range(box_count)}

    biggest_edge_first_edge_placement_ratio = {it: [] for it in range(box_count)}
    biggest_edge_first_cloud_placement_ratio = {it: [] for it in range(box_count)}
    biggest_edge_first_timestamps = {it: [] for it in range(box_count)}

    ecmus_qos_aware_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_qos_aware_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_qos_aware_timestamps = {it: [] for it in range(box_count)}

    edge_nodes_count = len([node for node in config.nodes.values() if node.is_on_edge])
    cloud_nodes_count = len([node for node in config.nodes.values() if not node.is_on_edge])
    for id, history in enumerate(histories):
        # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
        index = id % INDEX_COUNT
        box_id = id // INDEX_COUNT
        for cycle in history.cycles:
            edge_usage = 0
            cloud_usage = 0
            for node in cycle.pod_placement.node_pods.keys():
                cpu_usage, memory_usage = calculate_resource_usage_for_node(cycle, node)

                if node.is_on_edge:
                    edge_usage += math.sqrt(cpu_usage * memory_usage)

                if not node.is_on_edge:
                    cloud_usage += math.sqrt(cpu_usage * memory_usage)

            fragmentation_edge = 1 - (edge_usage / edge_nodes_count)
            fragmentation_cloud = 1 - (cloud_usage / cloud_nodes_count)

            if index == ECMUS_INDEX:
                ecmus_timestamps[box_id].append(cycle.timestamp)
                ecmus_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == KUBE_SCHEDULE_INDEX:
                kube_schedule_timestamps[box_id].append(cycle.timestamp)
                kube_schedule_edge_placement_ratio[box_id].append(fragmentation_edge)
                kube_schedule_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == ECMUS_NO_MIGRATION_INDEX:
                ecmus_no_migration_timestamps[box_id].append(cycle.timestamp)
                ecmus_no_migration_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_no_migration_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == RANDOM_INDEX:
                random_timestamps[box_id].append(cycle.timestamp)
                random_edge_placement_ratio[box_id].append(fragmentation_edge)
                random_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == CLOUD_FIRST_INDEX:
                cloud_first_timestamps[box_id].append(cycle.timestamp)
                cloud_first_edge_placement_ratio[box_id].append(fragmentation_edge)
                cloud_first_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == SMALLEST_EDGE_FIRST_INDEX:
                smallest_edge_first_timestamps[box_id].append(cycle.timestamp)
                smallest_edge_first_edge_placement_ratio[box_id].append(fragmentation_edge)
                smallest_edge_first_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == BIGGEST_EDGE_FIRST_INDEX:
                biggest_edge_first_timestamps[box_id].append(cycle.timestamp)
                biggest_edge_first_edge_placement_ratio[box_id].append(fragmentation_edge)
                biggest_edge_first_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == ECMUS_QOS_AWARE_INDEX:
                ecmus_qos_aware_timestamps[box_id].append(cycle.timestamp)
                ecmus_qos_aware_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_qos_aware_cloud_placement_ratio[box_id].append(fragmentation_cloud)

    ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])
    ecmus_edge_placement_ratio = merge_lists_by_average(*[ecmus_edge_placement_ratio[it] for it in range(box_count)])
    ecmus_cloud_placement_ratio = merge_lists_by_average(*[ecmus_cloud_placement_ratio[it] for it in range(box_count)])

    kube_schedule_timestamps = merge_lists_by_average(*[kube_schedule_timestamps[it] for it in range(box_count)])
    kube_schedule_edge_placement_ratio = merge_lists_by_average(
        *[kube_schedule_edge_placement_ratio[it] for it in range(box_count)])
    kube_schedule_cloud_placement_ratio = merge_lists_by_average(
        *[kube_schedule_cloud_placement_ratio[it] for it in range(box_count)])

    ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])
    ecmus_no_migration_edge_placement_ratio = merge_lists_by_average(
        *[ecmus_no_migration_edge_placement_ratio[it] for it in range(box_count)])
    ecmus_no_migration_cloud_placement_ratio = merge_lists_by_average(
        *[ecmus_no_migration_cloud_placement_ratio[it] for it in range(box_count)])

    random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])
    random_edge_placement_ratio = merge_lists_by_average(*[random_edge_placement_ratio[it] for it in range(box_count)])
    random_cloud_placement_ratio = merge_lists_by_average(*[random_cloud_placement_ratio[it] for it in range(box_count)])

    cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])
    cloud_first_edge_placement_ratio = merge_lists_by_average(*[cloud_first_edge_placement_ratio[it] for it in range(box_count)])
    cloud_first_cloud_placement_ratio = merge_lists_by_average(*[cloud_first_cloud_placement_ratio[it] for it in range(box_count)])

    smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])
    smallest_edge_first_edge_placement_ratio = merge_lists_by_average(
        *[smallest_edge_first_edge_placement_ratio[it] for it in range(box_count)])
    smallest_edge_first_cloud_placement_ratio = merge_lists_by_average(
        *[smallest_edge_first_cloud_placement_ratio[it] for it in range(box_count)])

    biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])
    biggest_edge_first_edge_placement_ratio = merge_lists_by_average(
        *[biggest_edge_first_edge_placement_ratio[it] for it in range(box_count)])
    biggest_edge_first_cloud_placement_ratio = merge_lists_by_average(
        *[biggest_edge_first_cloud_placement_ratio[it] for it in range(box_count)])

    ecmus_qos_aware_timestamps = merge_lists_by_average(
        *[ecmus_qos_aware_timestamps[it] for it in range(box_count)])
    ecmus_qos_aware_edge_placement_ratio = merge_lists_by_average(
        *[ecmus_qos_aware_edge_placement_ratio[it] for it in range(box_count)])
    ecmus_qos_aware_cloud_placement_ratio = merge_lists_by_average(
        *[ecmus_qos_aware_cloud_placement_ratio[it] for it in range(box_count)])

    fig, (ax_cloud, ax_edge) = plt.subplots(2, 1)
    fig.set_size_inches(10.5, 10.5)

    ax_edge.grid()
    ax_edge.plot(ecmus_timestamps, ecmus_edge_placement_ratio, label = "KubeDSM", marker='o')
    ax_edge.plot(kube_schedule_timestamps, kube_schedule_edge_placement_ratio, label = "Kube", marker='s')
    ax_edge.plot(ecmus_no_migration_timestamps, ecmus_no_migration_edge_placement_ratio, label = "KubeDSMNoMigration", marker='D')
    ax_edge.plot(random_timestamps, random_edge_placement_ratio, label = "Random", marker='^')
    ax_edge.plot(cloud_first_timestamps, cloud_first_edge_placement_ratio, label = "CloudFirst", marker='v')
    ax_edge.plot(smallest_edge_first_timestamps, smallest_edge_first_edge_placement_ratio, label = "SmallestEdgeFirst", marker='x')
    ax_edge.plot(biggest_edge_first_timestamps, biggest_edge_first_edge_placement_ratio, label = "BiggestEdgeFirst", marker='+')
    plt.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_edge_placement_ratio, label = "KubeDSMQOSAware", marker="*")

    ax_edge.set_xlabel("time (s)")
    ax_edge.set_ylabel("placement ratio")
    ax_edge.set_ylim(0, 1.10)
    ax_edge.set_yticks(list(map(lambda x: x / 100.0, range(0, 110, 5))))
    ax_edge.set_title("edge pod placement ratio")
    ax_edge.legend(loc="upper right")

    ax_cloud.grid()
    ax_cloud.plot(ecmus_timestamps, ecmus_cloud_placement_ratio, label = "KubeDSM", marker='o')
    ax_cloud.plot(kube_schedule_timestamps, kube_schedule_cloud_placement_ratio, label = "Kube", marker='s')
    ax_cloud.plot(ecmus_no_migration_timestamps, ecmus_no_migration_cloud_placement_ratio, label = "KubeDSMNoMigration", marker='D')
    ax_cloud.plot(random_timestamps, random_cloud_placement_ratio, label = "Random", marker='^')
    ax_cloud.plot(cloud_first_timestamps, cloud_first_cloud_placement_ratio, label = "CloudFirst", marker='v')
    ax_cloud.plot(smallest_edge_first_timestamps, smallest_edge_first_cloud_placement_ratio, label = "SmallestEdgeFirst", marker='x')
    ax_cloud.plot(biggest_edge_first_timestamps, biggest_edge_first_cloud_placement_ratio, label = "BiggestEdgeFirst", marker='+')
    ax_cloud.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_cloud_placement_ratio, label = "KubeDSMQOSAware", marker="*")

    ax_cloud.set_xlabel("time (s)")
    ax_cloud.set_ylabel("placement ratio")
    ax_cloud.set_ylim(0, 1.10)
    ax_cloud.set_yticks(list(map(lambda x: x / 100.0, range(0, 110, 5))))
    ax_cloud.set_title("cloud pod placement ratio")
    ax_cloud.legend(loc="upper right")

    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")
