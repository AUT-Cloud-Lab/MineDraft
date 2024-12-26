import math
import json
import os.path
import statistics
import pandas as pd
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
    calculate_placement_for_deployment, calculate_pod_count_for_deployment, calculate_edge_pod_count_for_deployment, get_nodes_of_a_deployment, get_edge_placed_pods

CLOUD_RESPONSE_TIME = 300
EDGE_RESPONSE_TIME = 50

# ECMUS_INDEX = 0
# KUBE_SCHEDULE_INDEX = 1
# ECMUS_NO_MIGRATION_INDEX = 2
# RANDOM_INDEX = 3
# CLOUD_FIRST_INDEX = 4
# SMALLEST_EDGE_FIRST_INDEX = 5
# BIGGEST_EDGE_FIRST_INDEX = 6
# ECMUS_QOS_AWARE_INDEX = 7
# ECMUS_ALL_HALF_INDEX = -1
# ECMUS_ONLY_D_INDEX = -1
# ECMUS_C_GT_A_INDEX = -1

# ECMUS_INDEX = -1
# KUBE_SCHEDULE_INDEX = 0
# ECMUS_NO_MIGRATION_INDEX = -1
# RANDOM_INDEX = 1
# CLOUD_FIRST_INDEX = 2
# SMALLEST_EDGE_FIRST_INDEX = 3
# BIGGEST_EDGE_FIRST_INDEX = 4
# ECMUS_QOS_AWARE_INDEX = 5
# ECMUS_ALL_HALF_INDEX = -1
# ECMUS_ONLY_D_INDEX = -1
# ECMUS_C_GT_A_INDEX = -1

ECMUS_INDEX = -1
KUBE_SCHEDULE_INDEX = -1
RANDOM_INDEX = -1
CLOUD_FIRST_INDEX = -1
SMALLEST_EDGE_FIRST_INDEX = -1
BIGGEST_EDGE_FIRST_INDEX = -1
ECMUS_ALL_HALF_INDEX = -1
ECMUS_ONLY_D_INDEX = -1
ECMUS_C_GT_A_INDEX = -1
ECMUS_QOS_AWARE_INDEX = 0
ECMUS_NO_CLOUD_OFFLOAD_INDEX = 1
ECMUS_NO_EDGE_MIGRATION_INDEX = 2
ECMUS_NO_MIGRATION_INDEX = 3
ECMUS_MID_MIGRATION_INDEX= 4

INDEX_COUNT = 5

METADATA_FILENAME = "metadata.json"


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
def migration_count_metadata(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
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

            if deployment not in cycle.hpa.deployment_metrics or ceil(cycle.hpa.deployment_metrics[deployment]) != number_of_pods:
                return False

        return True

    data = {
        # "Kube": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "KubeDSM": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        "KubeDSMNoMigration": {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        # "Random": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "CloudFirst": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "BiggestEdgeFirst": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "SmallestEdgeFirst": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        "KubeDSMQOSAware": {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        # "KubeDSMAllHalf": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "KubeDSMOnlyD": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "KubeDSMCgtA": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        "KubeDSMNoCloudOffload" : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "KubeDSMNoEdgeMigration" : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "KubeDSMMidMigration" : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
    }

    ensure_directory(save_path)
    metadata_filepath = os.path.join(save_path, METADATA_FILENAME)

    metadata = {scheduler: {} for scheduler in data.keys()}

    if not os.path.exists(metadata_filepath):
        with open(metadata_filepath, "w") as file:
            json.dump(metadata, file)

    else:
        with open(metadata_filepath, "r") as file:
            metadata = json.load(file)

    for deployment in config.deployments.values():
        for id, history in enumerate(histories):
            # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
            index = id % INDEX_COUNT
            migrations: List[Migration] = []
            for (start, cycle) in enumerate(history.cycles):
                if not is_cycle_valid(cycle):
                    continue

                for (_end, next_cycle) in enumerate(history.cycles[start + 1:]):
                    end = start + _end + 1

                    if deployment not in next_cycle.hpa.deployment_metrics \
                        or ceil(cycle.hpa.deployment_metrics[deployment]) != ceil(next_cycle.hpa.deployment_metrics[deployment]):
                        break

                    if not is_cycle_valid(next_cycle):
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

            if index == ECMUS_INDEX:
                data["KubeDSM"][deployment.name].append(len(migrations))

            if index == KUBE_SCHEDULE_INDEX:
                data["Kube"][deployment.name].append(len(migrations))

            if index == ECMUS_NO_MIGRATION_INDEX:
                data["KubeDSMNoMigration"][deployment.name].append(len(migrations))

            if index == RANDOM_INDEX:
                data["Random"][deployment.name].append(len(migrations))

            if index == CLOUD_FIRST_INDEX:
                data["CloudFirst"][deployment.name].append(len(migrations))

            if index == SMALLEST_EDGE_FIRST_INDEX:
                data["SmallestEdgeFirst"][deployment.name].append(len(migrations))

            if index == BIGGEST_EDGE_FIRST_INDEX:
                data["BiggestEdgeFirst"][deployment.name].append(len(migrations))

            if index == ECMUS_QOS_AWARE_INDEX:
                data["KubeDSMQOSAware"][deployment.name].append(len(migrations))

            if index == ECMUS_ALL_HALF_INDEX:
                data["KubeDSMAllHalf"][deployment.name].append(len(migrations))

            if index == ECMUS_ONLY_D_INDEX:
                data["KubeDSMOnlyD"][deployment.name].append(len(migrations))

            if index == ECMUS_C_GT_A_INDEX:
                data["KubeDSMCgtA"][deployment.name].append(len(migrations))

            if index == ECMUS_NO_CLOUD_OFFLOAD_INDEX:
                data["KubeDSMNoCloudOffload"][deployment.name].append(len(migrations))

            if index == ECMUS_NO_EDGE_MIGRATION_INDEX:
                data["KubeDSMNoEdgeMigration"][deployment.name].append(len(migrations))

    for scheduler in data.keys():
        for deployment in config.deployments:
            data[scheduler][deployment] = sum(data[scheduler][deployment]) / len(data[scheduler][deployment])

    for scheduler in metadata.keys():
        metadata[scheduler][scenario_name] = data[scheduler]

    with open(metadata_filepath, "w") as file:
        json.dump(metadata, file)


@register_extractor
def migration_count_boxplot(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    metadata_filepath = os.path.join(save_path, METADATA_FILENAME)

    if not os.path.exists(metadata_filepath):
        return

    with open(metadata_filepath, "r") as file:
        metadata = json.load(file)

    ensure_directory(save_path)

    scheduler_data = {}

    for scheduler, scenarios in metadata.items():
        scenario_data = {}
        for scenario, values in scenarios.items():
            scenario_data[scenario] = values
        scheduler_data[scheduler] = scenario_data

    num_schedulers = len(scheduler_data)
    fig, axs = plt.subplots(num_schedulers, 1, figsize=(12, 6 * num_schedulers), sharex=True)

    if num_schedulers == 1:
        axs = [axs]

    for ax, (scheduler, scenario_data) in zip(axs, scheduler_data.items()):
        scenarios = list(scenario_data.keys())
        num_scenarios = len(scenarios)

        bar_width = 0.2
        index = np.arange(num_scenarios)

        values_a = [scenario_data[sc]['a'] for sc in scenarios]
        values_b = [scenario_data[sc]['b'] for sc in scenarios]
        values_c = [scenario_data[sc]['c'] for sc in scenarios]
        values_d = [scenario_data[sc]['d'] for sc in scenarios]

        bars_a = ax.bar(index - 1.5 * bar_width, values_a, bar_width, label='a', color='blue')
        bars_b = ax.bar(index - 0.5 * bar_width, values_b, bar_width, label='b', color='orange')
        bars_c = ax.bar(index + 0.5 * bar_width, values_c, bar_width, label='c', color='green')
        bars_d = ax.bar(index + 1.5 * bar_width, values_d, bar_width, label='d', color='red')

        ax.set_title(f'Values for Each Scenario - {scheduler}')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Value')

        ax.legend()

        ax.set_xticks(index)
        ax.set_xticklabels(scenarios, rotation=90, ha='right')

        for bars in [bars_a, bars_b, bars_c, bars_d]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', 
                        ha='center', va='bottom')

    plt.savefig(save_path + f"/individual.png")

    scheduler_sums = {}

    for scheduler, scenarios in metadata.items():
        scenario_sums = {}
        for scenario, values in scenarios.items():
            total = values.get('a', 0) + values.get('b', 0) + values.get('c', 0) + values.get('d', 0)
            scenario_sums[scenario] = total
        scheduler_sums[scheduler] = scenario_sums

    num_schedulers = len(scheduler_sums)
    fig, axs = plt.subplots(num_schedulers, 1, figsize=(10, 6 * num_schedulers), sharex=True)

    if num_schedulers == 1:
        axs = [axs]

    for ax, (scheduler, scenario_sums) in zip(axs, scheduler_sums.items()):
        scenarios = list(scenario_sums.keys())
        sums = list(scenario_sums.values())

        bars = ax.bar(scenarios, sums, color='skyblue')

        ax.set_title(f"Sum of Migrations for Each Scenario - {scheduler}")
        ax.set_xlabel("Scenario")
        ax.set_xticks(scenarios)
        ax.set_xticklabels(scenarios, rotation = 90)
        ax.set_ylabel("Sum of Migrations")

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height}", 
                    ha="center", va="bottom")

    plt.savefig(save_path + f"/sum.png")


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

    ecmus_all_half_pod_count_list = []
    ecmus_all_half_timestamps_list = []

    ecmus_only_d_pod_count_list = []
    ecmus_only_d_timestamps_list = []

    ecmus_c_gt_a_pod_count_list = []
    ecmus_c_gt_a_timestamps_list = []

    ecmus_no_cloud_offload_pod_count_list = []
    ecmus_no_cloud_offload_timestamps_list = []

    ecmus_no_edge_migration_pod_count_list = []
    ecmus_no_edge_migration_timestamps_list = []

    ecmus_mid_migration_pod_count_list = []
    ecmus_mid_migration_timestamps_list = []

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

        ecmus_all_half_pod_count = {it: [] for it in range(box_count)}
        ecmus_all_half_timestamps = {it: [] for it in range(box_count)}

        ecmus_only_d_pod_count = {it: [] for it in range(box_count)}
        ecmus_only_d_timestamps = {it: [] for it in range(box_count)}

        ecmus_c_gt_a_pod_count = {it: [] for it in range(box_count)}
        ecmus_c_gt_a_timestamps = {it: [] for it in range(box_count)}

        ecmus_no_cloud_offload_pod_count = {it: [] for it in range(box_count)}
        ecmus_no_cloud_offload_timestamps = {it: [] for it in range(box_count)}

        ecmus_no_edge_migration_pod_count = {it: [] for it in range(box_count)}
        ecmus_no_edge_migration_timestamps = {it: [] for it in range(box_count)}

        ecmus_mid_migration_pod_count = {it: [] for it in range(box_count)}
        ecmus_mid_migration_timestamps = {it: [] for it in range(box_count)}

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

                if index == ECMUS_ALL_HALF_INDEX:
                    ecmus_all_half_pod_count[box_id].append(pod_count)
                    ecmus_all_half_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_ONLY_D_INDEX:
                    ecmus_only_d_pod_count[box_id].append(pod_count)
                    ecmus_only_d_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_C_GT_A_INDEX:
                    ecmus_c_gt_a_pod_count[box_id].append(pod_count)
                    ecmus_c_gt_a_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_NO_CLOUD_OFFLOAD_INDEX:
                    ecmus_no_cloud_offload_pod_count[box_id].append(pod_count)
                    ecmus_no_cloud_offload_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_NO_EDGE_MIGRATION_INDEX:
                    ecmus_no_edge_migration_pod_count[box_id].append(pod_count)
                    ecmus_no_edge_migration_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_MID_MIGRATION_INDEX:
                    ecmus_mid_migration_pod_count[box_id].append(pod_count)
                    ecmus_mid_migration_timestamps[box_id].append(cycle.timestamp)

        # ecmus_pod_count = merge_lists_by_average(*[ecmus_pod_count[it] for it in range(box_count)])
        # ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])
        #
        # kube_pod_count = merge_lists_by_average(*[kube_pod_count[it] for it in range(box_count)])
        # kube_timestamps = merge_lists_by_average(*[kube_timestamps[it] for it in range(box_count)])

        ecmus_no_migration_pod_count = merge_lists_by_average(*[ecmus_no_migration_pod_count[it] for it in range(box_count)])
        ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])

        # cloud_first_pod_count = merge_lists_by_average(*[cloud_first_pod_count[it] for it in range(box_count)])
        # cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])
        #
        # random_pod_count = merge_lists_by_average(*[random_pod_count[it] for it in range(box_count)])
        # random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])
        #
        # smallest_edge_first_pod_count = merge_lists_by_average(*[smallest_edge_first_pod_count[it] for it in range(box_count)])
        # smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])
        #
        # biggest_edge_first_pod_count = merge_lists_by_average(*[biggest_edge_first_pod_count[it] for it in range(box_count)])
        # biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])

        ecmus_qos_aware_pod_count = merge_lists_by_average(*[ecmus_qos_aware_pod_count[it] for it in range(box_count)])
        ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps[it] for it in range(box_count)])

        # ecmus_all_half_pod_count = merge_lists_by_average(*[ecmus_all_half_pod_count[it] for it in range(box_count)])
        # ecmus_all_half_timestamps = merge_lists_by_average(*[ecmus_all_half_timestamps[it] for it in range(box_count)])
        #
        # ecmus_only_d_pod_count = merge_lists_by_average(*[ecmus_only_d_pod_count[it] for it in range(box_count)])
        # ecmus_only_d_timestamps = merge_lists_by_average(*[ecmus_only_d_timestamps[it] for it in range(box_count)])
        #
        # ecmus_c_gt_a_pod_count = merge_lists_by_average(*[ecmus_c_gt_a_pod_count[it] for it in range(box_count)])
        # ecmus_c_gt_a_timestamps = merge_lists_by_average(*[ecmus_c_gt_a_timestamps[it] for it in range(box_count)])

        ecmus_no_cloud_offload_pod_count = merge_lists_by_average(*[ecmus_no_cloud_offload_pod_count[it] for it in range(box_count)])
        ecmus_no_cloud_offload_timestamps = merge_lists_by_average(*[ecmus_no_cloud_offload_timestamps[it] for it in range(box_count)])

        ecmus_no_edge_migration_pod_count = merge_lists_by_average(*[ecmus_no_edge_migration_pod_count[it] for it in range(box_count)])
        ecmus_no_edge_migration_timestamps = merge_lists_by_average(*[ecmus_no_edge_migration_timestamps[it] for it in range(box_count)])

        ecmus_mid_migration_pod_count = merge_lists_by_average(*[ecmus_mid_migration_pod_count[it] for it in range(box_count)])
        ecmus_mid_migration_timestamps = merge_lists_by_average(*[ecmus_mid_migration_timestamps[it] for it in range(box_count)])

        # ecmus_pod_count_list.append(ecmus_pod_count)
        # ecmus_timestamps_list.append(ecmus_timestamps)
        #
        # kube_pod_count_list.append(kube_pod_count)
        # kube_timestamps_list.append(kube_timestamps)

        ecmus_no_migration_pod_count_list.append(ecmus_no_migration_pod_count)
        ecmus_no_migration_timestamps_list.append(ecmus_no_migration_timestamps)

        # cloud_first_pod_count_list.append(cloud_first_pod_count)
        # cloud_first_timestamps_list.append(cloud_first_timestamps)
        #
        # random_pod_count_list.append(random_pod_count)
        # random_timestamps_list.append(random_timestamps)
        #
        # smallest_edge_first_pod_count_list.append(smallest_edge_first_pod_count)
        # smallest_edge_first_timestamps_list.append(smallest_edge_first_timestamps)
        #
        # biggest_edge_first_pod_count_list.append(biggest_edge_first_pod_count)
        # biggest_edge_first_timestamps_list.append(biggest_edge_first_timestamps)

        ecmus_qos_aware_pod_count_list.append(ecmus_qos_aware_pod_count)
        ecmus_qos_aware_timestamps_list.append(ecmus_qos_aware_timestamps)

        # ecmus_all_half_pod_count_list.append(ecmus_all_half_pod_count)
        # ecmus_all_half_timestamps_list.append(ecmus_all_half_timestamps)
        #
        # ecmus_only_d_pod_count_list.append(ecmus_only_d_pod_count)
        # ecmus_only_d_timestamps_list.append(ecmus_only_d_timestamps)
        #
        # ecmus_c_gt_a_pod_count_list.append(ecmus_c_gt_a_pod_count)
        # ecmus_c_gt_a_timestamps_list.append(ecmus_c_gt_a_timestamps)

        ecmus_no_cloud_offload_pod_count_list.append(ecmus_no_cloud_offload_pod_count)
        ecmus_no_cloud_offload_timestamps_list.append(ecmus_no_cloud_offload_timestamps)

        ecmus_no_edge_migration_pod_count_list.append(ecmus_no_edge_migration_pod_count)
        ecmus_no_edge_migration_timestamps_list.append(ecmus_no_edge_migration_timestamps)

        ecmus_mid_migration_pod_count_list.append(ecmus_mid_migration_pod_count)
        ecmus_mid_migration_timestamps_list.append(ecmus_mid_migration_timestamps)

        fig, ax = plt.subplots()
        plt.grid(True, axis='y')
        fig.set_size_inches(10.5, 7.5)
        marker_interval = 2
        # ax.plot(kube_timestamps, kube_pod_count, label = "Kube", marker='o', markevery=marker_interval)
        # ax.plot(ecmus_timestamps, ecmus_pod_count, label = "KubeDSM", marker='s', markevery=marker_interval)
        ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_pod_count, label = "KubeDSMNoMigration", marker='D', markevery=marker_interval)
        # ax.plot(random_timestamps, random_pod_count, label = "Random", marker='^', markevery=marker_interval)
        # ax.plot(cloud_first_timestamps, cloud_first_pod_count, label = "CloudFirst", marker='v', markevery=marker_interval)
        # ax.plot(biggest_edge_first_timestamps, biggest_edge_first_pod_count, label = "BiggestEdgeFirst", marker='x', markevery=marker_interval)
        # ax.plot(smallest_edge_first_timestamps, smallest_edge_first_pod_count, label = "SmallestEdgeFirst", marker='+', markevery=marker_interval)
        ax.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_pod_count, label = "KubeDSMQOSAware", marker="*", markevery=marker_interval)
        # ax.plot(ecmus_all_half_timestamps, ecmus_all_half_pod_count, label = "KubeDSMAllHalf", marker="+", markevery=marker_interval)
        # ax.plot(ecmus_only_d_timestamps, ecmus_only_d_pod_count, label = "KubeDSMOnlyD", marker="x", markevery=marker_interval)
        # ax.plot(ecmus_c_gt_a_timestamps, ecmus_c_gt_a_pod_count, label = "KubeDSMCgtA", marker="v", markevery=marker_interval)
        ax.plot(ecmus_no_cloud_offload_timestamps, ecmus_no_cloud_offload_pod_count, label = "KubeDSMNoCloudOffload", marker="v", markevery=marker_interval)
        ax.plot(ecmus_no_edge_migration_timestamps, ecmus_no_edge_migration_pod_count, label = "KubeDSMNoEdgeMigration", marker="x", markevery=marker_interval)
        ax.plot(ecmus_mid_migration_timestamps, ecmus_mid_migration_pod_count, label = "KubeDSMMidMigration", marker="s", markevery=marker_interval)
        max_val = int(np.max(np.concatenate([
            # kube_pod_count,
            # ecmus_pod_count,
            ecmus_no_migration_pod_count,
            # random_pod_count,
            # cloud_first_pod_count,
            # biggest_edge_first_pod_count,
            # smallest_edge_first_pod_count,
            # ecmus_qos_aware_pod_count,
            # ecmus_all_half_pod_count,
            # ecmus_only_d_pod_count,
            # ecmus_c_gt_a_pod_count,
            ecmus_no_cloud_offload_pod_count,
            ecmus_no_edge_migration_pod_count,
            ecmus_mid_migration_pod_count,
        ]))) + 1
        ax.set_ylim(0, max_val)
        ax.set_yticks(range(0, max_val, 1))
        plt.xlabel("time(s)")
        plt.ylabel("pod count")
        plt.title(f"pod count - workload: {deployment.name}")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=12, frameon=True, ncol=3)
        plt.tight_layout()
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")

    deployment_count = len(config.deployments)

    # ecmus_pod_count = merge_lists_by_sum(*[ecmus_pod_count_list[it] for it in range(deployment_count)])
    # ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps_list[it] for it in range(deployment_count)])
    #
    # kube_pod_count = merge_lists_by_sum(*[kube_pod_count_list[it] for it in range(deployment_count)])
    # kube_timestamps = merge_lists_by_average(*[kube_timestamps_list[it] for it in range(deployment_count)])

    ecmus_no_migration_pod_count = merge_lists_by_sum(*[ecmus_no_migration_pod_count_list[it] for it in range(deployment_count)])
    ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps_list[it] for it in range(deployment_count)])

    # random_pod_count = merge_lists_by_sum(*[random_pod_count_list[it] for it in range(deployment_count)])
    # random_timestamps = merge_lists_by_average(*[random_timestamps_list[it] for it in range(deployment_count)])
    #
    # cloud_first_pod_count = merge_lists_by_sum(*[cloud_first_pod_count_list[it] for it in range(deployment_count)])
    # cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps_list[it] for it in range(deployment_count)])
    #
    # smallest_edge_first_pod_count = merge_lists_by_sum(*[smallest_edge_first_pod_count_list[it] for it in range(deployment_count)])
    # smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps_list[it] for it in range(deployment_count)])
    #
    # biggest_edge_first_pod_count = merge_lists_by_sum(*[biggest_edge_first_pod_count_list[it] for it in range(deployment_count)])
    # biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps_list[it] for it in range(deployment_count)])

    ecmus_qos_aware_pod_count = merge_lists_by_sum(*[ecmus_qos_aware_pod_count_list[it] for it in range(deployment_count)])
    ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps_list[it] for it in range(deployment_count)])

    # ecmus_all_half_pod_count = merge_lists_by_sum(*[ecmus_all_half_pod_count_list[it] for it in range(deployment_count)])
    # ecmus_all_half_timestamps = merge_lists_by_average(*[ecmus_all_half_timestamps_list[it] for it in range(deployment_count)])
    #
    # ecmus_only_d_pod_count = merge_lists_by_sum(*[ecmus_only_d_pod_count_list[it] for it in range(deployment_count)])
    # ecmus_only_d_timestamps = merge_lists_by_average(*[ecmus_only_d_timestamps_list[it] for it in range(deployment_count)])
    #
    # ecmus_c_gt_a_pod_count = merge_lists_by_sum(*[ecmus_c_gt_a_pod_count_list[it] for it in range(deployment_count)])
    # ecmus_c_gt_a_timestamps = merge_lists_by_average(*[ecmus_c_gt_a_timestamps_list[it] for it in range(deployment_count)])

    ecmus_no_cloud_offload_pod_count = merge_lists_by_sum(*[ecmus_no_cloud_offload_pod_count_list[it] for it in range(deployment_count)])
    ecmus_no_cloud_offload_timestamps = merge_lists_by_average(*[ecmus_no_cloud_offload_timestamps_list[it] for it in range(deployment_count)])

    ecmus_no_edge_migration_pod_count = merge_lists_by_sum(*[ecmus_no_edge_migration_pod_count_list[it] for it in range(deployment_count)])
    ecmus_no_edge_migration_timestamps = merge_lists_by_average(*[ecmus_no_edge_migration_timestamps_list[it] for it in range(deployment_count)])

    ecmus_mid_migration_pod_count = merge_lists_by_sum(*[ecmus_mid_migration_pod_count_list[it] for it in range(deployment_count)])
    ecmus_mid_migration_timestamps = merge_lists_by_average(*[ecmus_mid_migration_timestamps_list[it] for it in range(deployment_count)])

    fig, ax = plt.subplots()
    plt.grid(True, axis='y')
    fig.set_size_inches(10.5, 7.5)
    markevery=marker_interval
    # ax.plot(kube_timestamps, kube_pod_count, label = "Kube", marker='o', markevery=marker_interval)
    # ax.plot(ecmus_timestamps, ecmus_pod_count, label = "KubeDSM", marker='s', markevery=marker_interval)
    ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_pod_count, label = "KubeDSMNoMigration", marker='D', markevery=marker_interval)
    # ax.plot(random_timestamps, random_pod_count, label = "Random", marker='^', markevery=marker_interval)
    # ax.plot(cloud_first_timestamps, cloud_first_pod_count, label = "CloudFirst", marker='v', markevery=marker_interval)
    # ax.plot(biggest_edge_first_timestamps, biggest_edge_first_pod_count, label = "BiggestEdgeFirst", marker='x', markevery=marker_interval)
    # ax.plot(smallest_edge_first_timestamps, smallest_edge_first_pod_count, label = "SmallestEdgeFirst", marker='+', markevery=marker_interval)
    ax.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_pod_count, label = "KubeDSMQOSAware", marker="*", markevery=marker_interval)
    # ax.plot(ecmus_all_half_timestamps, ecmus_all_half_pod_count, label = "KubeDSMAllHalf", marker="+", markevery=marker_interval)
    # ax.plot(ecmus_only_d_timestamps, ecmus_only_d_pod_count, label = "KubeDSMOnlyD", marker="x", markevery=marker_interval)
    # ax.plot(ecmus_c_gt_a_timestamps, ecmus_c_gt_a_pod_count, label = "KubeDSMCgtA", marker="v", markevery=marker_interval)
    ax.plot(ecmus_no_cloud_offload_timestamps, ecmus_no_cloud_offload_pod_count, label = "KubeDSMNoCloudOffload", marker="v", markevery=marker_interval)
    ax.plot(ecmus_no_edge_migration_timestamps, ecmus_no_edge_migration_pod_count, label = "KubeDSMNoEdgeMigration", marker="x", markevery=marker_interval)
    ax.plot(ecmus_mid_migration_timestamps, ecmus_mid_migration_pod_count, label = "KubeDSMMidMigration", marker="s", markevery=marker_interval)
    max_val = int(np.max(np.concatenate([
        # kube_pod_count,
        # ecmus_pod_count,
        ecmus_no_migration_pod_count,
        # random_pod_count,
        # cloud_first_pod_count,
        # biggest_edge_first_pod_count,
        # smallest_edge_first_pod_count,
        # ecmus_qos_aware_pod_count,
        # ecmus_all_half_pod_count,
        # ecmus_only_d_pod_count,
        # ecmus_c_gt_a_pod_count,
        ecmus_no_cloud_offload_pod_count,
        ecmus_no_edge_migration_pod_count,
        ecmus_mid_migration_pod_count,
    ]))) + 1
    ax.set_ylim(0, max_val)
    ax.set_yticks(range(0, max_val, 1))
    plt.xlabel("time(s)")
    plt.ylabel("pod count")
    plt.title(f"pod count - workload total")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=12, frameon=True, ncol=3)
    plt.tight_layout()
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

        ecmus_all_half_latencies = {it: [] for it in range(box_count)}
        ecmus_all_half_timestamps = {it: [] for it in range(box_count)}

        ecmus_only_d_latencies = {it: [] for it in range(box_count)}
        ecmus_only_d_timestamps = {it: [] for it in range(box_count)}

        ecmus_c_gt_a_latencies = {it: [] for it in range(box_count)}
        ecmus_c_gt_a_timestamps = {it: [] for it in range(box_count)}

        ecmus_no_cloud_offload_latencies = {it: [] for it in range(box_count)}
        ecmus_no_cloud_offload_timestamps = {it: [] for it in range(box_count)}

        ecmus_no_edge_migration_latencies = {it: [] for it in range(box_count)}
        ecmus_no_edge_migration_timestamps = {it: [] for it in range(box_count)}

        ecmus_mid_migration_latencies = {it: [] for it in range(box_count)}
        ecmus_mid_migration_timestamps = {it: [] for it in range(box_count)}

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

                if index == ECMUS_ALL_HALF_INDEX:
                    ecmus_all_half_latencies[box_id].append(latency)
                    ecmus_all_half_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_ONLY_D_INDEX:
                    ecmus_only_d_latencies[box_id].append(latency)
                    ecmus_only_d_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_C_GT_A_INDEX:
                    ecmus_c_gt_a_latencies[box_id].append(latency)
                    ecmus_c_gt_a_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_NO_CLOUD_OFFLOAD_INDEX:
                    ecmus_no_cloud_offload_latencies[box_id].append(latency)
                    ecmus_no_cloud_offload_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_NO_EDGE_MIGRATION_INDEX:
                    ecmus_no_edge_migration_latencies[box_id].append(latency)
                    ecmus_no_edge_migration_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_MID_MIGRATION_INDEX:
                    ecmus_mid_migration_latencies[box_id].append(latency)
                    ecmus_mid_migration_timestamps[box_id].append(cycle.timestamp)

        # ecmus_latencies = merge_lists_by_average(*[ecmus_latencies[it] for it in range(box_count)])
        # ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])
        #
        # kube_latencies = merge_lists_by_average(*[kube_latencies[it] for it in range(box_count)])
        # kube_timestamps = merge_lists_by_average(*[kube_timestamps[it] for it in range(box_count)])

        ecmus_no_migration_latencies = merge_lists_by_average(*[ecmus_no_migration_latencies[it] for it in range(box_count)])
        ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])

        # cloud_first_latencies = merge_lists_by_average(*[cloud_first_latencies[it] for it in range(box_count)])
        # cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])
        #
        # random_latencies = merge_lists_by_average(*[random_latencies[it] for it in range(box_count)])
        # random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])
        #
        # smallest_edge_first_latencies = merge_lists_by_average(*[smallest_edge_first_latencies[it] for it in range(box_count)])
        # smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])
        #
        # biggest_edge_first_latencies = merge_lists_by_average(*[biggest_edge_first_latencies[it] for it in range(box_count)])
        # biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])

        ecmus_qos_aware_latencies = merge_lists_by_average(*[ecmus_qos_aware_latencies[it] for it in range(box_count)])
        ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps[it] for it in range(box_count)])

        # ecmus_all_half_latencies = merge_lists_by_average(*[ecmus_all_half_latencies[it] for it in range(box_count)])
        # ecmus_all_half_timestamps = merge_lists_by_average(*[ecmus_all_half_timestamps[it] for it in range(box_count)])
        #
        # ecmus_only_d_latencies = merge_lists_by_average(*[ecmus_only_d_latencies[it] for it in range(box_count)])
        # ecmus_only_d_timestamps = merge_lists_by_average(*[ecmus_only_d_timestamps[it] for it in range(box_count)])
        #
        # ecmus_c_gt_a_latencies = merge_lists_by_average(*[ecmus_c_gt_a_latencies[it] for it in range(box_count)])
        # ecmus_c_gt_a_timestamps = merge_lists_by_average(*[ecmus_c_gt_a_timestamps[it] for it in range(box_count)])

        ecmus_no_cloud_offload_latencies = merge_lists_by_average(*[ecmus_no_cloud_offload_latencies[it] for it in range(box_count)])
        ecmus_no_cloud_offload_timestamps = merge_lists_by_average(*[ecmus_no_cloud_offload_timestamps[it] for it in range(box_count)])

        ecmus_no_edge_migration_latencies = merge_lists_by_average(*[ecmus_no_edge_migration_latencies[it] for it in range(box_count)])
        ecmus_no_edge_migration_timestamps = merge_lists_by_average(*[ecmus_no_edge_migration_timestamps[it] for it in range(box_count)])

        ecmus_mid_migration_latencies = merge_lists_by_average(*[ecmus_mid_migration_latencies[it] for it in range(box_count)])
        ecmus_mid_migration_timestamps = merge_lists_by_average(*[ecmus_mid_migration_timestamps[it] for it in range(box_count)])

        fig, ax = plt.subplots()
        plt.grid(True, axis='y')
        fig.set_size_inches(10.5, 7.5)
        marker_interval = 2
        # ax.plot(kube_timestamps, kube_latencies, label = "Kube", marker='o', markevery=marker_interval)
        # ax.plot(ecmus_timestamps, ecmus_latencies, label = "KubeDSM", marker='s', markevery=marker_interval)
        ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_latencies, label = "KubeDSMNoMigration", marker='D', markevery=marker_interval)
        # ax.plot(random_timestamps, random_latencies, label = "Random", marker='^', markevery=marker_interval)
        # ax.plot(cloud_first_timestamps, cloud_first_latencies, label = "CloudFirst", marker='v', markevery=marker_interval)
        # ax.plot(biggest_edge_first_timestamps, biggest_edge_first_latencies, label = "BiggestEdgeFirst", marker='x', markevery=marker_interval)
        # ax.plot(smallest_edge_first_timestamps, smallest_edge_first_latencies, label = "SmallestEdgeFirst", marker='+', markevery=marker_interval)
        ax.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_latencies, label = "KubeDSMQOSAware", marker="*", markevery=marker_interval)
        # ax.plot(ecmus_all_half_timestamps, ecmus_all_half_latencies, label = "KubeDSMAllHalf", marker="+", markevery=marker_interval)
        # ax.plot(ecmus_only_d_timestamps, ecmus_only_d_latencies, label = "KubeDSMOnlyD", marker="x", markevery=marker_interval)
        # ax.plot(ecmus_c_gt_a_timestamps, ecmus_c_gt_a_latencies, label = "KubeDSMCgtA", marker="v", markevery=marker_interval)
        ax.plot(ecmus_no_cloud_offload_timestamps, ecmus_no_cloud_offload_latencies, label = "KubeDSMNoCloudOffload", marker="v", markevery=marker_interval)
        ax.plot(ecmus_no_edge_migration_timestamps, ecmus_no_edge_migration_latencies, label = "KubeDSMNoEdgeMigration", marker="x", markevery=marker_interval)
        ax.plot(ecmus_mid_migration_timestamps, ecmus_mid_migration_latencies, label = "KubeDSMMidMigration", marker="s", markevery=marker_interval)
        plt.xlabel("time(s)")
        plt.ylabel("average latency(ms)")
        plt.title(f"average latency - workload: {deployment.name}")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=12, frameon=True, ncol=3)
        plt.tight_layout()
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")


def ensure_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


@register_extractor
def average_latency_boxplot(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    data = {
        # "Kube"               : {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "KubeDSM"                        : {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        "KubeDSMNoMigration"           : {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        # "Random"             : {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "CloudFirst"        : {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "BiggestEdgeFirst" : {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "SmallestEdgeFirst": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        "KubeDSMQOSAware": {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        # "KubeDSMAllHalf": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "KubeDSMOnlyD": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        # "KubeDSMCgtA": {
        #     "a": [],
        #     "b": [],
        #     "c": [],
        #     "d": [],
        # },
        "KubeDSMNoCloudOffload": {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "KubeDSMNoEdgeMigration": {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        },
        "KubeDSMMidMigration": {
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

                if index == ECMUS_ALL_HALF_INDEX:
                    data["KubeDSMAllHalf"][deployment.name].append(latency)

                if index == ECMUS_ONLY_D_INDEX:
                    data["KubeDSMOnlyD"][deployment.name].append(latency)

                if index == ECMUS_C_GT_A_INDEX:
                    data["KubeDSMCgtA"][deployment.name].append(latency)

                if index == ECMUS_NO_CLOUD_OFFLOAD_INDEX:
                    data["KubeDSMNoCloudOffload"][deployment.name].append(latency)

                if index == ECMUS_NO_EDGE_MIGRATION_INDEX:
                    data["KubeDSMNoEdgeMigration"][deployment.name].append(latency)

                if index == ECMUS_MID_MIGRATION_INDEX:
                    data["KubeDSMMidMigration"][deployment.name].append(latency)

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

    x = np.arange(len(data.keys())) * 1.5
    width = 0.3

    fig, ax = plt.subplots(layout = "constrained")
    fig.set_size_inches(10.5, 7.5)

    rects1 = ax.bar(x - 3 * width / 2, a_means, width, label = 'a')
    ax.bar_label(rects1, padding = 10, fontsize=8)
    rects2 = ax.bar(x - width / 2, b_means, width, label = 'b')
    ax.bar_label(rects2, padding = 10, fontsize=8)
    rects3 = ax.bar(x + width / 2, c_means, width, label = 'c')
    ax.bar_label(rects3, padding = 10, fontsize=8)
    rects4 = ax.bar(x + 3 * width / 2, d_means, width, label = 'd')
    ax.bar_label(rects4, padding = 10, fontsize=8)

    plt.grid(True, axis='y')
    plt.xlabel("scheduler")
    plt.ylabel("average latency(ms)")
    ax.set_title('average latency')
    ax.set_xticks(x)
    ax.set_xticklabels(data.keys(), rotation = 0)
    ax.set_ylim(0, 350)
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")
    # plt.show()


@register_extractor
def average_latency_metadata(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    data = {
        # "Kube": [],
        # "KubeDSM": [],
        "KubeDSMNoMigration": [],
        # "Random": [],
        # "CloudFirst": [],
        # "BiggestEdgeFirst": [],
        # "SmallestEdgeFirst": [],
        "KubeDSMQOSAware": [],
        # "KubeDSMAllHalf": [],
        # "KubeDSMOnlyD": [],
        # "KubeDSMCgtA": [],
        "KubeDSMNoCloudOffload": [],
        "KubeDSMNoEdgeMigration": [],
        "KubeDSMMidMigration": [],
    }

    ensure_directory(save_path)
    metadata_filepath = os.path.join(save_path, METADATA_FILENAME)

    metadata = {scheduler: {} for scheduler in data.keys()}

    if not os.path.exists(metadata_filepath):
        with open(metadata_filepath, "w") as file:
            json.dump(metadata, file)

    else:
        with open(metadata_filepath, "r") as file:
            metadata = json.load(file)

    for deployment in config.deployments.values():
        for id, history in enumerate(histories):
            # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
            index = id % INDEX_COUNT
            for cycle in history.cycles:
                cloud_pods_count, edge_pods_count = calculate_placement_for_deployment(cycle, deployment)
                all_pods_count = cloud_pods_count + edge_pods_count
                latency = (cloud_pods_count * CLOUD_RESPONSE_TIME + edge_pods_count * EDGE_RESPONSE_TIME) / all_pods_count

                if index == ECMUS_INDEX:
                    data["KubeDSM"].append(latency)

                if index == KUBE_SCHEDULE_INDEX:
                    data["Kube"].append(latency)

                if index == ECMUS_NO_MIGRATION_INDEX:
                    data["KubeDSMNoMigration"].append(latency)

                if index == RANDOM_INDEX:
                    data["Random"].append(latency)

                if index == CLOUD_FIRST_INDEX:
                    data["CloudFirst"].append(latency)

                if index == SMALLEST_EDGE_FIRST_INDEX:
                    data["SmallestEdgeFirst"].append(latency)

                if index == BIGGEST_EDGE_FIRST_INDEX:
                    data["BiggestEdgeFirst"].append(latency)

                if index == ECMUS_QOS_AWARE_INDEX:
                    data["KubeDSMQOSAware"].append(latency)

                if index == ECMUS_ALL_HALF_INDEX:
                    data["KubeDSMAllHalf"].append(latency)

                if index == ECMUS_ONLY_D_INDEX:
                    data["KubeDSMOnlyD"].append(latency)

                if index == ECMUS_C_GT_A_INDEX:
                    data["KubeDSMCgtA"].append(latency)

                if index == ECMUS_NO_CLOUD_OFFLOAD_INDEX:
                    data["KubeDSMNoCloudOffload"].append(latency)

                if index == ECMUS_NO_EDGE_MIGRATION_INDEX:
                    data["KubeDSMNoEdgeMigration"].append(latency)

                if index == ECMUS_MID_MIGRATION_INDEX:
                    data["KubeDSMMidMigration"].append(latency)

    for scheduler in metadata.keys():
        metadata[scheduler][scenario_name] = sum(data[scheduler]) / len(data[scheduler])

    with open(metadata_filepath, "w") as file:
        json.dump(metadata, file)


@register_extractor
def average_latency_table(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    metadata_filepath = os.path.join(save_path, METADATA_FILENAME)

    if not os.path.exists(metadata_filepath):
        return

    with open(metadata_filepath, "r") as file:
        metadata = json.load(file)

    ensure_directory(save_path)

    scheduler_data = {}

    for scheduler, scenarios in metadata.items():
        scenario_data = {}
        for scenario, values in scenarios.items():
            scenario_data[scenario] = values
        scheduler_data[scheduler] = scenario_data

    for scheduler in scheduler_data.keys():
        for scenario in scheduler_data[scheduler].keys():
            if scheduler != "Kube":
                scheduler_data[scheduler][scenario] /= scheduler_data["Kube"][scenario]

    for scenario in scheduler_data["Kube"].keys():
        scheduler_data["Kube"][scenario] = 1.0

    latencies_df = pd.DataFrame.from_dict(scheduler_data, orient='index')

    latencies_df.fillna('-', inplace=True)

    fig, ax = plt.subplots()
    ax.axis('off')

    table = ax.table(cellText=latencies_df.values,
                     colLabels=latencies_df.columns,
                     rowLabels=latencies_df.index,
                     cellLoc='center',
                     loc='center')

    plt.title('Latencies by Scheduler and Scenario')

    ensure_directory(save_path)
    plt.savefig(save_path + "/relative_latencies_table.png", bbox_inches='tight', dpi=300)


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

    ecmus_all_half_utilization = {it: [] for it in range(box_count)}
    ecmus_all_half_timestamps = {it: [] for it in range(box_count)}

    ecmus_only_d_utilization = {it: [] for it in range(box_count)}
    ecmus_only_d_timestamps = {it: [] for it in range(box_count)}

    ecmus_c_gt_a_utilization = {it: [] for it in range(box_count)}
    ecmus_c_gt_a_timestamps = {it: [] for it in range(box_count)}

    ecmus_no_cloud_offload_utilization = {it: [] for it in range(box_count)}
    ecmus_no_cloud_offload_timestamps = {it: [] for it in range(box_count)}

    ecmus_no_edge_migration_utilization = {it: [] for it in range(box_count)}
    ecmus_no_edge_migration_timestamps = {it: [] for it in range(box_count)}

    ecmus_mid_migration_utilization = {it: [] for it in range(box_count)}
    ecmus_mid_migration_timestamps = {it: [] for it in range(box_count)}

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

            if index == ECMUS_ALL_HALF_INDEX:
                ecmus_all_half_timestamps[box_id].append(cycle.timestamp)
                ecmus_all_half_utilization[box_id].append(utilization)

            if index == ECMUS_ONLY_D_INDEX:
                ecmus_only_d_timestamps[box_id].append(cycle.timestamp)
                ecmus_only_d_utilization[box_id].append(utilization)

            if index == ECMUS_C_GT_A_INDEX:
                ecmus_c_gt_a_timestamps[box_id].append(cycle.timestamp)
                ecmus_c_gt_a_utilization[box_id].append(utilization)

            if index == ECMUS_NO_CLOUD_OFFLOAD_INDEX:
                ecmus_no_cloud_offload_timestamps[box_id].append(cycle.timestamp)
                ecmus_no_cloud_offload_utilization[box_id].append(utilization)

            if index == ECMUS_NO_EDGE_MIGRATION_INDEX:
                ecmus_no_edge_migration_timestamps[box_id].append(cycle.timestamp)
                ecmus_no_edge_migration_utilization[box_id].append(utilization)

            if index == ECMUS_MID_MIGRATION_INDEX:
                ecmus_mid_migration_timestamps[box_id].append(cycle.timestamp)
                ecmus_mid_migration_utilization[box_id].append(utilization)

    # ecmus_utilization = merge_lists_by_average(*[ecmus_utilization[it] for it in range(box_count)])
    # ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])
    #
    # kube_schedule_utilization = merge_lists_by_average(*[kube_schedule_utilization[it] for it in range(box_count)])
    # kube_schedule_timestamps = merge_lists_by_average(*[kube_schedule_timestamps[it] for it in range(box_count)])

    ecmus_no_migration_utilization = merge_lists_by_average(*[ecmus_no_migration_utilization[it] for it in range(box_count)])
    ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])

    # cloud_first_utilization = merge_lists_by_average(*[cloud_first_utilization[it] for it in range(box_count)])
    # cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])
    #
    # random_utilization = merge_lists_by_average(*[random_utilization[it] for it in range(box_count)])
    # random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])
    #
    # smallest_edge_first_utilization = merge_lists_by_average(*[smallest_edge_first_utilization[it] for it in range(box_count)])
    # smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])
    #
    # biggest_edge_first_utilization = merge_lists_by_average(*[biggest_edge_first_utilization[it] for it in range(box_count)])
    # biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])

    ecmus_qos_aware_utilization = merge_lists_by_average(*[ecmus_qos_aware_utilization[it] for it in range(box_count)])
    ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps[it] for it in range(box_count)])

    # ecmus_all_half_utilization = merge_lists_by_average(*[ecmus_all_half_utilization[it] for it in range(box_count)])
    # ecmus_all_half_timestamps = merge_lists_by_average(*[ecmus_all_half_timestamps[it] for it in range(box_count)])
    #
    # ecmus_only_d_utilization = merge_lists_by_average(*[ecmus_only_d_utilization[it] for it in range(box_count)])
    # ecmus_only_d_timestamps = merge_lists_by_average(*[ecmus_only_d_timestamps[it] for it in range(box_count)])
    #
    # ecmus_c_gt_a_utilization = merge_lists_by_average(*[ecmus_c_gt_a_utilization[it] for it in range(box_count)])
    # ecmus_c_gt_a_timestamps = merge_lists_by_average(*[ecmus_c_gt_a_timestamps[it] for it in range(box_count)])

    ecmus_no_cloud_offload_utilization = merge_lists_by_average(*[ecmus_no_cloud_offload_utilization[it] for it in range(box_count)])
    ecmus_no_cloud_offload_timestamps = merge_lists_by_average(*[ecmus_no_cloud_offload_timestamps[it] for it in range(box_count)])

    ecmus_no_edge_migration_utilization = merge_lists_by_average(*[ecmus_no_edge_migration_utilization[it] for it in range(box_count)])
    ecmus_no_edge_migration_timestamps = merge_lists_by_average(*[ecmus_no_edge_migration_timestamps[it] for it in range(box_count)])

    ecmus_mid_migration_utilization = merge_lists_by_average(*[ecmus_mid_migration_utilization[it] for it in range(box_count)])
    ecmus_mid_migration_timestamps = merge_lists_by_average(*[ecmus_mid_migration_timestamps[it] for it in range(box_count)])

    fig, ax = plt.subplots()
    marker_interval = 2
    # plt.plot(ecmus_timestamps, ecmus_utilization, label = "KubeDSM", marker='o', markevery=marker_interval)
    # plt.plot(kube_schedule_timestamps, kube_schedule_utilization, label = "Kube", marker='s', markevery=marker_interval)
    plt.plot(ecmus_no_migration_timestamps, ecmus_no_migration_utilization, label = "KubeDSMNoMigration", marker='D', markevery=marker_interval)
    # plt.plot(random_timestamps, random_utilization, label = "Random", marker='^', markevery=marker_interval)
    # plt.plot(cloud_first_timestamps, cloud_first_utilization, label = "CloudFirst", marker='v', markevery=marker_interval)
    # plt.plot(smallest_edge_first_timestamps, smallest_edge_first_utilization, label = "SmallestEdgeFirst", marker='x', markevery=marker_interval)
    # plt.plot(biggest_edge_first_timestamps, biggest_edge_first_utilization, label = "BiggestEdgeFirst", marker='+', markevery=marker_interval)
    plt.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_utilization, label = "KubeDSMQOSAware", marker="*", markevery=marker_interval)
    # plt.plot(ecmus_all_half_timestamps, ecmus_all_half_utilization, label = "KubeDSMAllHalf", marker="+", markevery=marker_interval)
    # plt.plot(ecmus_only_d_timestamps, ecmus_only_d_utilization, label = "KubeDSMOnlyD", marker="x", markevery=marker_interval)
    # plt.plot(ecmus_c_gt_a_timestamps, ecmus_c_gt_a_utilization, label = "KubeDSMCgtA", marker="v", markevery=marker_interval)
    plt.plot(ecmus_no_cloud_offload_timestamps, ecmus_no_cloud_offload_utilization, label = "KubeDSMNoCloudOffload", marker="v", markevery=marker_interval)
    plt.plot(ecmus_no_edge_migration_timestamps, ecmus_no_edge_migration_utilization, label = "KubeDSMNoEdgeMigration", marker="x", markevery=marker_interval)
    plt.plot(ecmus_mid_migration_timestamps, ecmus_mid_migration_utilization, label = "KubeDSMMidMigration", marker="s", markevery=marker_interval)

    fig.set_size_inches(10.5, 7.5)
    plt.grid(True, axis='y')
    plt.xlabel("time (s)")
    plt.ylabel("edge utilization")
    plt.ylim(0, 1.10)
    plt.yticks(list(map(lambda x: x / 100.0, range(0, 110, 5))))
    plt.title("edge utilization - per scheduler")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=12, frameon=True, ncol=3)
    plt.tight_layout()
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

    ecmus_all_half_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_all_half_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_all_half_timestamps = {it: [] for it in range(box_count)}

    ecmus_only_d_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_only_d_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_only_d_timestamps = {it: [] for it in range(box_count)}

    ecmus_c_gt_a_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_c_gt_a_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_c_gt_a_timestamps = {it: [] for it in range(box_count)}

    ecmus_no_cloud_offload_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_no_cloud_offload_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_no_cloud_offload_timestamps = {it: [] for it in range(box_count)}

    ecmus_no_edge_migration_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_no_edge_migration_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_no_edge_migration_timestamps = {it: [] for it in range(box_count)}

    ecmus_mid_migration_edge_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_mid_migration_cloud_placement_ratio = {it: [] for it in range(box_count)}
    ecmus_mid_migration_timestamps = {it: [] for it in range(box_count)}

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

            if index == ECMUS_ALL_HALF_INDEX:
                ecmus_all_half_timestamps[box_id].append(cycle.timestamp)
                ecmus_all_half_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_all_half_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == ECMUS_ONLY_D_INDEX:
                ecmus_only_d_timestamps[box_id].append(cycle.timestamp)
                ecmus_only_d_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_only_d_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == ECMUS_C_GT_A_INDEX:
                ecmus_c_gt_a_timestamps[box_id].append(cycle.timestamp)
                ecmus_c_gt_a_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_c_gt_a_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == ECMUS_NO_CLOUD_OFFLOAD_INDEX:
                ecmus_no_cloud_offload_timestamps[box_id].append(cycle.timestamp)
                ecmus_no_cloud_offload_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_no_cloud_offload_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == ECMUS_NO_EDGE_MIGRATION_INDEX:
                ecmus_no_edge_migration_timestamps[box_id].append(cycle.timestamp)
                ecmus_no_edge_migration_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_no_edge_migration_cloud_placement_ratio[box_id].append(fragmentation_cloud)

            if index == ECMUS_MID_MIGRATION_INDEX:
                ecmus_mid_migration_timestamps[box_id].append(cycle.timestamp)
                ecmus_mid_migration_edge_placement_ratio[box_id].append(fragmentation_edge)
                ecmus_mid_migration_cloud_placement_ratio[box_id].append(fragmentation_cloud)

    # ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])
    # ecmus_edge_placement_ratio = merge_lists_by_average(*[ecmus_edge_placement_ratio[it] for it in range(box_count)])
    # ecmus_cloud_placement_ratio = merge_lists_by_average(*[ecmus_cloud_placement_ratio[it] for it in range(box_count)])
    #
    # kube_schedule_timestamps = merge_lists_by_average(*[kube_schedule_timestamps[it] for it in range(box_count)])
    # kube_schedule_edge_placement_ratio = merge_lists_by_average(
    #     *[kube_schedule_edge_placement_ratio[it] for it in range(box_count)])
    # kube_schedule_cloud_placement_ratio = merge_lists_by_average(
    #     *[kube_schedule_cloud_placement_ratio[it] for it in range(box_count)])

    ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])
    ecmus_no_migration_edge_placement_ratio = merge_lists_by_average(
        *[ecmus_no_migration_edge_placement_ratio[it] for it in range(box_count)])
    ecmus_no_migration_cloud_placement_ratio = merge_lists_by_average(
        *[ecmus_no_migration_cloud_placement_ratio[it] for it in range(box_count)])

    # random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])
    # random_edge_placement_ratio = merge_lists_by_average(*[random_edge_placement_ratio[it] for it in range(box_count)])
    # random_cloud_placement_ratio = merge_lists_by_average(*[random_cloud_placement_ratio[it] for it in range(box_count)])
    #
    # cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])
    # cloud_first_edge_placement_ratio = merge_lists_by_average(*[cloud_first_edge_placement_ratio[it] for it in range(box_count)])
    # cloud_first_cloud_placement_ratio = merge_lists_by_average(*[cloud_first_cloud_placement_ratio[it] for it in range(box_count)])
    #
    # smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])
    # smallest_edge_first_edge_placement_ratio = merge_lists_by_average(
    #     *[smallest_edge_first_edge_placement_ratio[it] for it in range(box_count)])
    # smallest_edge_first_cloud_placement_ratio = merge_lists_by_average(
    #     *[smallest_edge_first_cloud_placement_ratio[it] for it in range(box_count)])
    #
    # biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])
    # biggest_edge_first_edge_placement_ratio = merge_lists_by_average(
    #     *[biggest_edge_first_edge_placement_ratio[it] for it in range(box_count)])
    # biggest_edge_first_cloud_placement_ratio = merge_lists_by_average(
    #     *[biggest_edge_first_cloud_placement_ratio[it] for it in range(box_count)])

    ecmus_qos_aware_timestamps = merge_lists_by_average(
        *[ecmus_qos_aware_timestamps[it] for it in range(box_count)])
    ecmus_qos_aware_edge_placement_ratio = merge_lists_by_average(
        *[ecmus_qos_aware_edge_placement_ratio[it] for it in range(box_count)])
    ecmus_qos_aware_cloud_placement_ratio = merge_lists_by_average(
        *[ecmus_qos_aware_cloud_placement_ratio[it] for it in range(box_count)])

    # ecmus_all_half_timestamps = merge_lists_by_average(
    #     *[ecmus_all_half_timestamps[it] for it in range(box_count)])
    # ecmus_all_half_edge_placement_ratio = merge_lists_by_average(
    #     *[ecmus_all_half_edge_placement_ratio[it] for it in range(box_count)])
    # ecmus_all_half_cloud_placement_ratio = merge_lists_by_average(
    #     *[ecmus_all_half_cloud_placement_ratio[it] for it in range(box_count)])
    #
    # ecmus_only_d_timestamps = merge_lists_by_average(
    #     *[ecmus_only_d_timestamps[it] for it in range(box_count)])
    # ecmus_only_d_edge_placement_ratio = merge_lists_by_average(
    #     *[ecmus_only_d_edge_placement_ratio[it] for it in range(box_count)])
    # ecmus_only_d_cloud_placement_ratio = merge_lists_by_average(
    #     *[ecmus_only_d_cloud_placement_ratio[it] for it in range(box_count)])
    #
    # ecmus_c_gt_a_timestamps = merge_lists_by_average(
    #     *[ecmus_c_gt_a_timestamps[it] for it in range(box_count)])
    # ecmus_c_gt_a_edge_placement_ratio = merge_lists_by_average(
    #     *[ecmus_c_gt_a_edge_placement_ratio[it] for it in range(box_count)])
    # ecmus_c_gt_a_cloud_placement_ratio = merge_lists_by_average(
    #     *[ecmus_c_gt_a_cloud_placement_ratio[it] for it in range(box_count)])

    ecmus_no_cloud_offload_timestamps = merge_lists_by_average(
        *[ecmus_no_cloud_offload_timestamps[it] for it in range(box_count)])
    ecmus_no_cloud_offload_edge_placement_ratio = merge_lists_by_average(
        *[ecmus_no_cloud_offload_edge_placement_ratio[it] for it in range(box_count)])
    ecmus_no_cloud_offload_cloud_placement_ratio = merge_lists_by_average(
        *[ecmus_no_cloud_offload_cloud_placement_ratio[it] for it in range(box_count)])

    ecmus_no_edge_migration_timestamps = merge_lists_by_average(
        *[ecmus_no_edge_migration_timestamps[it] for it in range(box_count)])
    ecmus_no_edge_migration_edge_placement_ratio = merge_lists_by_average(
        *[ecmus_no_edge_migration_edge_placement_ratio[it] for it in range(box_count)])
    ecmus_no_edge_migration_cloud_placement_ratio = merge_lists_by_average(
        *[ecmus_no_edge_migration_cloud_placement_ratio[it] for it in range(box_count)])

    ecmus_mid_migration_timestamps = merge_lists_by_average(
        *[ecmus_mid_migration_timestamps[it] for it in range(box_count)])
    ecmus_mid_migration_edge_placement_ratio = merge_lists_by_average(
        *[ecmus_mid_migration_edge_placement_ratio[it] for it in range(box_count)])
    ecmus_mid_migration_cloud_placement_ratio = merge_lists_by_average(
        *[ecmus_mid_migration_cloud_placement_ratio[it] for it in range(box_count)])

    fig, (ax_cloud, ax_edge) = plt.subplots(2, 1)
    fig.set_size_inches(10.5, 7.5)
    marker_interval = 2

    ax_edge.grid(True, axis='y')
    # ax_edge.plot(ecmus_timestamps, ecmus_edge_placement_ratio, label = "KubeDSM", marker='o', markevery=marker_interval)
    # ax_edge.plot(kube_schedule_timestamps, kube_schedule_edge_placement_ratio, label = "Kube", marker='s', markevery=marker_interval)
    ax_edge.plot(ecmus_no_migration_timestamps, ecmus_no_migration_edge_placement_ratio, label = "KubeDSMNoMigration", marker='D', markevery=marker_interval)
    # ax_edge.plot(random_timestamps, random_edge_placement_ratio, label = "Random", marker='^', markevery=marker_interval)
    # ax_edge.plot(cloud_first_timestamps, cloud_first_edge_placement_ratio, label = "CloudFirst", marker='v', markevery=marker_interval)
    # ax_edge.plot(smallest_edge_first_timestamps, smallest_edge_first_edge_placement_ratio, label = "SmallestEdgeFirst", marker='x', markevery=marker_interval)
    # ax_edge.plot(biggest_edge_first_timestamps, biggest_edge_first_edge_placement_ratio, label = "BiggestEdgeFirst", marker='+', markevery=marker_interval)
    ax_edge.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_edge_placement_ratio, label = "KubeDSMQOSAware", marker="*", markevery=marker_interval)
    # ax_edge.plot(ecmus_all_half_timestamps, ecmus_all_half_edge_placement_ratio, label = "KubeDSMAllHalf", marker="+", markevery=marker_interval)
    # ax_edge.plot(ecmus_only_d_timestamps, ecmus_only_d_edge_placement_ratio, label = "KubeDSMOnlyD", marker="x", markevery=marker_interval)
    # ax_edge.plot(ecmus_c_gt_a_timestamps, ecmus_c_gt_a_edge_placement_ratio, label = "KubeDSMCgtA", marker="v", markevery=marker_interval)
    ax_edge.plot(ecmus_no_cloud_offload_timestamps, ecmus_no_cloud_offload_edge_placement_ratio, label = "KubeDSMNoCloudOffload", marker="v", markevery=marker_interval)
    ax_edge.plot(ecmus_no_edge_migration_timestamps, ecmus_no_edge_migration_edge_placement_ratio, label = "KubeDSMNoEdgeMigration", marker="x", markevery=marker_interval)
    ax_edge.plot(ecmus_mid_migration_timestamps, ecmus_mid_migration_edge_placement_ratio, label = "KubeDSMMidMigration", marker="s", markevery=marker_interval)

    ax_edge.set_xlabel("time (s)")
    ax_edge.set_ylabel("placement ratio")
    ax_edge.set_ylim(0, 1.10)
    ax_edge.set_yticks(list(map(lambda x: x / 100.0, range(0, 110, 10))))
    ax_edge.set_title("edge pod placement ratio")
    ax_edge.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=12, frameon=True, ncol=3)

    ax_cloud.grid(True, axis='y')
    # ax_cloud.plot(ecmus_timestamps, ecmus_cloud_placement_ratio, label = "KubeDSM", marker='o', markevery=marker_interval)
    # ax_cloud.plot(kube_schedule_timestamps, kube_schedule_cloud_placement_ratio, label = "Kube", marker='s', markevery=marker_interval)
    ax_cloud.plot(ecmus_no_migration_timestamps, ecmus_no_migration_cloud_placement_ratio, label = "KubeDSMNoMigration", marker='D', markevery=marker_interval)
    # ax_cloud.plot(random_timestamps, random_cloud_placement_ratio, label = "Random", marker='^', markevery=marker_interval)
    # ax_cloud.plot(cloud_first_timestamps, cloud_first_cloud_placement_ratio, label = "CloudFirst", marker='v', markevery=marker_interval)
    # ax_cloud.plot(smallest_edge_first_timestamps, smallest_edge_first_cloud_placement_ratio, label = "SmallestEdgeFirst", marker='x', markevery=marker_interval)
    # ax_cloud.plot(biggest_edge_first_timestamps, biggest_edge_first_cloud_placement_ratio, label = "BiggestEdgeFirst", marker='+', markevery=marker_interval)
    ax_cloud.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_cloud_placement_ratio, label = "KubeDSMQOSAware", marker="*", markevery=marker_interval)
    # ax_cloud.plot(ecmus_all_half_timestamps, ecmus_all_half_cloud_placement_ratio, label = "KubeDSMAllHalf", marker="+", markevery=marker_interval)
    # ax_cloud.plot(ecmus_only_d_timestamps, ecmus_only_d_cloud_placement_ratio, label = "KubeDSMOnlyD", marker="x", markevery=marker_interval)
    # ax_cloud.plot(ecmus_c_gt_a_timestamps, ecmus_c_gt_a_cloud_placement_ratio, label = "KubeDSMCgtA", marker="v", markevery=marker_interval)
    ax_cloud.plot(ecmus_no_cloud_offload_timestamps, ecmus_no_cloud_offload_cloud_placement_ratio, label = "KubeDSMNoCloudOffload", marker="v", markevery=marker_interval)
    ax_cloud.plot(ecmus_no_edge_migration_timestamps, ecmus_no_edge_migration_cloud_placement_ratio, label = "KubeDSMNoEdgeMigration", marker="x", markevery=marker_interval)
    ax_cloud.plot(ecmus_mid_migration_timestamps, ecmus_mid_migration_cloud_placement_ratio, label = "KubeDSMMidMigration", marker="s", markevery=marker_interval)

    ax_cloud.set_xlabel("time (s)")
    ax_cloud.set_ylabel("placement ratio")
    ax_cloud.set_ylim(0, 1.10)
    ax_cloud.set_yticks(list(map(lambda x: x / 100.0, range(0, 110, 10))))
    ax_cloud.set_title("cloud pod placement ratio")

    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")

@register_extractor
def average_f_linechart(config: Config, _: str, histories: List[History], save_path: str) -> None:
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

    ecmus_all_half_pod_count_list = []
    ecmus_all_half_timestamps_list = []

    ecmus_only_d_pod_count_list = []
    ecmus_only_d_timestamps_list = []

    ecmus_c_gt_a_pod_count_list = []
    ecmus_c_gt_a_timestamps_list = []

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

        ecmus_all_half_pod_count = {it: [] for it in range(box_count)}
        ecmus_all_half_timestamps = {it: [] for it in range(box_count)}

        ecmus_only_d_pod_count = {it: [] for it in range(box_count)}
        ecmus_only_d_timestamps = {it: [] for it in range(box_count)}

        ecmus_c_gt_a_pod_count = {it: [] for it in range(box_count)}
        ecmus_c_gt_a_timestamps = {it: [] for it in range(box_count)}

        for id, history in enumerate(histories):
            # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
            index = id % INDEX_COUNT
            box_id = id // INDEX_COUNT
            for cycle in history.cycles:
                pod_count = calculate_pod_count_for_deployment(cycle, deployment)
                edge_pod_count = calculate_edge_pod_count_for_deployment(cycle, deployment)
                edge_ratio = edge_pod_count / pod_count

                if index == ECMUS_INDEX:
                    ecmus_pod_count[box_id].append(edge_ratio)
                    ecmus_timestamps[box_id].append(cycle.timestamp)

                if index == KUBE_SCHEDULE_INDEX:
                    kube_pod_count[box_id].append(edge_ratio)
                    kube_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_NO_MIGRATION_INDEX:
                    ecmus_no_migration_pod_count[box_id].append(edge_ratio)
                    ecmus_no_migration_timestamps[box_id].append(cycle.timestamp)

                if index == RANDOM_INDEX:
                    random_pod_count[box_id].append(edge_ratio)
                    random_timestamps[box_id].append(cycle.timestamp)

                if index == CLOUD_FIRST_INDEX:
                    cloud_first_pod_count[box_id].append(edge_ratio)
                    cloud_first_timestamps[box_id].append(cycle.timestamp)

                if index == SMALLEST_EDGE_FIRST_INDEX:
                    smallest_edge_first_pod_count[box_id].append(edge_ratio)
                    smallest_edge_first_timestamps[box_id].append(cycle.timestamp)

                if index == BIGGEST_EDGE_FIRST_INDEX:
                    biggest_edge_first_pod_count[box_id].append(edge_ratio)
                    biggest_edge_first_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_QOS_AWARE_INDEX:
                    ecmus_qos_aware_pod_count[box_id].append(edge_ratio)
                    ecmus_qos_aware_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_ALL_HALF_INDEX:
                    ecmus_all_half_pod_count[box_id].append(edge_ratio)
                    ecmus_all_half_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_ONLY_D_INDEX:
                    ecmus_only_d_pod_count[box_id].append(edge_ratio)
                    ecmus_only_d_timestamps[box_id].append(cycle.timestamp)

                if index == ECMUS_C_GT_A_INDEX:
                    ecmus_c_gt_a_pod_count[box_id].append(edge_ratio)
                    ecmus_c_gt_a_timestamps[box_id].append(cycle.timestamp)

        # ecmus_pod_count = merge_lists_by_average(*[ecmus_pod_count[it] for it in range(box_count)])
        # ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps[it] for it in range(box_count)])
        #
        # kube_pod_count = merge_lists_by_average(*[kube_pod_count[it] for it in range(box_count)])
        # kube_timestamps = merge_lists_by_average(*[kube_timestamps[it] for it in range(box_count)])
        #
        # ecmus_no_migration_pod_count = merge_lists_by_average(*[ecmus_no_migration_pod_count[it] for it in range(box_count)])
        # ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps[it] for it in range(box_count)])
        #
        # cloud_first_pod_count = merge_lists_by_average(*[cloud_first_pod_count[it] for it in range(box_count)])
        # cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps[it] for it in range(box_count)])
        #
        # random_pod_count = merge_lists_by_average(*[random_pod_count[it] for it in range(box_count)])
        # random_timestamps = merge_lists_by_average(*[random_timestamps[it] for it in range(box_count)])
        #
        # smallest_edge_first_pod_count = merge_lists_by_average(*[smallest_edge_first_pod_count[it] for it in range(box_count)])
        # smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps[it] for it in range(box_count)])
        #
        # biggest_edge_first_pod_count = merge_lists_by_average(*[biggest_edge_first_pod_count[it] for it in range(box_count)])
        # biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps[it] for it in range(box_count)])

        ecmus_qos_aware_pod_count = merge_lists_by_average(*[ecmus_qos_aware_pod_count[it] for it in range(box_count)])
        ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps[it] for it in range(box_count)])

        ecmus_all_half_pod_count = merge_lists_by_average(*[ecmus_all_half_pod_count[it] for it in range(box_count)])
        ecmus_all_half_timestamps = merge_lists_by_average(*[ecmus_all_half_timestamps[it] for it in range(box_count)])

        ecmus_only_d_pod_count = merge_lists_by_average(*[ecmus_only_d_pod_count[it] for it in range(box_count)])
        ecmus_only_d_timestamps = merge_lists_by_average(*[ecmus_only_d_timestamps[it] for it in range(box_count)])

        ecmus_c_gt_a_pod_count = merge_lists_by_average(*[ecmus_c_gt_a_pod_count[it] for it in range(box_count)])
        ecmus_c_gt_a_timestamps = merge_lists_by_average(*[ecmus_c_gt_a_timestamps[it] for it in range(box_count)])

        # ecmus_pod_count_list.append(ecmus_pod_count)
        # ecmus_timestamps_list.append(ecmus_timestamps)
        #
        # kube_pod_count_list.append(kube_pod_count)
        # kube_timestamps_list.append(kube_timestamps)
        #
        # ecmus_no_migration_pod_count_list.append(ecmus_no_migration_pod_count)
        # ecmus_no_migration_timestamps_list.append(ecmus_no_migration_timestamps)
        #
        # cloud_first_pod_count_list.append(cloud_first_pod_count)
        # cloud_first_timestamps_list.append(cloud_first_timestamps)
        #
        # random_pod_count_list.append(random_pod_count)
        # random_timestamps_list.append(random_timestamps)
        #
        # smallest_edge_first_pod_count_list.append(smallest_edge_first_pod_count)
        # smallest_edge_first_timestamps_list.append(smallest_edge_first_timestamps)
        #
        # biggest_edge_first_pod_count_list.append(biggest_edge_first_pod_count)
        # biggest_edge_first_timestamps_list.append(biggest_edge_first_timestamps)

        ecmus_qos_aware_pod_count_list.append(ecmus_qos_aware_pod_count)
        ecmus_qos_aware_timestamps_list.append(ecmus_qos_aware_timestamps)

        ecmus_all_half_pod_count_list.append(ecmus_all_half_pod_count)
        ecmus_all_half_timestamps_list.append(ecmus_all_half_timestamps)

        ecmus_only_d_pod_count_list.append(ecmus_only_d_pod_count)
        ecmus_only_d_timestamps_list.append(ecmus_only_d_timestamps)

        ecmus_c_gt_a_pod_count_list.append(ecmus_c_gt_a_pod_count)
        ecmus_c_gt_a_timestamps_list.append(ecmus_c_gt_a_timestamps)

        fig, ax = plt.subplots()
        plt.grid(True, axis='y')
        fig.set_size_inches(10.5, 7.5)
        marker_interval = 2
        # ax.plot(kube_timestamps, kube_pod_count, label = "Kube", marker='o', markevery=marker_interval)
        # ax.plot(ecmus_timestamps, ecmus_pod_count, label = "KubeDSM", marker='s', markevery=marker_interval)
        # ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_pod_count, label = "KubeDSMNoMigration", marker='D', markevery=marker_interval)
        # ax.plot(random_timestamps, random_pod_count, label = "Random", marker='^', markevery=marker_interval)
        # ax.plot(cloud_first_timestamps, cloud_first_pod_count, label = "CloudFirst", marker='v', markevery=marker_interval)
        # ax.plot(biggest_edge_first_timestamps, biggest_edge_first_pod_count, label = "BiggestEdgeFirst", marker='x', markevery=marker_interval)
        # ax.plot(smallest_edge_first_timestamps, smallest_edge_first_pod_count, label = "SmallestEdgeFirst", marker='+', markevery=marker_interval)
        ax.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_pod_count, label = "KubeDSMQOSAware", marker="*", markevery=marker_interval)
        ax.plot(ecmus_all_half_timestamps, ecmus_all_half_pod_count, label = "KubeDSMAllHalf", marker="+", markevery=marker_interval)
        ax.plot(ecmus_only_d_timestamps, ecmus_only_d_pod_count, label = "KubeDSMOnlyD", marker="x", markevery=marker_interval)
        ax.plot(ecmus_c_gt_a_timestamps, ecmus_c_gt_a_pod_count, label = "KubeDSMCgtA", marker="v", markevery=marker_interval)
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel("time(s)")
        plt.ylabel("edge/all pod ratio")
        plt.title(f"edge/all pod ratio - workload: {deployment.name}")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=12, frameon=True, ncol=3)
        plt.tight_layout()
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")

    deployment_count = len(config.deployments)

    # ecmus_pod_count = merge_lists_by_sum(*[ecmus_pod_count_list[it] for it in range(deployment_count)])
    # ecmus_timestamps = merge_lists_by_average(*[ecmus_timestamps_list[it] for it in range(deployment_count)])
    #
    # kube_pod_count = merge_lists_by_average(*[kube_pod_count_list[it] for it in range(deployment_count)])
    # kube_timestamps = merge_lists_by_average(*[kube_timestamps_list[it] for it in range(deployment_count)])
    #
    # ecmus_no_migration_pod_count = merge_lists_by_sum(*[ecmus_no_migration_pod_count_list[it] for it in range(deployment_count)])
    # ecmus_no_migration_timestamps = merge_lists_by_average(*[ecmus_no_migration_timestamps_list[it] for it in range(deployment_count)])
    #
    # random_pod_count = merge_lists_by_average(*[random_pod_count_list[it] for it in range(deployment_count)])
    # random_timestamps = merge_lists_by_average(*[random_timestamps_list[it] for it in range(deployment_count)])
    #
    # cloud_first_pod_count = merge_lists_by_average(*[cloud_first_pod_count_list[it] for it in range(deployment_count)])
    # cloud_first_timestamps = merge_lists_by_average(*[cloud_first_timestamps_list[it] for it in range(deployment_count)])
    #
    # smallest_edge_first_pod_count = merge_lists_by_average(*[smallest_edge_first_pod_count_list[it] for it in range(deployment_count)])
    # smallest_edge_first_timestamps = merge_lists_by_average(*[smallest_edge_first_timestamps_list[it] for it in range(deployment_count)])
    #
    # biggest_edge_first_pod_count = merge_lists_by_average(*[biggest_edge_first_pod_count_list[it] for it in range(deployment_count)])
    # biggest_edge_first_timestamps = merge_lists_by_average(*[biggest_edge_first_timestamps_list[it] for it in range(deployment_count)])

    ecmus_qos_aware_pod_count = merge_lists_by_average(*[ecmus_qos_aware_pod_count_list[it] for it in range(deployment_count)])
    ecmus_qos_aware_timestamps = merge_lists_by_average(*[ecmus_qos_aware_timestamps_list[it] for it in range(deployment_count)])

    ecmus_all_half_pod_count = merge_lists_by_average(*[ecmus_all_half_pod_count_list[it] for it in range(deployment_count)])
    ecmus_all_half_timestamps = merge_lists_by_average(*[ecmus_all_half_timestamps_list[it] for it in range(deployment_count)])

    ecmus_only_d_pod_count = merge_lists_by_average(*[ecmus_only_d_pod_count_list[it] for it in range(deployment_count)])
    ecmus_only_d_timestamps = merge_lists_by_average(*[ecmus_only_d_timestamps_list[it] for it in range(deployment_count)])

    ecmus_c_gt_a_pod_count = merge_lists_by_average(*[ecmus_c_gt_a_pod_count_list[it] for it in range(deployment_count)])
    ecmus_c_gt_a_timestamps = merge_lists_by_average(*[ecmus_c_gt_a_timestamps_list[it] for it in range(deployment_count)])

    fig, ax = plt.subplots()
    plt.grid(True, axis='y')
    fig.set_size_inches(10.5, 7.5)
    markevery=marker_interval
    # ax.plot(kube_timestamps, kube_pod_count, label = "Kube", marker='o', markevery=marker_interval)
    # ax.plot(ecmus_timestamps, ecmus_pod_count, label = "KubeDSM", marker='s', markevery=marker_interval)
    # ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_pod_count, label = "KubeDSMNoMigration", marker='D', markevery=marker_interval)
    # ax.plot(random_timestamps, random_pod_count, label = "Random", marker='^', markevery=marker_interval)
    # ax.plot(cloud_first_timestamps, cloud_first_pod_count, label = "CloudFirst", marker='v', markevery=marker_interval)
    # ax.plot(biggest_edge_first_timestamps, biggest_edge_first_pod_count, label = "BiggestEdgeFirst", marker='x', markevery=marker_interval)
    # ax.plot(smallest_edge_first_timestamps, smallest_edge_first_pod_count, label = "SmallestEdgeFirst", marker='+', markevery=marker_interval)
    ax.plot(ecmus_qos_aware_timestamps, ecmus_qos_aware_pod_count, label = "KubeDSMQOSAware", marker="*", markevery=marker_interval)
    ax.plot(ecmus_all_half_timestamps, ecmus_all_half_pod_count, label = "KubeDSMAllHalf", marker="+", markevery=marker_interval)
    ax.plot(ecmus_only_d_timestamps, ecmus_only_d_pod_count, label = "KubeDSMOnlyD", marker="x", markevery=marker_interval)
    ax.plot(ecmus_c_gt_a_timestamps, ecmus_c_gt_a_pod_count, label = "KubeDSMCgtA", marker="v", markevery=marker_interval)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlabel("time(s)")
    plt.ylabel("edge/all pod ratio")
    plt.title(f"edge/all pod ratio - workload total")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=12, frameon=True, ncol=3)
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(f"{save_path}/all.png")

# TODO: ecmus self compare works till here

@register_extractor
def average_f_metadata(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    data = {
        "Kube": [],
        # "KubeDSM": [],
        # "KubeDSMNoMigration": [],
        "Random": [],
        "CloudFirst": [],
        "BiggestEdgeFirst": [],
        "SmallestEdgeFirst": [],
        "KubeDSMQOSAware": [],
    }

    ensure_directory(save_path)
    metadata_filepath = os.path.join(save_path, METADATA_FILENAME)

    metadata = {scheduler: {} for scheduler in data.keys()}

    if not os.path.exists(metadata_filepath):
        with open(metadata_filepath, "w") as file:
            json.dump(metadata, file)

    else:
        with open(metadata_filepath, "r") as file:
            metadata = json.load(file)

    for deployment in config.deployments.values():
        for id, history in enumerate(histories):
            # IMPORTANT NOTICE: histories have to be in order of INDICES for this to work
            index = id % INDEX_COUNT
            for cycle in history.cycles:
                pod_count = calculate_pod_count_for_deployment(cycle, deployment)
                edge_pod_count = calculate_edge_pod_count_for_deployment(cycle, deployment)
                edge_ratio = edge_pod_count / pod_count

                if index == ECMUS_INDEX:
                    data["KubeDSM"].append(edge_ratio)

                if index == KUBE_SCHEDULE_INDEX:
                    data["Kube"].append(edge_ratio)

                if index == ECMUS_NO_MIGRATION_INDEX:
                    data["KubeDSMNoMigration"].append(edge_ratio)

                if index == RANDOM_INDEX:
                    data["Random"].append(edge_ratio)

                if index == CLOUD_FIRST_INDEX:
                    data["CloudFirst"].append(edge_ratio)

                if index == SMALLEST_EDGE_FIRST_INDEX:
                    data["SmallestEdgeFirst"].append(edge_ratio)

                if index == BIGGEST_EDGE_FIRST_INDEX:
                    data["BiggestEdgeFirst"].append(edge_ratio)

                if index == ECMUS_QOS_AWARE_INDEX:
                    data["KubeDSMQOSAware"].append(edge_ratio)

    for scheduler in metadata.keys():
        metadata[scheduler][scenario_name] = sum(data[scheduler]) / len(data[scheduler])

    with open(metadata_filepath, "w") as file:
        json.dump(metadata, file)


@register_extractor
def average_f_table(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    metadata_filepath = os.path.join(save_path, METADATA_FILENAME)

    if not os.path.exists(metadata_filepath):
        return

    with open(metadata_filepath, "r") as file:
        metadata = json.load(file)

    ensure_directory(save_path)

    scheduler_data = {}

    for scheduler, scenarios in metadata.items():
        scenario_data = {}
        for scenario, values in scenarios.items():
            scenario_data[scenario] = values
        scheduler_data[scheduler] = scenario_data

    ratio_df = pd.DataFrame.from_dict(scheduler_data, orient='index')

    ratio_df.fillna('-', inplace=True)

    fig, ax = plt.subplots()
    ax.axis('off')

    table = ax.table(cellText=ratio_df.values,
                     colLabels=ratio_df.columns,
                     rowLabels=ratio_df.index,
                     cellLoc='center',
                     loc='center')

    plt.title('edge/all pod ratio by Scheduler and Scenario')

    ensure_directory(save_path)
    plt.savefig(save_path + "/edge_ratio_table.png", bbox_inches='tight', dpi=300)
