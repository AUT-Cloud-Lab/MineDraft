import math
import os.path
from math import ceil
from typing import List, Dict

import matplotlib.pyplot as plt

from extractors.decorator import register_extractor
from historical.common import Deployment
from historical.config import Config
from historical.data import History, Cycle, Migration
from historical.utils import calculate_edge_usage_sum, \
    calculate_cluster_usage_sum, calculate_resource_usage_for_node, calculate_placement_for_deployment
from historical.utils import get_nodes_of_a_deployment, get_edge_placed_pods

CLOUD_RESPONSE_TIME = 300
EDGE_RESPONSE_TIME = 50

ECMUS_INDEX = 0
KUBE_SCHEDULE_INDEX = 1
ECMUS_NO_MIGRATION_INDEX = 2
RANDOM_INDEX = 3


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
                    key=lambda node: node.name,
                )
                target_nodes = sorted(
                    get_nodes_of_a_deployment(next_cycle.pod_placement, deployment),
                    key=lambda node: node.name,
                )

                print(list(map(lambda node: node.name, source_nodes)), file=output_file)
                print(list(map(lambda node: node.name, target_nodes)), file=output_file)

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
                print(f"here with {len(real_sources)}", file=output_file)
                print(f"{start}, {end}", file=output_file)
                for i in range(len(real_sources)):
                    migrations.append(
                        Migration(
                            deployment=deployment,
                            source=real_sources[i],
                            target=real_targets[i],
                            start=start,
                            end=end,
                        )
                    )

        print(f"Number of migrations for {deployment.name}: {len(migrations)}", file=output_file)
        for migration in migrations:
            print(migration, file=output_file)


@register_extractor
def check_equality(config: Config, histories: List[History], save_path: str) -> None:
    output_file = open(save_path, 'w')
    """
    Check if the histories are equal.
    """
    number_of_differences: Dict[Deployment, int] = {
        deployment: 0 for _, deployment in config.deployments.items()
    }
    print(len(histories[0].cycles), file=output_file)
    print(len(histories[1].cycles), file=output_file)

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
        print(f"{deployment}'s number of differences are {number_of_differences}!", file=output_file)


@register_extractor
def average_latency_linechart(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    for deployment in config.deployments.values():
        kube_latencies = []
        kube_timestamps = []

        ecmus_latencies = []
        ecmus_timestamps = []

        ecmus_no_migration_latencies = []
        ecmus_no_migration_timestamps = []

        random_latencies = []
        random_timestamps = []
        for index, history in enumerate(histories):
            for cycle in history.cycles:
                cloud_pods_count, edge_pods_count = calculate_placement_for_deployment(cycle, deployment)
                all_pods_count = cloud_pods_count + edge_pods_count
                # portion = calculate_request_portion_for_deployment(config, cycle, deployment)
                latency = (
                                  cloud_pods_count * CLOUD_RESPONSE_TIME + edge_pods_count * EDGE_RESPONSE_TIME) / all_pods_count

                if index == ECMUS_INDEX:
                    ecmus_latencies.append(latency)
                    ecmus_timestamps.append(cycle.timestamp)

                if index == KUBE_SCHEDULE_INDEX:
                    kube_latencies.append(latency)
                    kube_timestamps.append(cycle.timestamp)

                if index == ECMUS_NO_MIGRATION_INDEX:
                    ecmus_no_migration_latencies.append(latency)
                    ecmus_no_migration_timestamps.append(cycle.timestamp)

                if index == RANDOM_INDEX:
                    random_latencies.append(latency)
                    random_timestamps.append(cycle.timestamp)
        # data = {
        #     "ecmus": ecmus_latencies,
        #     "kube-schedule": kube_latencies
        # }

        fig, ax = plt.subplots()
        ax.plot(kube_timestamps, kube_latencies, label="kube")
        ax.plot(ecmus_timestamps, ecmus_latencies, label="ecmus")
        ax.plot(ecmus_no_migration_timestamps, ecmus_no_migration_latencies, label="ecmus-no-migration")
        plt.xlabel("time(s)")
        plt.ylabel("average latency(ms)")
        plt.title(f"average latency - workload: {deployment.name}")
        plt.legend()
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")


def ensure_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


@register_extractor
def average_latency_boxplot(config: Config, scenario_name: str, histories: List[History], save_path: str) -> None:
    for deployment in config.deployments.values():
        kube_latencies = []
        ecmus_latencies = []
        ecmus_no_migration_latencies = []
        random_latencies = []
        for index, history in enumerate(histories):
            for cycle in history.cycles:
                cloud_pods_count, edge_pods_count = calculate_placement_for_deployment(cycle, deployment)
                all_pods_count = cloud_pods_count + edge_pods_count
                latency = (
                                  cloud_pods_count * CLOUD_RESPONSE_TIME + edge_pods_count * EDGE_RESPONSE_TIME) / all_pods_count

                if index == ECMUS_INDEX:
                    ecmus_latencies.append(latency)

                if index == KUBE_SCHEDULE_INDEX:
                    kube_latencies.append(latency)

                if index == ECMUS_NO_MIGRATION_INDEX:
                    ecmus_no_migration_latencies.append(latency)

                if index == RANDOM_INDEX:
                    random_latencies.append(latency)

        data = {
            "ecmus": ecmus_latencies,
            "kube-schedule": kube_latencies,
            "ecmus_no_migration": ecmus_no_migration_latencies,
            "random_scheduler": random_latencies,
        }

        fig, ax = plt.subplots()
        plt.title(f"average latency - workload: {deployment.name.upper()}")
        plt.xlabel("time(s)")
        plt.ylabel("average latency(ms)")
        ax.boxplot(data.values(), showfliers=False)
        ax.set_xticklabels(data.keys())
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")


@register_extractor
def edge_utilization_linechart(config: Config, _: str, histories: List[History], save_path: str) -> None:
    ecmus_utilization = []
    ecmus_timestamps = []

    kube_schedule_utilization = []
    kube_schedule_timestamps = []

    ecmus_no_migration_utilization = []
    ecmus_no_migration_timestamps = []

    random_utilization = []
    random_timestamps = []
    for index, history in enumerate(histories):
        for cycle in history.cycles:
            edge_pods = get_edge_placed_pods(cycle)

            edge_usage_sum_cpu, edge_usage_sum_memory = calculate_edge_usage_sum(config, edge_pods)
            cluster_usage_sum_cpu, cluster_usage_sum_memory = calculate_cluster_usage_sum(cycle)

            utilization = math.sqrt(
                (edge_usage_sum_cpu / cluster_usage_sum_cpu) * (edge_usage_sum_memory / cluster_usage_sum_memory)
            )

            if index == ECMUS_INDEX:
                ecmus_timestamps.append(cycle.timestamp)
                ecmus_utilization.append(utilization)

            if index == KUBE_SCHEDULE_INDEX:
                kube_schedule_timestamps.append(cycle.timestamp)
                kube_schedule_utilization.append(utilization)

            if index == ECMUS_NO_MIGRATION_INDEX:
                ecmus_no_migration_timestamps.append(cycle.timestamp)
                ecmus_no_migration_utilization.append(utilization)

            if index == RANDOM_INDEX:
                random_timestamps.append(cycle.timestamp)
                random_utilization.append(utilization)

    plt.plot(ecmus_timestamps, ecmus_utilization, label="ecmus")
    plt.plot(kube_schedule_timestamps, kube_schedule_utilization, label="kube-schedule")
    plt.plot(ecmus_no_migration_timestamps, ecmus_no_migration_utilization, label="ecmus-no-migration")
    plt.plot(random_timestamps, random_utilization, label="random")

    plt.xlabel("time (s)")
    plt.ylabel("edge utilization")
    plt.ylim(0, 1.10)
    plt.title("edge utilization - per algorithm")
    plt.legend()
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")


@register_extractor
def placement_ratio_linechart(config: Config, _: str, histories: List[History], save_path: str) -> None:
    ecmus_edge_placement_ratio = []
    ecmus_cloud_placement_ratio = []
    ecmus_timestamps = []

    kube_schedule_edge_placement_ratio = []
    kube_schedule_cloud_placement_ratio = []
    kube_schedule_timestamps = []

    ecmus_no_migration_edge_placement_ratio = []
    ecmus_no_migration_cloud_placement_ratio = []
    ecmus_no_migration_timestamps = []

    random_edge_placement_ratio = []
    random_cloud_placement_ratio = []
    random_timestamps = []

    edge_nodes_count = len([node for node in config.nodes.values() if node.is_on_edge])
    cloud_nodes_count = len([node for node in config.nodes.values() if not node.is_on_edge])
    for index, history in enumerate(histories):
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
                ecmus_timestamps.append(cycle.timestamp)
                ecmus_edge_placement_ratio.append(fragmentation_edge)
                ecmus_cloud_placement_ratio.append(fragmentation_cloud)

            if index == KUBE_SCHEDULE_INDEX:
                kube_schedule_timestamps.append(cycle.timestamp)
                kube_schedule_edge_placement_ratio.append(fragmentation_edge)
                kube_schedule_cloud_placement_ratio.append(fragmentation_cloud)

            if index == ECMUS_NO_MIGRATION_INDEX:
                ecmus_no_migration_timestamps.append(cycle.timestamp)
                ecmus_no_migration_edge_placement_ratio.append(fragmentation_edge)
                ecmus_no_migration_cloud_placement_ratio.append(fragmentation_cloud)

            if index == RANDOM_INDEX:
                random_timestamps.append(cycle.timestamp)
                random_edge_placement_ratio.append(fragmentation_edge)
                random_cloud_placement_ratio.append(fragmentation_cloud)

    plt.plot(ecmus_timestamps, ecmus_edge_placement_ratio, label="ecmus - edge")
    plt.plot(ecmus_timestamps, ecmus_cloud_placement_ratio, label="ecmus - cloud")

    plt.plot(kube_schedule_timestamps, kube_schedule_edge_placement_ratio, label="kube-schedule - edge")
    plt.plot(kube_schedule_timestamps, kube_schedule_cloud_placement_ratio, label="kube-schedule - cloud")

    plt.plot(ecmus_no_migration_timestamps, ecmus_no_migration_edge_placement_ratio, label="ecmus-no-migration - edge")
    plt.plot(ecmus_no_migration_timestamps, ecmus_no_migration_cloud_placement_ratio,
             label="ecmus-no-migration - cloud")

    plt.plot(random_timestamps, random_edge_placement_ratio, label="random - edge")
    plt.plot(random_timestamps, random_cloud_placement_ratio, label="random - cloud")

    plt.xlabel("time (s)")
    plt.ylabel("placement ratio")
    plt.ylim(0, 1.10)
    plt.title("pod placement ratio")
    plt.legend()
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")
