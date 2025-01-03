import math
from math import ceil
from typing import Callable, Dict, List, Tuple, TypeVar

from extractors.utils import (
    calculate_edge_usage_sum,
    calculate_placement_for_deployment,
    get_edge_placed_pods,
    get_nodes_of_a_deployment,
    merge_lists_by_average,
)
from historical.common import Deployment, Scheduler
from historical.config import Config
from historical.data import Cycle, Migration, ScenarioData

T = TypeVar("T")


def calc_through_time(
    config: Config,
    scenarios: List[ScenarioData],
    on_each_cycle: Callable[[Deployment, Cycle], T],
) -> Tuple[
    Dict[Scheduler, Dict[Deployment, List[T]]],
    Dict[Scheduler, Dict[Deployment, List[float]]],
]:
    assert len(set([scen.name for scen in scenarios])) == 1

    results = {
        sched: {deployment: [] for deployment in config.deployments.values()}
        for sched in config.schedulers
    }
    timestamps = {
        sched: {deployment: [] for deployment in config.deployments.values()}
        for sched in config.schedulers
    }

    for deployment in config.deployments.values():
        for sched in config.schedulers:
            current_results_lists = []
            current_timestamps_lists = []
            for scen in scenarios:
                current_results_lists.append(
                    [
                        on_each_cycle(deployment, cycle)
                        for cycle in scen.scheduler_histories[sched].cycles
                    ]
                )
                current_timestamps_lists.append(
                    [
                        cycle.timestamp
                        for cycle in scen.scheduler_histories[sched].cycles
                    ]
                )

            results[sched][deployment] = merge_lists_by_average(*current_results_lists)
            timestamps[sched][deployment] = merge_lists_by_average(
                *current_timestamps_lists
            )

    return results, timestamps


def calc_average_latency_through_time(
    config: Config, scenarios: List[ScenarioData]
) -> Tuple[
    Dict[Scheduler, Dict[Deployment, List[float]]],
    Dict[Scheduler, Dict[Deployment, List[float]]],
]:
    assert len(set([scen.name for scen in scenarios])) == 1

    def latency_on_each_cycle(deployment: Deployment, cycle: Cycle) -> float:
        cloud_pods_count, edge_pods_count = calculate_placement_for_deployment(
            cycle, deployment
        )
        all_pods_count = cloud_pods_count + edge_pods_count
        return (
            cloud_pods_count * config.CLOUD_RESPONSE_TIME
            + edge_pods_count * config.EDGE_RESPONSE_TIME
        ) / all_pods_count

    return calc_through_time(config, scenarios, latency_on_each_cycle)


def calc_edge_utilization_through_time(
    config: Config, scenarios: List[ScenarioData]
) -> Tuple[
    Dict[Scheduler, List[float]],
    Dict[Scheduler, List[float]],
]:
    assert len(set([scen.name for scen in scenarios])) == 1

    edge_total_cpu, edge_total_memory = config.get_edge_resources()

    def edge_utilization_on_each_cycle(deployment: Deployment, cycle: Cycle) -> float:
        edge_pods = get_edge_placed_pods(cycle)

        edge_usage_sum_cpu, edge_usage_sum_memory = calculate_edge_usage_sum(
            config, edge_pods
        )

        utilization = math.sqrt(
            (edge_usage_sum_cpu / edge_total_cpu)
            * (edge_usage_sum_memory / edge_total_memory)
        )

        return utilization

    res_by_deployment, timestamps_by_deployment = calc_through_time(
        config, scenarios, edge_utilization_on_each_cycle
    )
    any_deployment = next(iter(config.deployments.values()))
    res = {
        sched: res_by_deployment[sched][any_deployment] for sched in config.schedulers
    }
    timestamps = {
        sched: timestamps_by_deployment[sched][any_deployment]
        for sched in config.schedulers
    }

    return res, timestamps


def migration_count(
    config: Config, scenario_data: ScenarioData
) -> Dict[Scheduler, Dict[Deployment, float]]:
    # TODO refactor this to use calc_through_time
    def is_cycle_valid(cycle: Cycle) -> bool:
        """
        Check if number of running pods are equal to the desired number of pods.
        """
        for deployment_name in config.deployments:
            number_of_pods = 0
            for _, pods in cycle.pod_placement.node_pods.items():
                number_of_pods += len(
                    list(filter(lambda pod: pod.name == deployment_name, pods))
                )

            if (
                deployment not in cycle.hpa.deployment_metrics
                or ceil(cycle.hpa.deployment_metrics[deployment]) != number_of_pods
            ):
                return False

        return True

    data = {
        scheduler: {deployment: [] for deployment in config.deployments.values()}
        for scheduler in config.schedulers
    }

    res = {
        scheduler: {deployment: [] for deployment in config.deployments.values()}
        for scheduler in config.schedulers
    }

    for deployment in config.deployments.values():
        for sched, history in scenario_data.scheduler_histories.items():
            migrations: List[Migration] = []
            for start, cycle in enumerate(history.cycles):
                if not is_cycle_valid(cycle):
                    continue

                for _end, next_cycle in enumerate(history.cycles[start + 1 :]):
                    end = start + _end + 1

                    if deployment not in next_cycle.hpa.deployment_metrics or ceil(
                        cycle.hpa.deployment_metrics[deployment]
                    ) != ceil(next_cycle.hpa.deployment_metrics[deployment]):
                        break

                    if not is_cycle_valid(next_cycle):
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

                    real_sources = []
                    real_targets = []

                    it_source = 0
                    it_target = 0
                    while it_source < len(source_nodes) and it_target < len(
                        target_nodes
                    ):
                        if source_nodes[it_source] == target_nodes[it_target]:
                            it_source += 1
                            it_target += 1
                        elif (
                            source_nodes[it_source].name < target_nodes[it_target].name
                        ):
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
                                deployment=deployment,
                                source=real_sources[i],
                                target=real_targets[i],
                                start=start,
                                end=end,
                            )
                        )

            data[sched][deployment].append(len(migrations))

    for scheduler in data.keys():
        for deployment in config.deployments:
            res[scheduler][deployment] = sum(data[scheduler][deployment]) / len(
                data[scheduler][deployment]
            )

    return res
