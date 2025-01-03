import math
from typing import Callable, Dict, List, Tuple, TypeVar

from extractors.utils import (
    calculate_edge_usage_sum,
    calculate_placement_for_deployment,
    calculate_pod_count_for_deployment,
    merge_lists_by_average,
)
from historical.common import Deployment, Scheduler
from historical.config import Config
from historical.data import Cycle, ScenarioData

T = TypeVar("T")


def calc_through_time(
    config: Config,
    scenario: ScenarioData,
    schedulers: List[Scheduler],
    on_each_cycle: Callable[[Deployment, Cycle], T],
) -> Tuple[
    Dict[Scheduler, Dict[Deployment, List[T]]],
    Dict[Scheduler, Dict[Deployment, List[float]]],
]:
    results = {
        sched: {deployment: [] for deployment in config.deployments.values()}
        for sched in schedulers
    }
    timestamps = {
        sched: {deployment: [] for deployment in config.deployments.values()}
        for sched in schedulers
    }

    for deployment in config.deployments.values():
        for sched in schedulers:
            current_results_lists = []
            current_timestamps_lists = []
            for history in scenario.scheduler_histories[sched]:
                current_results_lists.append(
                    [on_each_cycle(deployment, cycle) for cycle in history.cycles]
                )
                current_timestamps_lists.append(
                    [cycle.timestamp for cycle in history.cycles]
                )

            results[sched][deployment] = merge_lists_by_average(*current_results_lists)
            timestamps[sched][deployment] = merge_lists_by_average(
                *current_timestamps_lists
            )

    return results, timestamps


def calc_average_latency_through_time(
    config: Config,
    scenario: ScenarioData,
    schedulers: List[Scheduler],
) -> Tuple[
    Dict[Scheduler, Dict[Deployment, List[float]]],
    Dict[Scheduler, Dict[Deployment, List[float]]],
]:
    def latency_on_each_cycle(deployment: Deployment, cycle: Cycle) -> float:
        cloud_pods_count, edge_pods_count = calculate_placement_for_deployment(
            cycle, deployment
        )
        all_pods_count = cloud_pods_count + edge_pods_count
        return (
            cloud_pods_count * config.CLOUD_RESPONSE_TIME
            + edge_pods_count * config.EDGE_RESPONSE_TIME
        ) / all_pods_count

    return calc_through_time(config, scenario, schedulers, latency_on_each_cycle)


def calc_edge_utilization_through_time(
    config: Config,
    scenario: ScenarioData,
    schedulers: List[Scheduler],
) -> Tuple[
    Dict[Scheduler, List[float]],
    Dict[Scheduler, List[float]],
]:
    edge_total_cpu, edge_total_memory = config.get_edge_resources()

    def edge_utilization_on_each_cycle(deployment: Deployment, cycle: Cycle) -> float:
        edge_usage_sum_cpu, edge_usage_sum_memory = calculate_edge_usage_sum(cycle)

        if edge_total_cpu < edge_usage_sum_cpu + 1e-9:
            print(
                f"WARNING: edge_total_cpu({edge_total_cpu}) < edge_usage_sum_cpu({edge_usage_sum_cpu}) as in cycle({cycle.timestamp}) of scenario({scenario.name})"
            )
            edge_usage_sum_cpu = edge_total_cpu

        if edge_total_memory < edge_usage_sum_memory + 1e-9:
            print(
                f"WARNING: edge_total_memory({edge_total_memory}) < edge_usage_sum_memory({edge_usage_sum_memory}) as in cycle({cycle.timestamp}) of scenario({scenario.name})"
            )
            edge_usage_sum_memory = edge_total_memory

        utilization = math.sqrt(
            (edge_usage_sum_cpu / edge_total_cpu)
            * (edge_usage_sum_memory / edge_total_memory)
        )

        return utilization

    res_by_deployment, timestamps_by_deployment = calc_through_time(
        config, scenario, schedulers, edge_utilization_on_each_cycle
    )
    any_deployment = next(iter(config.deployments.values()))
    res = {sched: res_by_deployment[sched][any_deployment] for sched in schedulers}
    timestamps = {
        sched: timestamps_by_deployment[sched][any_deployment] for sched in schedulers
    }

    return res, timestamps


def calc_pod_count_through_time(
    config: Config,
    scenario: ScenarioData,
    schedulers: List[Scheduler],
) -> Tuple[
    Dict[Scheduler, Dict[Deployment, List[int]]],
    Dict[Scheduler, Dict[Deployment, List[float]]],
]:
    def pod_count_on_each_cycle(deployment: Deployment, cycle: Cycle) -> int:
        return calculate_pod_count_for_deployment(cycle, deployment)

    return calc_through_time(config, scenario, schedulers, pod_count_on_each_cycle)
