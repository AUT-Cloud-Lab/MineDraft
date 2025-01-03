from math import ceil
import statistics
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from extractors.decorator import register_extractor
from extractors.logic import (
    calc_average_latency_through_time,
    calc_edge_utilization_through_time,
    migration_count,
)
from extractors.utils import (
    calculate_pod_count_for_deployment,
    ensure_directory,
    merge_for_each_deployment,
    merge_lists_by_average,
    merge_lists_by_sum,
)
from historical.common import Deployment, Scheduler
from historical.config import Config
from historical.data import Cycle, History, ScenarioData

# TODO make sure these are long enough
COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"]
MARKERS = ["^", "v", "x", "+", "*", "o", "s", "D", "p", "P", "X", "d"]


@register_extractor
def migration_count_boxplot(
    config: Config,
    scenarios: List[ScenarioData],
    schedulers: List[Scheduler],
    save_path: str,
) -> None:

    migration_count_list_for_all_scenarios = {
        scenario.name: [] for scenario in scenarios
    }

    for scenario in scenarios:
        migration_count_list_for_all_scenarios[scenario.name].append(
            migration_count(config, scenario)
        )

    migration_count_for_all_scenarios = {
        scenario.name: merge_for_each_deployment(
            migration_count_list_for_all_scenarios[scenario.name]
        )
        for scenario in scenarios
    }

    num_schedulers = len(schedulers)
    axs: List[plt.Axes] = []
    fig, axs = plt.subplots(
        # TODO move 12, 6 to config
        num_schedulers,
        1,
        figsize=(12, 6 * num_schedulers),
        sharex=True,
    )

    if num_schedulers == 1:
        axs = [axs]

    for ax, sched in zip(axs, schedulers):
        scenarios_name = list(migration_count_for_all_scenarios.keys())
        num_scenarios = len(scenarios_name)

        bar_width = 0.2
        index = np.arange(num_scenarios)

        values = {
            deployment: [
                migration_count_for_all_scenarios[scen][sched][deployment]
                for scen in scenarios_name
            ]
            for deployment in config.deployments.values()
        }

        bar_containers = [
            ax.bar(
                index + (ind - len(config.deployments) / 2 + 0.5) * bar_width,
                values[deployment],
                bar_width,
                label=deployment.name,
                color=COLORS[ind],
            )
            for (ind, deployment) in enumerate(config.deployments.values())
        ]

        ax.set_title(f"Values for Each Scenario - {sched.name}")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Value")

        ax.legend()

        ax.set_xticks(index)
        ax.set_xticklabels(scenarios_name, rotation=90, ha="right")

        for bars in bar_containers:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height}",
                    ha="center",
                    va="bottom",
                )

    plt.savefig(save_path + f"/individual.png")

    sched_scen_to_migrations = {}

    for scen_name, sched_migrations in migration_count_for_all_scenarios.items():
        for sched, migrations in sched_migrations.items():
            sched_scen_to_migrations[(sched, scen_name)] = sum(migrations.values())

    fig, axs = plt.subplots(
        num_schedulers, 1, figsize=(10, 6 * num_schedulers), sharex=True
    )

    if num_schedulers == 1:
        axs = [axs]

    for ax, sched in zip(axs, schedulers):
        scenarios_name = list(migration_count_for_all_scenarios.keys())

        bars = ax.bar(
            scenarios_name,
            [
                sched_scen_to_migrations[(sched, scen_name)]
                for scen_name in scenarios_name
            ],
            color="skyblue",
        )

        ax.set_title(f"Sum of Migrations for Each Scenario - {sched.name}")
        ax.set_xlabel("Scenario")
        ax.set_xticks(scenarios)
        ax.set_xticklabels(scenarios, rotation=90)
        ax.set_ylabel("Sum of Migrations")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height}",
                ha="center",
                va="bottom",
            )

    plt.savefig(save_path + f"/sum.png")


@register_extractor
def pod_count_linechart(
    config: Config,
    scenarios: List[ScenarioData],
    schedulers: List[Scheduler],
    save_path: str,
) -> None:
    # TODO make these kind of extractors a different type of extractor, i.e.:
    # All per cycle extractors can be defined using a simple function that applies to each cycle of the history and then the diagram can be drawn using the results of this function.
    assert len(set([scen.name for scen in scenarios])) == 1

    pod_counts = {
        sched: {deployment: [] for deployment in config.deployments.values()}
        for sched in schedulers
    }
    timestamps = {
        sched: {deployment: [] for deployment in config.deployments.values()}
        for sched in schedulers
    }

    for deployment in config.deployments.values():
        for sched in schedulers:
            current_pod_counts = []
            current_timestamps = []
            for scen in scenarios:
                current_pod_counts.append(
                    [
                        calculate_pod_count_for_deployment(cycle, deployment)
                        for cycle in scen.scheduler_histories[sched].cycles
                    ]
                )
                current_timestamps.append(
                    [
                        cycle.timestamp
                        for cycle in scen.scheduler_histories[sched].cycles
                    ]
                )

            pod_counts[sched][deployment] = merge_lists_by_average(*current_pod_counts)
            timestamps[sched][deployment] = merge_lists_by_average(*current_timestamps)

        fig, ax = plt.subplots()
        plt.grid(True, axis="y")
        fig.set_size_inches(10.5, 7.5)
        marker_interval = 2
        for ind, sched in enumerate(schedulers):
            ax.plot(
                timestamps[sched][deployment],
                pod_counts[sched][deployment],
                label=sched.name,
                marker=MARKERS[ind],
                markevery=marker_interval,
            )
        max_val = (
            int(
                np.max(
                    np.concatenate(
                        [pod_counts[sched][deployment] for sched in schedulers]
                    )
                )
            )
            + 1
        )
        ax.set_ylim(0, max_val)
        ax.set_yticks(range(0, max_val, 1))
        plt.xlabel("time(s)")
        plt.ylabel("pod count")
        plt.title(f"pod count - workload: {deployment.name}")
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fontsize=12,
            frameon=True,
            ncol=3,
        )
        plt.tight_layout()
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")

    all_pod_counts = {
        sched: merge_lists_by_sum(
            [
                pod_counts[sched][deployment]
                for deployment in config.deployments.values()
            ]
        )
        for sched in schedulers
    }
    all_timestamps = {
        sched: merge_lists_by_average(
            [
                timestamps[sched][deployment]
                for deployment in config.deployments.values()
            ]
        )
        for sched in schedulers
    }

    fig, ax = plt.subplots()
    plt.grid(True, axis="y")
    fig.set_size_inches(10.5, 7.5)
    markevery = marker_interval
    for ind, sched in enumerate(schedulers):
        ax.plot(
            all_timestamps[sched],
            all_pod_counts[sched],
            label=sched.name,
            marker=MARKERS[ind],
            markevery=markevery,
        )
    max_val = (
        int(np.max(np.concatenate([all_pod_counts[sched] for sched in schedulers]))) + 1
    )
    ax.set_ylim(0, max_val)
    ax.set_yticks(range(0, max_val, 1))
    plt.xlabel("time(s)")
    plt.ylabel("pod count")
    plt.title(f"pod count - workload total")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=12,
        frameon=True,
        ncol=3,
    )
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(f"{save_path}/all.png")


@register_extractor
def average_latency_linechart(
    config: Config,
    scenarios: List[ScenarioData],
    schedulers: List[Scheduler],
    save_path: str,
) -> None:
    assert len(set([scen.name for scen in scenarios])) == 1

    latencies, timestamps = calc_average_latency_through_time(config, scenarios)

    for deployment in config.deployments.values():
        fig, ax = plt.subplots()
        plt.grid(True, axis="y")
        fig.set_size_inches(10.5, 7.5)
        marker_interval = 2
        for ind, sched in enumerate(schedulers):
            ax.plot(
                timestamps[sched][deployment],
                latencies[sched][deployment],
                label=sched.name,
                marker=MARKERS[ind],
                markevery=marker_interval,
            )

        plt.xlabel("time(s)")
        plt.ylabel("average latency(ms)")
        plt.title(f"average latency - workload: {deployment.name}")
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fontsize=12,
            frameon=True,
            ncol=3,
        )
        plt.tight_layout()
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")


@register_extractor
def average_latency_boxplot(
    config: Config,
    scenarios: List[ScenarioData],
    schedulers: List[Scheduler],
    save_path: str,
) -> None:
    assert len(set([scen.name for scen in scenarios])) == 1

    latencies, _ = calc_average_latency_through_time(config, scenarios)

    latencies_means = {
        sched: {
            deployment: round(statistics.mean(latencies[sched][deployment]), 1)
            for deployment in config.deployments.values()
        }
        for sched in schedulers
    }

    latencies_stdevs = {
        sched: {
            deployment: round(statistics.stdev(latencies[sched][deployment]), 1)
            for deployment in config.deployments.values()
        }
        for sched in schedulers
    }

    x = np.arange(len(schedulers)) * 1.5
    width = 0.3

    fig, ax = plt.subplots(layout="constrained")
    fig.set_size_inches(10.5, 7.5)

    for ind, deployment in enumerate(config.deployments.values()):
        deployment_num = len(config.deployments.values()) / 2

        rects = ax.bar(
            x + (ind - deployment_num / 2) * width + width / 2,
            [latencies_means[sched][deployment] for sched in schedulers],
            width,
            label=deployment.name,
            yerr=[latencies_stdevs[sched][deployment] for sched in schedulers],
        )
        ax.bar_label(rects, padding=10, fontsize=8)

    plt.grid(True, axis="y")
    plt.xlabel("scheduler")
    plt.ylabel("average latency(ms)")
    ax.set_title("average latency")
    ax.set_xticks(x)
    ax.set_xticklabels([sched.name for sched in schedulers], rotation=0)
    ax.set_ylim(0, 350)
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")
    # plt.show()


@register_extractor
def average_latency_CSV(
    config: Config, scenarios: List[ScenarioData], save_path: str
) -> None:
    # TODO add this script
    pass


@register_extractor
def edge_utilization_linechart(
    config: Config,
    scenarios: List[ScenarioData],
    schedulers: List[Scheduler],
    save_path: str,
) -> None:
    assert len(set([scen.name for scen in scenarios])) == 1

    edge_utilization, timestamps = calc_edge_utilization_through_time(config, scenarios)

    fig, ax = plt.subplots()
    marker_interval = 2
    for ind, sched in enumerate(schedulers):
        ax.plot(
            timestamps[sched],
            edge_utilization[sched],
            label=sched.name,
            marker=MARKERS[ind],
            markevery=marker_interval,
        )

    fig.set_size_inches(10.5, 7.5)
    plt.grid(True, axis="y")
    plt.xlabel("time (s)")
    plt.ylabel("edge utilization")
    plt.ylim(0, 1.10)
    plt.yticks(list(map(lambda x: x / 100.0, range(0, 110, 5))))
    plt.title("edge utilization - per scheduler")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=12,
        frameon=True,
        ncol=3,
    )
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")
