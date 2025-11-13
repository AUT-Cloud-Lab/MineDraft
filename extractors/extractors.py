import json
import statistics
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from extractors.decorator import register_extractor
from extractors.logic import (
    calc_model_latency_through_time,
    calc_edge_ratio_through_time,
    calc_edge_utilization_through_time,
    calc_pod_count_through_time,
)
from extractors.utils import (
    ensure_directory,
    merge_lists_by_average,
    merge_lists_by_sum,
)
from historical.common import Scheduler
from historical.config import Config
from historical.data import ScenarioData

# TODO make sure these are long enough
COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"]
MARKERS = ["^", "v", "x", "+", "*", "o", "s", "D", "p", "P", "X", "d"]


@register_extractor
def pod_count_linechart(
        config: Config,
        scenario: ScenarioData,
        schedulers: List[Scheduler],
        save_path: str,
) -> None:
    pod_counts, timestamps = calc_pod_count_through_time(config, scenario, schedulers)

    for deployment in config.deployments.values():
        fig, ax = plt.subplots()
        plt.grid(True, axis = "y")
        fig.set_size_inches(10.5, 7.5)
        marker_interval = 2
        for ind, sched in enumerate(schedulers):
            ax.plot(
                timestamps[sched][deployment],
                pod_counts[sched][deployment],
                label = sched.name,
                marker = MARKERS[ind],
                markevery = marker_interval,
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
            loc = "upper center",
            bbox_to_anchor = (0.5, -0.1),
            fontsize = 12,
            frameon = True,
            ncol = 3,
        )
        plt.tight_layout()
        ensure_directory(save_path)
        plt.savefig(f"{save_path}/{deployment.name}.png")

    all_pod_counts = {
        sched: merge_lists_by_sum(
            *[
                pod_counts[sched][deployment]
                for deployment in config.deployments.values()
            ]
        )
        for sched in schedulers
    }
    all_timestamps = {
        sched: merge_lists_by_average(
            *[
                timestamps[sched][deployment]
                for deployment in config.deployments.values()
            ]
        )
        for sched in schedulers
    }

    fig, ax = plt.subplots()
    plt.grid(True, axis = "y")
    fig.set_size_inches(10.5, 7.5)
    markevery = marker_interval
    for ind, sched in enumerate(schedulers):
        ax.plot(
            all_timestamps[sched],
            all_pod_counts[sched],
            label = sched.name,
            marker = MARKERS[ind],
            markevery = markevery,
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
        loc = "upper center",
        bbox_to_anchor = (0.5, -0.1),
        fontsize = 12,
        frameon = True,
        ncol = 3,
    )
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(f"{save_path}/all.png")


@register_extractor
def average_latency_linechart(
        config: Config,
        scenario: ScenarioData,
        schedulers: List[Scheduler],
        save_path: str,
) -> None:
    latencies, timestamps = calc_model_latency_through_time(
        config, scenario, schedulers
    )

    result = {}
    for scheduler, latencies in latencies.items():
        flat = np.array([value for values in latencies.values() for value in values])
        result[scheduler.name] = np.mean(flat)

    json_string = json.dumps({
        scenario.name: result
    })
    ensure_directory("./results")

    with open("./results/model-latency-result.json", "a") as fp:
        fp.write(json_string)
        fp.write("\n")


@register_extractor
def average_latency_boxplot(
        config: Config,
        scenario: ScenarioData,
        schedulers: List[Scheduler],
        save_path: str,
) -> None:
    latencies, _ = calc_model_latency_through_time(config, scenario, schedulers)

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

    fig, ax = plt.subplots(layout = "constrained")
    fig.set_size_inches(10.5, 7.5)

    for ind, deployment in enumerate(config.deployments.values()):
        deployment_num = len(config.deployments.values()) / 2

        rects = ax.bar(
            x + (ind - deployment_num / 2) * width + width / 2,
            [latencies_means[sched][deployment] for sched in schedulers],
            width,
            label = deployment.name,
            yerr = [latencies_stdevs[sched][deployment] for sched in schedulers],
        )
        ax.bar_label(rects, padding = 10, fontsize = 8)

    plt.grid(True, axis = "y")
    plt.xlabel("scheduler")
    plt.ylabel("average latency(ms)")
    ax.set_title("average latency")
    ax.set_xticks(x)
    ax.set_xticklabels([sched.name for sched in schedulers], rotation = 0)
    ax.set_ylim(0, 350)
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")
    # plt.show()


@register_extractor
def edge_utilization_linechart(
        config: Config,
        scenario: ScenarioData,
        schedulers: List[Scheduler],
        save_path: str,
) -> None:
    edge_utilization, timestamps = calc_edge_utilization_through_time(
        config, scenario, schedulers
    )

    fig, ax = plt.subplots()
    marker_interval = 2
    for ind, sched in enumerate(schedulers):
        ax.plot(
            timestamps[sched],
            edge_utilization[sched],
            label = sched.name,
            marker = MARKERS[ind],
            markevery = marker_interval,
        )

    fig.set_size_inches(10.5, 7.5)
    plt.grid(True, axis = "y")
    plt.xlabel("time (s)")
    plt.ylabel("edge utilization")
    plt.ylim(0, 1.10)
    plt.yticks(list(map(lambda x: x / 100.0, range(0, 110, 5))))
    plt.title("edge utilization - per scheduler")
    plt.legend(
        loc = "upper center",
        bbox_to_anchor = (0.5, -0.1),
        fontsize = 12,
        frameon = True,
        ncol = 3,
    )
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(save_path + "/result.png")


@register_extractor
def all_data_tables(
        config: Config,
        scenario: ScenarioData,
        schedulers: List[Scheduler],
        save_path: str,
) -> None:
    edge_utilization, _ = calc_edge_utilization_through_time(
        config, scenario, schedulers
    )
    edge_ratio_for_each_deployment, edge_ratio_overall = calc_edge_ratio_through_time(
        config, scenario, schedulers
    )

    with open(f"{save_path}/{scenario.name}.csv", "w") as f:
        header = ["scheduler", "edge-utilization", "edge-ratio-overall"]
        for deployment in config.deployments.values():
            header.append(f"edge-ratio-{deployment.name}")
        f.write(",".join(header) + "\n")
        for sched in schedulers:
            f.write(f"{sched.name},")
            f.write(f"{statistics.mean(edge_utilization[sched]):.2f},")
            f.write(f"{statistics.mean(edge_ratio_overall[sched]):.2f},")
            for ind, deployment in enumerate(config.deployments.values()):
                f.write(
                    f"{statistics.mean(edge_ratio_for_each_deployment[sched][deployment]):.2f}"
                )
                if ind != len(config.deployments.values()) - 1:
                    f.write(",")
            f.write("\n")
