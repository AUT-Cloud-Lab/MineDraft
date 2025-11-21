import json

import matplotlib.pyplot as plt
import numpy as np

with open("./results/fragmentation_data/model-fragmentation.json", "r") as f:
    data = json.load(f)

name_map = {
    "kube-schedule": "K8S",
    "ecmus-qos-aware": "KubeDSM",
    "smallest-edge-first-scheduler": "SEF",
    "biggest-edge-first-scheduler": "BEF",
    "cloud-first-scheduler": "CF",
}

style_map = {
    "KubeDSM": {"color": "orange", "marker": "s"},  # square
    "K8S": {"color": "blue", "marker": "o"},  # circle
    "SEF": {"color": "green", "marker": "D"},  # diamond
    "BEF": {"color": "red", "marker": "^"},  # triangle up
    "CF": {"color": "purple", "marker": "v"},  # triangle down
}

const_std_group = [
    "normal_scenario_1.1_0.4",
    "normal_scenario_1.2_0.4",
    "normal_scenario_1.3_0.4",
    "normal_scenario_1.4_0.4",
]

const_avg_group = [
    "normal_scenario_1.5_0.1",
    "normal_scenario_1.5_0.2",
    "normal_scenario_1.5_0.3",
    "normal_scenario_1.6_0.4",
]

overall_group = np.concatenate((const_std_group, const_avg_group), axis=0)


def plot_linecharts(scenarios, filename, size):
    # Prepare results for this group
    scheduler_results = {short: [] for short in style_map.keys()}
    for scenario in scenarios:
        for long_name, value in data[scenario].items():
            short_name = name_map[long_name]
            scheduler_results[short_name].append(value)

    # Create plot
    plt.figure(figsize=size)
    for scheduler, values in scheduler_results.items():
        plt.plot(
            scenarios,
            values,
            label=scheduler,
            color=style_map[scheduler]["color"],
            marker=style_map[scheduler]["marker"],
            markersize=6,
            linewidth=1.5,
        )

    plt.xlabel("Workloads", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_grouped_barchart(scenarios, filename, size):
    """
    Create a grouped bar chart for the given scenario list.
    """
    # Collect results per scheduler (ignore random-scheduler implicitly)
    scheduler_results = {short: [] for short in style_map.keys()}
    for scenario in scenarios:
        metrics = data[scenario]
        for long_name, short_name in name_map.items():
            scheduler_results[short_name].append(metrics[long_name])

    n_sched = len(scheduler_results)
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    total_width = 0.8
    bar_width = total_width / n_sched

    # ---- plotting ---- #
    fig, ax = plt.subplots(figsize=size)

    for idx, (scheduler, values) in enumerate(scheduler_results.items()):
        offset = (idx - n_sched / 2) * bar_width + bar_width / 2
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=scheduler,
            color=style_map[scheduler]["color"],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_xlabel("Workloads", fontsize=12)
    ax.set_ylabel("Overall Edge Ratio", fontsize=12)
    ax.set_ylim(0.0, 1.0)

    # Grid/guidelines similar to the provided reference
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # Legend at top (similar treatment as requested earlier)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


plot_linecharts(
    const_std_group, "./results/fragmentation_data/multiline-chart/const-std.png", (10, 6)
)
plot_linecharts(
    const_avg_group, "./results/fragmentation_data/multiline-chart/const-mean.png", (10, 6)
)
plot_linecharts(overall_group, "./results/fragmentation_data/multiline-chart/overall.png", (18, 8))
print("multi-line charts are saved...")


plot_grouped_barchart(
    const_std_group, "./results/fragmentation_data/grouped-barchart/const-std.png", (10, 6)
)

plot_grouped_barchart(
    const_avg_group, "./results/fragmentation_data/grouped-barchart/const-mean.png", (10, 6)
)

plot_grouped_barchart(
    overall_group, "./results/fragmentation_data/grouped-barchart/overall.png", (18, 8)
)
print("grouped barcharts are saved...")
