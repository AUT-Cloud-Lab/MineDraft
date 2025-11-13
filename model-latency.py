import json

import matplotlib.pyplot as plt
import numpy as np

# Read the JSON file
with open("results/model-latency-result.json", "r") as f:
    data = json.load(f)

# Mapping old names to short names used in diagram
name_map = {
    "kube-schedule"                : "K8S",
    "ecmus-qos-aware"              : "KubeDSM",
    "smallest-edge-first-scheduler": "SEF",
    "biggest-edge-first-scheduler" : "BEF",
    "cloud-first-scheduler"        : "CF"
}

# Colors and markers
style_map = {
    "KubeDSM": {"color": "orange", "marker": "s"},  # square
    "K8S"    : {"color": "blue", "marker": "o"},  # circle
    "SEF"    : {"color": "green", "marker": "D"},  # diamond
    "BEF"    : {"color": "red", "marker": "^"},  # triangle up
    "CF"     : {"color": "purple", "marker": "v"}  # triangle down
}

# Scenario groups
const_std_group = [
    "normal_scenario_1.1_0.4",
    "normal_scenario_1.2_0.4",
    "normal_scenario_1.3_0.4",
    "normal_scenario_1.4_0.4"
]

const_avg_group = [
    "normal_scenario_1.5_0.1",
    "normal_scenario_1.5_0.2",
    "normal_scenario_1.5_0.3",
    "normal_scenario_1.6_0.4"
]

overall_group = np.concatenate((const_std_group, const_avg_group), axis = 0)


def plot_group(scenarios, filename, size):
    # Prepare results for this group
    scheduler_results = {short: [] for short in style_map.keys()}
    for scenario in scenarios:
        for long_name, value in data[scenario].items():
            short_name = name_map[long_name]
            scheduler_results[short_name].append(value)

    # Create plot
    plt.figure(figsize = size)
    for scheduler, values in scheduler_results.items():
        plt.plot(
            scenarios,
            values,
            label = scheduler,
            color = style_map[scheduler]["color"],
            marker = style_map[scheduler]["marker"],
            markersize = 6,
            linewidth = 1.5
        )

    plt.xlabel("Workloads", fontsize = 12)
    plt.ylabel("Latency (ms)", fontsize = 12)
    plt.grid(True, linestyle = "--", alpha = 0.6)
    plt.legend(loc = "upper right", fontsize = 10)
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.savefig(filename, dpi = 300)
    plt.close()


# Generate both plots
plot_group(const_std_group, "./results/model-latency/const-std.png", (10, 6))
plot_group(const_avg_group, "./results/model-latency/const-mean.png", (10, 6))
plot_group(overall_group, "./results/model-latency/overall.png", (18, 8))

print("plots saved...")
