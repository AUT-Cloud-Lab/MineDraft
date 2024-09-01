FILE_PATH = "Makefile.iter"

KIND = "normal_multiple"

SCRIPT_NAMES = [
    "average_latency_boxplot",
    "average_latency_linechart",
    "edge_utilization_linechart",
    "placement_ratio_linechart",
    "pod_count_linechart",
]

REPORT_FILE_COUNT = 10

REPORT_DIR_NAMES = [
    "normal_scenario_0.75_0.1",
    "normal_scenario_0.75_0.5",
    "normal_scenario_0.75_0.7",
    "normal_scenario_1.0_0.1",
    "normal_scenario_1.0_0.5",
    "normal_scenario_1.0_0.7",
    "normal_scenario_1.25_0.1",
    "normal_scenario_1.25_0.5",
    "normal_scenario_1.25_0.7",
]

def generate_result_str(kind, script_name, report_dir_names, report_file_count):
    result_str = f"{kind}_{script_name}:\n"

    for report_dir_name in report_dir_names:
        parent_dir_name, child_dir_name = script_name.rsplit("_", 1)
        result_dir_name = f"./results/{parent_dir_name}/{child_dir_name}/{report_dir_name}"

        history_paths = r""""""
        for it in range(1, report_file_count + 1):
            report_filename = f"{report_dir_name}_{it}.json"
            base_history_paths = rf"        reports/ecmus/$(date_ecmus)/{report_dir_name}/{report_filename} \
                reports/kube-schedule/$(date_kube_schedule)/{report_dir_name}/{report_filename} \
                reports/ecmus-no-migration/$(date_ecmus_no_migration)/{report_dir_name}/{report_filename} \
                reports/random-scheduler/$(date_random)/{report_dir_name}/{report_filename} \
                reports/cloud-first-scheduler/$(date_cloud_first)/{report_dir_name}/{report_filename} \
                reports/smallest-edge-first-scheduler/$(date_smallest_edge_first)/{report_dir_name}/{report_filename} \
                reports/biggest-edge-first-scheduler/$(date_biggest_edge_first)/{report_dir_name}/{report_filename} \
                reports/ecmus-qos-aware/$(date_ecmus_qos_aware)/{report_dir_name}/{report_filename} " + "\\"

            if it != report_file_count:
                base_history_paths += "\n"

            history_paths += base_history_paths

        command_str = "\t python3 main.py \\\n" + rf"""    --script_name {script_name} \
            --config_path config.json \
            --history_paths \
        {history_paths}
            --scenario-name {report_dir_name} \
            --save-path {result_dir_name}
        """

        command_str += "\n"
        result_str += command_str

    return result_str


if __name__ == "__main__":
    with open(FILE_PATH, "a") as file:
        for script_name in SCRIPT_NAMES:
            result_str = generate_result_str(KIND, script_name, REPORT_DIR_NAMES, REPORT_FILE_COUNT) + "\n"
            file.write(result_str)
