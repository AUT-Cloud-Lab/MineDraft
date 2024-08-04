FILE_PATH = "Makefile.iter"

KIND = "normal_multiple"
SCRIPT_NAME = "placement_ratio_linechart"
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


result_str = f"{KIND}_{SCRIPT_NAME}:\n"

for report_dir_name in REPORT_DIR_NAMES:
    parent_dir_name, child_dir_name = SCRIPT_NAME.rsplit("_", 1)
    result_dir_name = f"./results/{parent_dir_name}/{child_dir_name}/{report_dir_name}"

    history_paths = r""""""
    for it in range(1, REPORT_FILE_COUNT + 1):
        report_filename = f"{report_dir_name}_{it}.json"
        base_history_paths = rf"        reports/ecmus/$(date_ecmus)/{report_dir_name}/{report_filename} \
            reports/kube-schedule/$(date_kube_schedule)/{report_dir_name}/{report_filename} \
            reports/ecmus-no-migration/$(date_ecmus_no_migration)/{report_dir_name}/{report_filename} \
            reports/random-scheduler/$(date_random)/{report_dir_name}/{report_filename} \
            reports/cloud-first-scheduler/$(date_cloud_first)/{report_dir_name}/{report_filename} \
            reports/smallest-edge-first-scheduler/$(date_smallest_edge_first)/{report_dir_name}/{report_filename} \
            reports/biggest-edge-first-scheduler/$(date_biggest_edge_first)/{report_dir_name}/{report_filename} " + "\\"

        if it != REPORT_FILE_COUNT:
            base_history_paths += "\n"

        history_paths += base_history_paths

    command_str = "\t python3 main.py \\\n" + rf"""    --script_name {SCRIPT_NAME} \
        --config_path config.json \
        --history_paths \
    {history_paths}
        --scenario-name {report_dir_name} \
        --save-path {result_dir_name}
    """

    command_str += "\n"
    result_str += command_str

if __name__ == "__main__":
    with open(FILE_PATH, "a") as file:
        file.write(result_str)
