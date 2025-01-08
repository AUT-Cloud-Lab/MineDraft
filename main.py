import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

from extractors.decorator import extractor_functions
from historical.common import Scheduler
from historical.config import Config
from historical.data import History, ScenarioData
from historical.parse import parse_config, parse_history


def get_scenarios_data_for_scheduler_and_scenario(
    config: Config, reports_dir: str, sched: str, scen: str
) -> List[History]:
    if not os.path.exists(reports_dir):
        raise Exception(f"Directory {reports_dir} not found!")

    sched_dir = os.path.join(reports_dir, sched)
    if not os.path.exists(sched_dir):
        raise Exception(
            f"Directory {sched_dir} does not exists, meaning the schedulers {sched} results are not available!"
        )

    latest_date = None
    latest_date_dir = None

    for date in os.listdir(sched_dir):
        date_dir = os.path.join(sched_dir, date)
        date_value = datetime.strptime(date, "%Y-%m-%d")
        if scen in os.listdir(date_dir):
            if latest_date is None or date_value > latest_date:
                latest_date = date_value
                latest_date_dir = os.path.join(date_dir, scen)

    if latest_date is None:
        raise Exception(f"No data found for scenario {scen} and scheduler {sched}!")

    histories = []

    for history_name in os.listdir(latest_date_dir):
        history_path = os.path.join(latest_date_dir, history_name)
        with open(history_path, "r") as history_file:
            histories.append(parse_history(config, json.load(history_file)))

    return histories


def main():
    parser = argparse.ArgumentParser(description="Extractors")
    parser.add_argument("--extractor_name", type=str, help="Extractor name")
    parser.add_argument("--config_path", type=str, help="Config path")
    parser.add_argument("--save_path", type=str, help="Save path of the output file(s)")
    parser.add_argument(
        "--reports_dir",
        type=str,
        required=False,
        help="Path to reports directory",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        required=False,
        help="Scenario's name that should the extractor run on",
    )
    parser.add_argument(
        "--schedulers_names",
        type=str,
        required=False,
        nargs="+",
        help="Schedulers' names that should the extractor run on",
    )

    args = vars(parser.parse_args())
    extractor_name = args["extractor_name"]
    config_path = args["config_path"]
    save_path = args["save_path"]

    reports_dir = args["reports_dir"]

    schedulers_names = args["schedulers_names"]
    scenario_name = args["scenario_name"]

    if extractor_name not in extractor_functions:
        print(f"Extractor {extractor_name} not found!")
        print("Available scripts:")
        for extractor_name in extractor_functions:
            print(f"\t{extractor_name}")
        return 1

    with open(config_path, "r") as config_file:
        config = parse_config(json.load(config_file))

    scen_data = ScenarioData(scenario_name, {})

    schedulers = [Scheduler(name) for name in schedulers_names]
    for scheduler in schedulers:
        histories = get_scenarios_data_for_scheduler_and_scenario(
            config, reports_dir, scheduler.name, scenario_name
        )
        scen_data.scheduler_histories[scheduler] = histories

    extractor_functions[extractor_name](config, scen_data, schedulers, save_path)


if __name__ == "__main__":
    sys.exit(main())
