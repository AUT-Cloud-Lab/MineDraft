import argparse
import json
import sys

from extractors.decorator import extractor_scripts
from historical.parse import parse_config, parse_history


def main():
    parser = argparse.ArgumentParser(description="Extractors")
    parser.add_argument("--script_name", type=str, help="Script name")
    parser.add_argument("--config_path", type=str, help="Config path")
    parser.add_argument("--history_paths", type=str, nargs="+", help="Histories path")
    parser.add_argument("--save-path", type=str, help="save path of the output file")

    args = vars(parser.parse_args())
    script_name = args["script_name"]
    config_path = args["config_path"]
    history_paths = args["history_paths"]
    save_path = args["save_path"]

    if script_name not in extractor_scripts:
        print(f"Script {script_name} not found!")
        print("Available scripts:")
        for extractor_name in extractor_scripts:
            print(f"\t{extractor_name}")
        return 1

    with open(config_path, "r") as config_file:
        config = parse_config(json.load(config_file))

    histories = []
    for history_path in history_paths:
        with open(history_path, "r") as history_file:
            histories.append(parse_history(config, json.load(history_file)))

    extractor_scripts[script_name](config, histories, save_path)


if __name__ == "__main__":
    sys.exit(main())
