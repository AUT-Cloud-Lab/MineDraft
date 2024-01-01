import sys
import json
import argparse

from extractors.decorator import extractor_scripts
from historical.parse import parse_config, parse_history


def main():
    parser = argparse.ArgumentParser(description="Extractors")
    parser.add_argument("--script_name", type=str, help="Script name")
    parser.add_argument("--config_path", type=str, help="Config path")
    parser.add_argument("--history_paths", type=str, nargs="+", help="Histories path")

    args = vars(parser.parse_args())
    if args["script_name"] not in extractor_scripts:
        print(f"Script {args['script_name']} not found!")
        print("Available scripts:")
        for extractor_name in extractor_scripts:
            print(f"\t{extractor_name}")
        return 1

    with open(args["config_path"], "r") as config_file:
        config = parse_config(json.load(config_file))
    histories = []
    for history_path in args["history_paths"]:
        with open(history_path, "r") as history_file:
            histories.append(parse_history(config, json.load(history_file)))

    extractor_scripts[args["script_name"]](config, histories)


if __name__ == "__main__":
    sys.exit(main())
