from typing import List, Dict
from math import ceil

from historical.config import Config
from historical.data import History, Cycle
from historical.common import Deployment
from extractors.decorator import register_extractor


@register_extractor
def check_equality(config: Config, histories: List[History]) -> None:
    """
    Check if the histories are equal.
    """
    number_of_differences: Dict[Deployment, int] = {
        deployment: 0 for _, deployment in config.deployments.items()
    }
    for cycle_number in range(max(map(lambda history: len(history.cycles), histories))):
        cycles: List[Cycle] = []
        for history in histories:
            if cycle_number < len(history.cycles):
                cycles.append(history.cycles[cycle_number])
            else:
                for _, deployment in config.deployments.items():
                    number_of_differences[deployment] += 1

        main_cycle = cycles[0]
        for other_cycle in cycles[1:]:
            for _, deployment in config.deployments.items():
                main_cycle_metric = ceil(main_cycle.hpa.deployment_metrics[deployment])
                other_cycle_metric = ceil(
                    other_cycle.hpa.deployment_metrics[deployment]
                )
                if main_cycle_metric != other_cycle_metric:
                    number_of_differences[deployment] += 1

    for deployment, number_of_differences in number_of_differences.items():
        print(f"{deployment}'s number of differences are {number_of_differences}!")
