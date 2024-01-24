from typing import List, Dict
from math import ceil

from historical.config import Config
from historical.data import History, Cycle, Migration
from historical.common import Deployment
from historical.utils import get_nodes_of_a_deployment
from extractors.decorator import register_extractor


@register_extractor
def calc_migrations(config: Config, histories: List[History]) -> None:
    """
    Calculate the number of migrations for each deployment.
    """
    assert len(histories) == 1
    history: History = histories[0]

    def is_cycle_valid(cycle: Cycle) -> bool:
        """
        Check if number of running pods are equal to the desired number of pods.
        """
        for deployment_name in config.deployments:
            number_of_pods = 0
            for (_, pods) in cycle.pod_placement.node_pods.items():
                number_of_pods += len(
                    list(filter(lambda pod: pod.name == deployment_name, pods))
                )

            if ceil(cycle.hpa.deployment_metrics[deployment]) != number_of_pods:
                return False

        return True

    for deployment in config.deployments.values():
        migrations: List[Migration] = []
        for (start, cycle) in enumerate(history.cycles):
            if not is_cycle_valid(cycle):
                continue
            for (_end, next_cycle) in enumerate(history.cycles[start + 1:]):
                end = start + _end + 1
                if not is_cycle_valid(next_cycle):
                    continue

                if ceil(cycle.hpa.deployment_metrics[deployment]) != ceil(
                    next_cycle.hpa.deployment_metrics[deployment]
                ):
                    continue

                # found two cycles with possible migrations!
                source_nodes = sorted(
                    get_nodes_of_a_deployment(cycle.pod_placement, deployment),
                    key=lambda node: node.name,
                )
                target_nodes = sorted(
                    get_nodes_of_a_deployment(next_cycle.pod_placement, deployment),
                    key=lambda node: node.name,
                )

                print(list(map(lambda node: node.name, source_nodes)))
                print(list(map(lambda node: node.name, target_nodes)))

                real_sources = []
                real_targets = []

                it_source = 0
                it_target = 0
                while it_source < len(source_nodes) and it_target < len(target_nodes):
                    if source_nodes[it_source] == target_nodes[it_target]:
                        it_source += 1
                        it_target += 1
                    elif source_nodes[it_source].name < target_nodes[it_target].name:
                        real_sources.append(source_nodes[it_source])
                        it_source += 1
                    else:
                        real_targets.append(target_nodes[it_target])
                        it_target += 1

                while it_source < len(source_nodes):
                    real_sources.append(source_nodes[it_source])
                    it_source += 1

                while it_target < len(target_nodes):
                    real_targets.append(target_nodes[it_target])
                    it_target += 1

                assert len(real_sources) == len(real_targets)
                print(f"here with {len(real_sources)}")
                print(f"{start}, {end}")
                for i in range(len(real_sources)):
                    migrations.append(
                        Migration(
                            deployment=deployment,
                            source=real_sources[i],
                            target=real_targets[i],
                            start=start,
                            end=end,
                        )
                    )

        print(f"Number of migrations for {deployment.name}: {len(migrations)}")
        for migration in migrations:
            print(migration)


@register_extractor
def check_equality(config: Config, histories: List[History]) -> None:
    """
    Check if the histories are equal.
    """
    number_of_differences: Dict[Deployment, int] = {
        deployment: 0 for _, deployment in config.deployments.items()
    }
    print(len(histories[0].cycles))
    print(len(histories[1].cycles))

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


@register_extractor
def average_latency(config: Config, histories: List[History]) -> None:
    print(config)
