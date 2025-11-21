from types import SimpleNamespace
from typing import List

import pytest

from extractors.logic import calc_fragmentation_through_time
from historical.common import Deployment, Node, Scheduler
from historical.config import Config
from historical.data import Cycle, History, PodPlacement, ScenarioData


@pytest.fixture
def config() -> Config:
    deployments = {
        "A": Deployment(name="A", resources=[1, 1]),
    }

    nodes = {
        "node-1": Node(name="node-1", is_on_edge=False, resources=[10, 10]),
        "node-2": Node(name="node-2", is_on_edge=False, resources=[5, 10]),
    }

    cloud_response_time = 300
    edge_response_time = 50

    return Config(deployments, nodes, cloud_response_time, edge_response_time)


@pytest.fixture
def scheduler() -> List[Scheduler]:
    return [Scheduler(name="sched-1")]


@pytest.fixture
def scenario() -> ScenarioData:
    name = "some-scenario-name"
    scheduler_histories = {
        Scheduler(name="sched-1"): [
            History(
                [
                    Cycle(
                        timestamp=1,
                        hpa=SimpleNamespace(),
                        pod_placement=PodPlacement(
                            {
                                Node(name="node-1", is_on_edge=False, resources=[10, 10]): [],
                                Node(name="node-2", is_on_edge=False, resources=[5, 10]): [],
                            }
                        ),
                    ),
                    Cycle(
                        timestamp=2,
                        hpa=SimpleNamespace(),
                        pod_placement=PodPlacement(
                            {
                                Node(name="node-1", is_on_edge=False, resources=[10, 10]): [
                                    Deployment(name="A", resources=[1, 1]),
                                    Deployment(name="A", resources=[1, 1]),
                                ],
                                Node(name="node-2", is_on_edge=False, resources=[5, 10]): [
                                    Deployment(name="A", resources=[1, 1]),
                                ],
                            }
                        ),
                    ),
                    Cycle(
                        timestamp=3,
                        hpa=SimpleNamespace(),
                        pod_placement=PodPlacement(
                            {
                                Node(name="node-1", is_on_edge=False, resources=[10, 10]): [
                                    Deployment(name="A", resources=[1, 1]),
                                    Deployment(name="A", resources=[1, 1]),
                                    Deployment(name="A", resources=[1, 1]),
                                    Deployment(name="A", resources=[1, 1]),
                                ],
                                Node(name="node-2", is_on_edge=False, resources=[5, 10]): [
                                    Deployment(name="A", resources=[1, 1]),
                                    Deployment(name="A", resources=[1, 1]),
                                    Deployment(name="A", resources=[1, 1]),
                                    Deployment(name="A", resources=[1, 1]),
                                ],
                            }
                        ),
                    ),
                ]
            )
        ],
    }

    return ScenarioData(name, scheduler_histories)


def test_calc_fragmentation_through_time(
    config: Config, scenario: ScenarioData, scheduler: list[Scheduler]
):

    frags, times = calc_fragmentation_through_time(config, scenario, scheduler)

    expected_frags = {Scheduler(name="sched-1"): [1, 0.97, 0.76]}
    expected_times = {Scheduler(name="sched-1"): [1, 2, 3]}

    assert frags == expected_frags
    assert times == expected_times
