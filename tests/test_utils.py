from types import SimpleNamespace

import pytest

from extractors.utils import calc_node_fragmentation, calc_node_used_resources
from historical.common import Deployment, Node
from historical.data import Cycle, HpaState, PodPlacement


def test_calc_node_used_resources():
    cases = {
        "node with zero memory": {
            "node": Node(name="some name", is_on_edge=True, resources=[10, 10]),
            "pod_placement": PodPlacement(
                node_pods={
                    Node(name="some name", is_on_edge=True, resources=[10, 10]): [
                        Deployment(name="a", resources=[1, 0]),
                        Deployment(name="a", resources=[1, 0]),
                        Deployment(name="a", resources=[1, 0]),
                        Deployment(name="a", resources=[1, 0]),
                    ]
                }
            ),
            "expected": (4, 0),
        },
        "node with zero cpu": {
            "node": Node(name="some name", is_on_edge=True, resources=[10, 10]),
            "pod_placement": PodPlacement(
                node_pods={
                    Node(name="some name", is_on_edge=True, resources=[10, 10]): [
                        Deployment(name="a", resources=[0, 1]),
                        Deployment(name="a", resources=[0, 1]),
                        Deployment(name="a", resources=[0, 1]),
                        Deployment(name="a", resources=[0, 1]),
                    ]
                }
            ),
            "expected": (0, 4),
        },
        "pod with non-zero cpu and memory": {
            "node": Node(name="some name", is_on_edge=True, resources=[10, 10]),
            "pod_placement": PodPlacement(
                node_pods={
                    Node(name="some name", is_on_edge=True, resources=[10, 10]): [
                        Deployment(name="a", resources=[1, 4]),
                        Deployment(name="a", resources=[2, 3]),
                        Deployment(name="a", resources=[3, 2]),
                        Deployment(name="a", resources=[4, 1]),
                    ]
                }
            ),
            "expected": (10, 10),
        },
        "node doesn't exist in pod_placement": {
            "node": Node(name="some other node name", is_on_edge=True, resources=[10, 10]),
            "pod_placement": PodPlacement(
                node_pods={
                    Node(name="some name", is_on_edge=True, resources=[10, 10]): [
                        Deployment(name="a", resources=[1, 4]),
                        Deployment(name="a", resources=[2, 3]),
                        Deployment(name="a", resources=[3, 2]),
                        Deployment(name="a", resources=[4, 1]),
                    ]
                }
            ),
            "expected": (0, 0),
        },
    }

    for case in cases.values():
        placement = case["pod_placement"]
        state = HpaState(deployment_metrics={Deployment(name="a", resources=[1, 1]): 1})
        cycle = Cycle(1, state, placement)

        actual = calc_node_used_resources(case["node"], cycle)
        expected = case["expected"]

        assert actual == expected


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "node_resources": (8, 32),
                "used": (4, 16),
                "expected": 0.75,  # 1 - (0.5 * 0.5)
            },
            id="half-cpu_half-mem",
        ),
        pytest.param(
            {
                "node_resources": (16, 64),
                "used": (0, 0),
                "expected": 1.0,  # completely free
            },
            id="idle-node",
        ),
        pytest.param(
            {
                "node_resources": (12, 48),
                "used": (12, 48),
                "expected": 0.0,  # fully utilized
            },
            id="fully-used",
        ),
        pytest.param(
            {
                "node_resources": (10, 40),
                "used": (5, 10),
                "expected": 1 - ((0.5) * (0.25)),  # 0.875
            },
            id="skewed-usage",
        ),
    ],
)
def test_calc_node_fragmentation(monkeypatch, case):
    # Minimal stand-ins for the real domain objects; add attributes if your Node/Cycle need more.
    node = SimpleNamespace(resources=case["node_resources"])
    cycle = SimpleNamespace()

    def fake_calc_node_used_resources(node_arg, cycle_arg):
        # Ensure the function is called with the objects we expect.
        assert node_arg is node
        assert cycle_arg is cycle
        return case["used"]

    # Patch the helper so we don’t re-test it.
    monkeypatch.setattr(
        "extractors.utils.calc_node_used_resources",  # update to real module path
        fake_calc_node_used_resources,
    )

    fragmentation = calc_node_fragmentation(node, cycle)

    assert fragmentation == pytest.approx(case["expected"])
