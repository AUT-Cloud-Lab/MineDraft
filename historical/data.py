from typing import Dict, List
from datetime import timedelta
from dataclasses import dataclass

from historical.common import Deployment, Node

@dataclass
class HpaState:
    deployment_metrics: Dict[Deployment, float]

@dataclass
class PodPlacement:
    node_pods: Dict[Node, List[Deployment]]

@dataclass
class Cycle:
    timestamp: int
    hpa: HpaState
    pod_placement: PodPlacement

@dataclass
class History:
    cycles: List[Cycle]
