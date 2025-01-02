from typing import Dict, List
from dataclasses import dataclass

from historical.common import Deployment, Node, Scheduler


@dataclass
class Config:
    deployments: Dict[str, Deployment]
    nodes: Dict[str, Node]
    schedulers: List[Scheduler]

    CLOUD_RESPONSE_TIME: int  # ms
    EDGE_RESPONSE_TIME: int  # ms
