from dataclasses import dataclass
from typing import Dict, List

from historical.common import Deployment, Node


@dataclass
class Config:
    deployments: Dict[str, Deployment]
    nodes: Dict[str, Node]

    CLOUD_RESPONSE_TIME: int  # ms
    EDGE_RESPONSE_TIME: int  # ms

    def get_edge_resources(self) -> List[float]:
        res_lists = [node.resources for node in self.nodes.values() if node.is_on_edge]
        return [sum(x) for x in zip(*res_lists)]
