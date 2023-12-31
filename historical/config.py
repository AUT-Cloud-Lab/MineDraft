from typing import Dict
from dataclasses import dataclass

from historical.common import Deployment, Node

@dataclass
class Config:
    deployments: Dict[str, Deployment]
    nodes: Dict[str, Node]
