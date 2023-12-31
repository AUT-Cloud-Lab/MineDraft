from typing import List
from dataclasses import dataclass

@dataclass
class Deployment:
    name: str
    resources: List[float]

    def __hash__(self):
        return hash(self.name)

@dataclass
class Node:
    name: str
    is_on_edge: bool

    def __hash__(self):
        return hash(self.name)
