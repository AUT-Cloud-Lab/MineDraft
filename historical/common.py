from dataclasses import dataclass
from typing import List


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
    resources: List[float]

    def __hash__(self):
        return hash(self.name)


@dataclass
class Scheduler:
    name: str

    def __hash__(self):
        return hash(self.name)
