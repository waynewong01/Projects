"""
Data structures used across the positioning system.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Position:
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class GroundTruth:
    x: float
    y: float
    set_time: float = field(default_factory=time.time)


@dataclass
class AlgorithmResult:
    name: str
    position: Optional[Position]
    distances: Dict[str, float]
    error: Optional[float] = None
