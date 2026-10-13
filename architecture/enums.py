"""
Type definitions and enumerations for configuration management.

This module defines shared type aliases, enumerations, and TypedDict
structures used to formalize data types involved in configuration
management.
"""
try:
    from enum import StrEnum, auto
except ImportError:  # Python < 3.11: StrEnum non esiste, fallback equivalente
    from enum import Enum, auto

    class StrEnum(str, Enum):
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()
from typing import TypedDict, Any, List, Dict
from skyfield.api import EarthSatellite

tle_data= list[str]             # Type alias for TLE data representation
Topology = list[EarthSatellite] # Topology

class AccPointMode(StrEnum):
    OPTIMAL = auto()    # "STATIC"
    BASE    = auto()    # "DYNAMIC"

# Alias per il tipo di ritorno di genConfigs() in topology.py
class GenConfigsConfigEntry(TypedDict):
    time: str
    configuration: List[Dict[str, Any]]

class GenConfigsOutput(TypedDict):
    t0: str
    interval: int
    total_seconds: int
    observer_position: Any
    configurations: List[GenConfigsConfigEntry]

