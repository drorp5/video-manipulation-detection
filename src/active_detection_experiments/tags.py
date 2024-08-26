"""
tags.py - Enumerations for Experiment Scenarios and Conditions

This module defines enumerations used to categorize and describe different aspects
of active detection experiments.

Enumerations:
- Scenario: Defines the type of scenario (e.g., DRIVING, PARKING)
- RoadType: Defines the type of road in the experiment (e.g., URBAN, HIGHWAY)
- TimeOfDay: Defines the time of day for the experiment (e.g., DAY, NIGHT)

These enumerations are used to standardize and categorize experiment parameters
across the active detection experiments.

Usage:
Import and use these enumerations in other modules to ensure consistent
categorization of experiment scenarios and conditions.
"""

from enum import Enum


class Scenario(Enum):
    """Experiment Scenario"""

    DRIVING = "driving"
    PARKING = "parking"


class RoadType(Enum):
    """Type of road in the experiment"""

    URBAN = "urban"
    HIGHWAY = "highway"


class TimeOfDay(Enum):
    """Time of day for the experiment"""

    DAY = "day"
    NIGHT = "night"
