from enum import Enum


class Scenario(Enum):
    """Experiment Scenario"""

    DRIVING = "driving"
    PARKING = "parking"


class RoadType(Enum):
    URBAN = "urban"
    HIGHWAY = "highway"


class TimeOfDay(Enum):
    DAY = "day"
    NIGHT = "night"
