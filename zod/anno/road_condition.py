from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoadConditionAnnotation:
    wet: bool
    snowy: bool

    @classmethod
    def from_dict(cls, data: dict) -> RoadConditionAnnotation:
        return RoadConditionAnnotation(
            wet=data["wetness"],
            snowy=data["snow_coverage"],
        )
