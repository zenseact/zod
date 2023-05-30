from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from pyquaternion import Quaternion

from zod.constants import Camera, Lidar
from zod.data_classes.box import Box2D, Box3D

OBJECT_CLASSES_DYNAMIC = (
    "Vehicle",
    "VulnerableVehicle",
    "Pedestrian",
    "Animal",
)
OBJECT_CLASSES_STATIC = (
    "PoleObject",
    "TrafficBeacon",
    "TrafficSign",
    "TrafficSignal",
    "TrafficGuide",
    "DynamicBarrier",
)
OBJECT_CLASSES = (
    *OBJECT_CLASSES_DYNAMIC,
    *OBJECT_CLASSES_STATIC,
    "Unclear",
)

CLASSES_WITH_SUBCLASSES = ("Vehicle", "VulnerableVehicle", "TrafficSign", "TrafficSignal")
VEHICLE_SUBCLASSES = (
    "Car",
    "Van",
    "Truck",
    "Bus",
    "Trailer",
    "TramTrain",
    "HeavyEquip",
    "Emergency",
    "Other",
)
VULNERABLE_VEHICLE_SUBCLASSES = (
    "Bicycle",
    "Motorcycle",
    "Stroller",
    "Wheelchair",
    "PersonalTransporter",
    "NoRider",
    "Other",
)
OBJECT_SUBCLASSES = (
    *[cls_ for cls_ in OBJECT_CLASSES if cls_ not in CLASSES_WITH_SUBCLASSES],
    *[f"Vehicle_{type_}" for type_ in VEHICLE_SUBCLASSES],
    *[f"VulnerableVehicle_{type_}" for type_ in VULNERABLE_VEHICLE_SUBCLASSES],
    "TrafficSign_Front",
    "TrafficSign_Back",
    "TrafficSignal_Front",
    "TrafficSignal_Back",
)


@dataclass
class ObjectAnnotation:
    """Class to store dynamic object information."""

    # These are always available
    box2d: Box2D
    unclear: bool
    name: str
    uuid: str

    # These are not set if the object is unclear
    box3d: Optional[Box3D]  # This can be None even if the object is not unclear
    object_type: Optional[str]
    occlusion_level: Optional[str]
    artificial: Optional[str]
    with_rider: Optional[bool]
    emergency: Optional[bool]
    traffic_content_visible: Optional[bool]

    @property
    def superclass(self):
        """Get the super-class of the object."""
        return self.name

    @property
    def subclass(self):
        """Get the sub-class of the object."""
        if self.unclear:
            return f"Unclear"
        superclass = self.superclass
        if superclass not in CLASSES_WITH_SUBCLASSES:
            return superclass
        if self.artificial is True:
            return f"{superclass}_Artificial"
        if self.traffic_content_visible is not None:
            return f"{superclass}_{'Front' if self.traffic_content_visible else 'Back'}"
        if self.with_rider is False:
            return f"{superclass}_NoRider"
        if self.emergency is True and superclass == "Vehicle":
            return f"{superclass}_Emergency"  # other superclasses have too few emergency objs
        if self.object_type is not None:
            return f"{superclass}_{self.object_type}"
        # the above should have been exhaustive
        raise ValueError("Object subclass could not be determined")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObjectAnnotation:
        """Create an ObjectAnnotation from a dictionary."""
        properties: Dict[str, Any] = data["properties"]

        box2d = Box2D.from_points(points=data["geometry"]["coordinates"], frame=Camera.FRONT)
        box3d = None
        if "location_3d" in properties:
            box3d = Box3D(
                center=np.array(properties["location_3d"]["coordinates"]),
                size=np.array(
                    [
                        properties["size_3d_length"],
                        properties["size_3d_width"],
                        properties["size_3d_height"],
                    ]
                ),
                orientation=Quaternion(
                    properties["orientation_3d_qw"],
                    properties["orientation_3d_qx"],
                    properties["orientation_3d_qy"],
                    properties["orientation_3d_qz"],
                ),
                frame=Lidar.VELODYNE,
            )

        def _parse_bool_prop(key: str) -> Optional[bool]:
            return None if key not in properties else properties[key] == "True"

        superclass, obj_type = properties["class"], properties.get("type")
        unclear = "Inconclusive" in (superclass, obj_type) or properties["unclear"]
        if superclass == "Inconclusive":
            superclass = "Unclear"
        return cls(
            uuid=properties["annotation_uuid"],
            box2d=box2d,
            box3d=box3d,
            unclear=unclear,
            name=superclass,
            object_type=obj_type,
            occlusion_level=properties.get("occlusion_ratio", None),
            artificial=properties.get("artificial", None),
            with_rider=_parse_bool_prop("with_rider"),
            emergency=properties.get("emergency", None),
            traffic_content_visible=_parse_bool_prop("traffic_content_visible"),
        )


# Compatibility with old naming
AnnotatedObject = ObjectAnnotation
