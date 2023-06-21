from zod.anno.object import OBJECT_CLASSES, OBJECT_SUBCLASSES

# Try to evaluate all classes, including subclasses.
all_classes = set(OBJECT_CLASSES + OBJECT_SUBCLASSES)
# Remove classes that
all_classes -= {
    "Unclear",
    "Vehicle_Other",
    "VulnerableVehicle_Other",
    "VulnerableVehicle_Wheelchair",
}
EVALUATION_CLASSES = sorted(all_classes)
