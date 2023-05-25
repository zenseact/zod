from zod.anno.object import OBJECT_CLASSES, OBJECT_SUBCLASSES

# Try to evaluate all classes, including subclasses.
all_classes = set(OBJECT_CLASSES + OBJECT_SUBCLASSES)
all_classes -= {"Unclear"}
EVALUATION_CLASSES = sorted(all_classes)
