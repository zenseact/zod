from zod.anno.object import OBJECT_CLASSES, OBJECT_SUBCLASSES

# Try to evaluate all classes, including subclasses.
all_classes = set(OBJECT_CLASSES + OBJECT_SUBCLASSES)
all_classes -= {"Unclear"}
all_classes = {cls_ for cls_ in all_classes if "InconclusiveOrOther" not in cls_}
EVALUATION_CLASSES = sorted(all_classes)
