from zod.constants import AnnotationProject
from zod.dataclasses.info import Information
from zod.dataclasses.metadata import SequenceMetadata
from zod.dataclasses.oxts import EgoMotion
from zod.dataclasses.zod_dataclasses import Calibration
from zod.frames.annotation_parser import ANNOTATION_PARSERS


class ZodSequence:
    def __init__(self, info: Information):
        self.info: Information = info  # holds all the paths to the files
        self._ego_motion: EgoMotion = None  # this is the light-weight version of oxts
        self._oxts: EgoMotion = None
        self._calibration: Calibration = None
        self._metadata: SequenceMetadata = None

    @property
    def ego_motion(self) -> EgoMotion:
        """Get the oxts file."""
        if self._ego_motion is None:
            self._ego_motion = EgoMotion.from_json_path(self.info.ego_motion_path)
        return self._ego_motion

    @property
    def oxts(self) -> EgoMotion:
        """Get the oxts."""
        if self._oxts is None:
            self._oxts = EgoMotion.from_oxts_path(self.info.oxts_path)
        return self._oxts

    @property
    def calibration(self) -> Calibration:
        """Get the calibration."""
        if self._calibration is None:
            self._calibration = Calibration.from_json_path(self.info.calibration_path)
        return self._calibration

    @property
    def metadata(self) -> SequenceMetadata:
        """Get the metadata."""
        if self._metadata is None:
            self._metadata = SequenceMetadata.from_json_path(self.info.metadata_path)
        return self._metadata

    def get_annotation(self, project: AnnotationProject):
        """Get the annotation for a given project."""
        path = self.info.get_keyframe_annotation(project).filepath
        return ANNOTATION_PARSERS[project](path)
