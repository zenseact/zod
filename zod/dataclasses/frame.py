from zod.constants import AnnotationProject
from zod.dataclasses.info import Information
from zod.dataclasses.metadata import FrameMetaData
from zod.dataclasses.oxts import EgoMotion
from zod.dataclasses.zod_dataclasses import Calibration
from zod.frames.annotation_parser import ANNOTATION_PARSERS


class ZodFrame:
    def __init__(self, info: Information):
        self.info: Information = info
        self._ego_motion: EgoMotion = None
        self._oxts: EgoMotion = None
        self._calibration: Calibration = None
        self._metadata: FrameMetaData = None

    @property
    def ego_motion(self) -> EgoMotion:
        """Get the oxts file."""
        if self.ego_motion is None:
            self._ego_motion = EgoMotion.from_json(self.info.ego_motion_path)
        return self.ego_motion

    @property
    def oxts(self) -> EgoMotion:
        """Get the oxts."""
        raise NotImplementedError("Someone do this.pls.")
        if self.oxts is None:
            self.oxts = EgoMotion.from_frame_oxts(self.info.oxts_path)
        return self.oxts

    @property
    def calibration(self) -> Calibration:
        """Get the calibration."""
        if self.calibration is None:
            self._calibration = Calibration.from_json(self.info.calibration_path)
        return self.calibration

    @property
    def metadata(self) -> FrameMetaData:
        """Get the metadata."""
        if self.metadata is None:
            self._metadata = FrameMetaData.from_json(self.info.metadata_path)
        return self.metadata

    def get_annotation(self, project: AnnotationProject):
        """Get the annotation for a given project."""
        path = self.info.get_keyframe_annotation(project).filepath
        return ANNOTATION_PARSERS[project](path)

    def get_aggregated_point_cloud(self):
        # TODO: adjust core timestamp so that it always points "forward"
        # like this: 0.75*timestamps.max()+0.25*timestamps.min()
        # This is extra important since zodframe scans are not pointwise compensated
        # Or maybe even better to look at the angles of all points and find/interpolate
        # the time that corresponds to -pi/2
        raise NotImplementedError
