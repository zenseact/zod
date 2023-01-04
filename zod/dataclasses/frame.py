import json
from typing import List

from zod.frames.annotation_parser import parse_ego_road_annotation, parse_lane_markings_annotation
from zod.frames.info import FrameInfo
from zod.dataclasses.metadata import FrameMetaData
from zod.utils.objects import AnnotatedObject
from zod.dataclasses.oxts import EgoMotion
from zod.dataclasses.zod_dataclasses import Calibration


class ZodFrame:
    def __init__(self, info: FrameInfo):
        self.info: FrameInfo = info
        self._ego_motion: EgoMotion = None
        self._oxts: EgoMotion = None
        self._calibration: Calibration = None
        self._metadata: FrameMetaData = None

    @property
    def ego_motion(self) -> EgoMotion:
        """Get the oxts file."""
        if self.ego_motion is None:
            self.ego_motion = EgoMotion.from_json(self.info.ego_motion_path)
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
            self.calibration = Calibration.from_json(self.info.calibration_path)
        return self.calibration

    @property
    def metadata(self) -> FrameMetaData:
        """Get the metadata."""
        if self.metadata is None:
            self.metadata = FrameMetaData.from_json(self.info.metadata_path)
        return self.metadata

    def get_lane_markings_annotation(self):
        return parse_lane_markings_annotation(self.info.lane_markings_annotation_path)

    def get_ego_road_annotation(self):
        return parse_ego_road_annotation(self.info.ego_road_annotation_path)

    def get_object_detection_annotation(self) -> List[AnnotatedObject]:
        """Read object detection annotation from json format."""
        with open(self.info.object_detection_annotation_path) as f:
            objs = json.load(f)
        return [AnnotatedObject.from_dict(anno) for anno in objs]

    def get_traffic_sign_annotation(self):
        raise NotImplementedError

    def get_road_condition_annotation(self):
        raise NotImplementedError
