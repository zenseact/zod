from zod.sequences.info import ZodSequenceInfo
from zod.dataclasses.metadata import SequenceMetadata
from zod.dataclasses.oxts import EgoMotion
from zod.dataclasses.zod_dataclasses import Calibration


class ZodSequence:
    def __init__(self, info: ZodSequenceInfo):
        self.info: ZodSequenceInfo = info
        self._ego_motion: EgoMotion = None
        self._oxts: EgoMotion = None
        self._calibration: Calibration = None
        self._metadata: SequenceMetadata = None

    @property
    def ego_motion(self) -> EgoMotion:
        """Get the oxts file."""
        if self.ego_motion is None:
            self.ego_motion = EgoMotion.from_json(self.info.ego_motion_path)
        return self.ego_motion

    @property
    def oxts(self) -> EgoMotion:
        """Get the oxts."""
        if self.oxts is None:
            self.oxts = EgoMotion.from_sequence_oxts(self.info.oxts_path)
        return self.oxts

    @property
    def calibration(self) -> Calibration:
        """Get the calibration."""
        if self.calibration is None:
            self.calibration = Calibration.from_json(self.info.calibration_path)
        return self.calibration

    @property
    def metadata(self) -> SequenceMetadata:
        """Get the metadata."""
        if self.metadata is None:
            self.metadata = SequenceMetadata.from_json(self.info.metadata_path)
        return self.metadata
