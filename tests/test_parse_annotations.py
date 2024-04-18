import pytest

from zod import AnnotationProject, ZodFrames
from zod.constants import NotAnnotatedError


@pytest.fixture()
def zod_frames():
    return ZodFrames("data/zod", version="full")


def test_parse_object_detection(zod_frames):
    for frame in zod_frames:
        frame.get_annotation(AnnotationProject.OBJECT_DETECTION)


def test_parse_traffic_signs(zod_frames):
    for frame in zod_frames:
        try:
            frame.get_annotation(AnnotationProject.TRAFFIC_SIGNS)
        except NotAnnotatedError:
            assert not frame.is_annotated(AnnotationProject.TRAFFIC_SIGNS)


def test_parse_lane_markings(zod_frames):
    for frame in zod_frames:
        frame.get_annotation(AnnotationProject.LANE_MARKINGS)


def test_parse_ego_road(zod_frames):
    for frame in zod_frames:
        try:
            frame.get_annotation(AnnotationProject.EGO_ROAD)
        except NotAnnotatedError:
            assert not frame.is_annotated(AnnotationProject.EGO_ROAD)


def test_parse_road_condition(zod_frames):
    for frame in zod_frames:
        frame.get_annotation(AnnotationProject.ROAD_CONDITION)
