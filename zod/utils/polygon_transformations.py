import cv2
import numpy as np


def _create_image_mask(img_shape):
    """
    Creating empty (black) image
    """
    return np.zeros(img_shape, dtype=np.uint8)


def polygon_to_array(polygon):
    """
    Reshape and make sure the polygon is numpy
    """
    poly_array = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)
    return poly_array


def polygon_to_mask(polygon, img_shape=(2168, 3848, 3), fill_color=(100, 0, 0)):
    """
    Create mask with color for a single polygon
    """
    poly_array = polygon_to_array(polygon)
    image_mask = _create_image_mask(img_shape)
    cv2.fillPoly(image_mask, [poly_array], fill_color)
    return image_mask


def polygons_to_binary_mask(polygons, img_shape=(2168, 3848)):
    """
    Create binary mask for list of polygons
    """
    assert len(img_shape) == 2

    mask = _create_image_mask(img_shape)
    for polygon in polygons:
        mask += polygon_to_mask(polygon, img_shape, 1)
    return mask.astype(bool)
