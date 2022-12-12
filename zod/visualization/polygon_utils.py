import cv2
import numpy as np

from ..frames.polygon_annotations.polygon_transformations import polygon_to_mask


def overlay_polygon_on_image(polygon, image, fill_color=(100, 0, 0), alpha=1.0):
    img_shape = np.shape(image)
    out_img = image.copy()

    image_mask = polygon_to_mask(polygon, img_shape, fill_color)

    out_img = cv2.addWeighted(out_img, 1, image_mask, alpha, -1)

    return out_img


def overlay_mask_on_image(mask, image, fill_color=(100, 0, 0), alpha=1.0):
    img_shape = np.shape(image)
    out_img = image.copy()
    mask_img = np.zeros(img_shape, dtype=np.uint8)

    for i, c in enumerate(fill_color):
        mask_img[:, :, i] = mask.astype(np.uint8) * np.uint8(c)

    out_img = cv2.addWeighted(out_img, 1, mask_img, alpha, -1)

    return out_img
