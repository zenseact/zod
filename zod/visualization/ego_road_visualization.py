from .polygon_utils import overlay_polygon_on_image


def overlay_ego_road_on_image(polygons, image, fill_color=(0, 100, 0), alpha=1.0):
    for polygon in polygons:
        image = overlay_polygon_on_image(polygon, image, fill_color, alpha)
    return image
