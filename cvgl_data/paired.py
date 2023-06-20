import cosy, math

# TODO: move to cosy?
def bev_to_aerial(ground_frame, aerial_frame, scale=1.0):
    world_to_epsg3857 = cosy.proj.eastnorthmeters_at_latlon_to_epsg3857(ground_frame.geopose.latlon)
    x_to_world = lambda latlon, bearing: cosy.Rigid(
        rotation=cosy.angle_to_rotation_matrix(cosy.proj.epsg4326_to_epsg3857.transform_angle(math.radians(bearing))),
        translation=world_to_epsg3857.inverse()(cosy.proj.epsg4326_to_epsg3857(latlon)),
    )

    bev_to_aerial = x_to_world(aerial_frame.geopose.latlon, aerial_frame.geopose.bearing).inverse() * x_to_world(ground_frame.geopose.latlon, ground_frame.geopose.bearing)
    bev_to_aerial.translation *= scale
    return bev_to_aerial
