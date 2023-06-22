import tqdm, skimage.draw, skimage.transform, threading, cv2, cosy, cvgl_data, math
import numpy as np
from collections import defaultdict
import pypeln as pl

def draw_line(image, t1, t2, color, thickness=1):
    if not t1 is None and not t2 is None:
        color = np.asarray(color)
        rr, cc = skimage.draw.line(int(t1[0]), int(t1[1]), int(t2[0]), int(t2[1]))
        for ri in range(-thickness + 1, thickness):
            for ci in range(-thickness + 1, thickness):
                rri = rr + ri
                cci = cc + ci
                in_bounds = np.logical_and(np.logical_and(0 <= rri, rri < image.shape[0]), np.logical_and(0 <= cci, cci < image.shape[1]))
                rri = rri[in_bounds]
                cci = cci[in_bounds]
                image[rri, cci] = color

# def draw_points(image, points, color, radius=0.6):
#     points = np.asarray(points)
#     if len(points.shape) == 1:
#         points = points[np.newaxis, :]
#     color = np.asarray(color)
#     if len(color.shape) == 1:
#         color = np.repeat(color[np.newaxis, :], axis=0, repeats=points.shape[0])
#
#     xxc, yyc = np.mgrid[:2 * radius, :2 * radius]
#     circle = (xxc - radius) ** 2 + (yyc - radius) ** 2 < radius ** 2
#     xxc = xxc[circle] - radius
#     yyc = yyc[circle] - radius
#
#     pixels_per_circle = xxc.shape[0]
#     points_num = points.shape[0]
#
#     # Add circle coordinates to every point
#     color = np.repeat(color, pixels_per_circle, axis=0)
#     points = np.repeat(points, pixels_per_circle, axis=0) + np.tile(np.stack([xxc, yyc], axis=1), [points_num, 1])
#
#     # Filter out of bounds points
#     mask = np.all(np.logical_and(0 <= points, points < np.asarray(image.shape[:2])[np.newaxis, :]), axis=-1)
#     color = color[mask, :]
#     points = points[mask, :]
#
#     points = points.astype("int32")
#     if color.shape[1] == 3:
#         image[points[:, 0], points[:, 1]] = color
#     elif color.shape[1] == 4:
#         image[points[:, 0], points[:, 1]] = np.clip(color[:, :3].astype("float32") * color[:, 3:] / 255.0 + image[points[:, 0], points[:, 1]].astype("float32") * (255 - color[:, 3:]) / 255.0, 0.0, 255.0).astype("uint8")
#     else:
#         raise ValueError(f"Color must have 3 or 4 channels, found {color.shape[1]}")

def draw_trajectories(latlons, tile_loader, zoom, bearings=None, colors=None, tile_padding=1, downsample=1, bearing_length=2.0, bearing_stride=5, return_min_pixel=False, verbose=False):
    lock = threading.Lock()
    images = {}
    positions = defaultdict(dict)
    if not bearings is None:
        positions2 = defaultdict(dict)
    if len(latlons) == 0:
        raise ValueError("Must pass at least one scene")

    stream = [(scene_index, frame_index) for scene_index in range(len(latlons)) for frame_index in range(len(latlons[scene_index]))]
    @cvgl_data.unwrap
    def load(scene_index, frame_index):
        latlon = latlons[scene_index][frame_index]

        center_tile = tile_loader.layout.epsg4326_to_tile(latlon, zoom=zoom).astype("int64")
        for x in range(-tile_padding, tile_padding + 1):
            for y in range(-tile_padding, tile_padding + 1):
                tile = center_tile + np.asarray([x, y])
                if not (tile[1], tile[0]) in images:
                    tile_image = tile_loader.load(tile, zoom=zoom)
                    if downsample != 1:
                        tile_image = skimage.transform.downscale_local_mean(tile_image, (downsample, downsample, 1))
                    with lock:
                        images[(tile[1], tile[0])] = tile_image

        position = (tile_loader.layout.epsg4326_to_pixel(latlon, zoom=zoom) // downsample).astype("int64")
        with lock:
            positions[scene_index][frame_index] = position

        if not bearings is None:
            bearing = bearings[scene_index][frame_index]
            latlon2 = cosy.geo.move_from_latlon(latlon, bearing, distance=bearing_length)
            position2 = (tile_loader.layout.epsg4326_to_pixel(latlon2, zoom=zoom) // downsample).astype("int64")
            with lock:
                positions2[scene_index][frame_index] = position2
    
    stream = pl.thread.map(load, stream, workers=4, maxsize=4)
    if verbose:
        stream = tqdm.tqdm(stream, total=len(input))
    for _ in stream:
        pass

    # Construct image
    tile_shape = tile_loader.layout.tile_shape // downsample
    tile_coords = np.asarray(list(images.keys()))
    tile_min = np.amin(tile_coords, axis=0)
    tile_max = np.amax(tile_coords, axis=0)
    tile_num = (tile_max - tile_min) + 1
    image_shape = tile_num * tile_shape
    image = np.zeros((image_shape[0], image_shape[1], 3), dtype="uint8")
    for tile, subimage in images.items():
        start = ((tile - tile_min) * tile_shape).astype("int32")
        end = start + tile_shape
        image[start[0]:end[0], start[1]:end[1]] = subimage
    min_pixel = tile_min * tile_shape

    # Draw on image
    for scene_index in range(len(latlons)):
        # Trajectory
        positions_scene = [position for frame_index, position in sorted(positions[scene_index].items())]
        positions_scene = np.asarray(positions_scene) - min_pixel
        for i in range(positions_scene.shape[0] - 1):
            color = colors[scene_index][i] if not colors is None else np.asarray([255, 0, 0])
            draw_line(image, positions_scene[i], positions_scene[i + 1], color, thickness=2)

        # Bearings
        if not bearings is None:
            positions2_scene = [position2 for frame_index, position2 in sorted(positions2[scene_index].items())]
            positions2_scene = np.asarray(positions2_scene) - tile_min * tile_shape
            color = np.asarray([0, 0, 255])
            for i in range(0, positions2_scene.shape[0] - 1, bearing_stride):
                draw_line(image, positions_scene[i], positions2_scene[i], color, thickness=1)

    results = [image]
    if return_min_pixel:
        results.append(min_pixel)
    return tuple(results) if len(results) > 1 else results[0]

def draw_paired_frame(ground_frame, aerial_frame, points_alpha=1.0, radius=None, color="height"):
    aerial_image = np.copy(aerial_frame.map.image)

    points = ground_frame.lidar.points
    if points.shape[0] > 0:
        if isinstance(color, str) and color == "height":
            height = points[:, 2]
            min_height = np.amin(height)
            max_height = np.amax(height)
            color = (height - min_height) / (max_height - min_height + 1e-7)
            color = cv2.applyColorMap(np.asarray([color * 255.0]).astype("uint8"), cv2.COLORMAP_JET)[0, :, ::-1]
        color = np.asarray(color)

        # Rotate points to vertical
        ego_to_world = ground_frame.ego_to_world.transform
        rotation = np.eye(3, dtype="float")
        rotation[:2, :2] = cosy.angle_to_rotation_matrix(cosy.angle(np.asarray([1.0, 0.0]), (ego_to_world.rotation @ np.asarray([1.0, 0.0, 0.0]))[:2]))
        verticalego_to_world = cosy.Rigid(
            rotation=rotation,
            translation=ego_to_world.translation,
        )
        ego_to_verticalego = verticalego_to_world.inverse() * ego_to_world
        points = ego_to_verticalego(points)

        # Transform to aerial coordinates
        bevpixels_to_aerialpixels = cvgl_data.bev_to_aerial(ground_frame, aerial_frame, scale=1.0 / aerial_frame.map.meters_per_pixel)
        aerial_pixels = points[:, :2] / aerial_frame.map.meters_per_pixel
        aerial_pixels = bevpixels_to_aerialpixels(aerial_pixels) # TODO: proper projection using up-vector
        aerial_pixels = aerial_pixels + np.asarray(aerial_frame.map.image.shape[:2])[np.newaxis, :] / 2 # TODO: offset vehicle from center of bev map

        mask = np.all(np.logical_and(0 <= aerial_pixels, aerial_pixels < np.asarray(aerial_frame.map.image.shape[:2])[np.newaxis, :]), axis=-1)
        aerial_pixels = aerial_pixels[mask]
        if len(color.shape) >= 2:
            color = color[mask]

        if not (not radius is None and radius == 0):
            if not radius is None:
                xxc, yyc = np.mgrid[:2 * int(radius), :2 * int(radius)]
                circle = (xxc - radius) ** 2 + (yyc - radius) ** 2 < radius ** 2
                xxc = xxc[circle] - radius
                yyc = yyc[circle] - radius

                pixels_per_circle = xxc.shape[0]
                points_num = aerial_pixels.shape[0]

                aerial_pixels = np.repeat(aerial_pixels, pixels_per_circle, axis=0) + np.tile(np.stack([xxc, yyc], axis=1), [points_num, 1])

                mask = np.all(np.logical_and(0 <= aerial_pixels, aerial_pixels < np.asarray(aerial_image.shape[:2])[np.newaxis, :]), axis=-1)
                aerial_pixels = aerial_pixels[mask, :]
            aerial_pixels = aerial_pixels.astype("int32")
            aerial_image[aerial_pixels[:, 0], aerial_pixels[:, 1]] = np.clip(points_alpha * color + (1.0 - points_alpha) * aerial_image[aerial_pixels[:, 0], aerial_pixels[:, 1]], 0.0, 255.0).astype("uint8")

    return aerial_image

def draw_camera(camera, points=None, log=True, points_alpha=0.5, radius=None):
    image = np.copy(camera.image)

    if not points is None:
        # Project points to camera image
        points = camera.cam_to_ego.inverse()(points)
        depths = points[:, 2]
        pixels = (camera.intr @ points.T).T
        pixels = pixels[:, :2] / pixels[:, 2:]
        pixels = pixels[:, ::-1]

        mask = np.logical_and(
            depths > 0,
            np.all(np.logical_and(0 <= pixels, pixels < np.asarray(image.shape[:2])[np.newaxis, :]), axis=1),
        )
        depths = depths[mask]
        pixels = pixels[mask]

        if depths.shape[0] > 0:
            if log:
                depths = np.log(depths)
            min_depth = np.amin(depths)
            max_depth = np.amax(depths)
            if max_depth > min_depth:
                depths = (depths - min_depth) / (max_depth - min_depth)
            else:
                depths = depths * 0 + 0.5

            # Create coordinates of single circle
            if radius is None:
                radius = max(1, int(min(image.shape[0], image.shape[1]) * 0.008))
            radius = int(radius)
            xxc, yyc = np.mgrid[:2 * radius, :2 * radius]
            circle = (xxc - radius) ** 2 + (yyc - radius) ** 2 < radius ** 2
            xxc = xxc[circle] - radius
            yyc = yyc[circle] - radius

            pixels_per_circle = xxc.shape[0]
            points_num = pixels.shape[0]

            # Add circle coordinates to every point
            color = cv2.applyColorMap(np.asarray([depths * 255.0]).astype("uint8"), cv2.COLORMAP_JET)[0, :, ::-1]
            color = np.repeat(color, pixels_per_circle, axis=0)
            pixels = np.repeat(pixels, pixels_per_circle, axis=0) + np.tile(np.stack([xxc, yyc], axis=1), [points_num, 1])

            # Filter out of bounds pixels
            mask = np.all(np.logical_and(0 <= pixels, pixels < np.asarray(image.shape[:2])[np.newaxis, :]), axis=1)
            color = color[mask, :]
            pixels = pixels[mask, :]

            # Apply color
            pixels = pixels.astype("int")
            image[pixels[:, 0], pixels[:, 1]] = np.clip(points_alpha * color + (1.0 - points_alpha) * image[pixels[:, 0], pixels[:, 1]], 0.0, 255.0).astype("uint8")
    return image

def draw_aerial_probs(probs, mask, aerial_image, alpha=0.5, normalize=True):
    offset = np.asarray(aerial_image.shape[:2]).astype("float32") / 2
    colormap = cv2.applyColorMap(np.asarray([np.arange(256).astype("uint8")]), cv2.COLORMAP_JET)[0, :, ::-1] # 256, 3

    if normalize:
        probs = probs / np.amax(probs)
    padding = (np.asarray(probs.shape[:2]) - np.asarray(aerial_image.shape[:2])) // 2
    assert np.all(padding >= 0) or np.all(padding <= 0)
    image = colormap[(probs * 255.0).astype("uint8")]
    if padding[0] > 0:
        image = image[padding[0]:image.shape[0] - padding[0], padding[1]:image.shape[1] - padding[1]]
        mask = mask[padding[0]:mask.shape[0] - padding[0], padding[1]:mask.shape[1] - padding[1]]
    else:
        padding = -padding
        image = np.pad(image, [[padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], mode="constant", constant_values=0.0)
        mask = np.pad(mask, [[padding[0], padding[0]], [padding[1], padding[1]]], mode="constant", constant_values=False)

    image = ((1 - alpha) * aerial_image + alpha * image).astype("uint8")
    image = np.where(mask[:, :, np.newaxis], image, aerial_image)

    return image
