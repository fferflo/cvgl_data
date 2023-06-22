import os, cvgl_data
from collections import defaultdict
import numpy as np
import concurrent.futures, os, yaml

def load_tiledwebmaps(tileloader, name, zoom):
    return cvgl_data.TiledWebMapsLoader(tileloader, name, zoom)

def load(path, cam_ops=[], lidar_ops=[], updates=None, as_dict=False, threads=None):
    if threads is None:
        threads = os.cpu_count()

    def make_loader(scene_path):
        with open(os.path.join(scene_path, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)

        return cvgl_data.FrameLoader(
            scene_path,
            cam_ops=cam_ops,
            lidar_ops=lidar_ops,
            updates=[os.path.join(p, config["dataset"], os.path.basename(scene_path)) for p in updates] if not updates is None else [],
        )

    result = [os.path.join(path, f) for f in os.listdir(path)]
    result = [p for p in result if os.path.isfile(os.path.join(p, "config.yaml"))]
    if threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            result = list(executor.map(make_loader, result))
    else:
        result = [make_loader(scene_path) for scene_path in result]

    result = sorted(result, key=lambda s: s.scene_name)

    license_path = os.path.join(path, "LICENSE")
    if os.path.isfile(license_path):
        with open(license_path, "r") as f:
            license = f.read()
        license = license.strip(" \n\r\t")
        print(f"Loading {result[0].dataset} with license: {license}")

    if as_dict:
        result2 = defaultdict(list)
        for scene in result:
            result2[scene.location].append(scene)
        return dict(result2)
    else:
        return result


def get_intersect_range(all_timestamps):
    max_start = np.amax([ts[0] for ts in all_timestamps])
    min_end = np.amax([ts[-1] for ts in all_timestamps])
    return max_start, min_end

def intersect_timestamps(timestamps, all_timestamps):
    max_start, min_end = get_intersect_range(all_timestamps)
    return timestamps[np.logical_and(max_start <= timestamps, timestamps <= min_end)]

def get_all_timestamps(node):
    timestamps = []
    if isinstance(node, cvgl_data.NamedDataLoader):
        for node2 in node.values():
            timestamps.extend(get_all_timestamps(node2))
    else:
        assert isinstance(node, cvgl_data.Loader)
        timestamps.append(node.timestamps)
    return timestamps

def unwrap(func):
    return lambda args: func(*args)